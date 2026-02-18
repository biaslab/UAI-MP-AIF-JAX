"""Loopy VBP planning with θ as a variable in the temporal factor graph.

Like loopy_bp.py, θ participates as a full variable connected to each
transition factor via equality constraints. The key difference is VBP (ε→0):
actions use max instead of logsumexp everywhere:
  - Backward: V(x) = max_a Q(x,a)  (value iteration)
  - Forward: propagate under argmax policy
  - dyn→θ: max over u instead of logsumexp over u

All internal computation is in log-space.
"""

import jax
import jax.numpy as jnp
from jax import nn, lax
from jax.scipy.special import logsumexp
from functools import partial

from .planning import LOG_ZERO, safe_log
from .loopy_bp import compute_reduced_per_t, compute_theta_cavities


def backward_pass_vbp(log_reduced_per_t, log_goal, horizon):
    """
    Backward pass: value iteration using per-timestep reduced tensors.

    Args:
        log_reduced_per_t: (T, x_new, x_old, u)
        log_goal: (n_states,)
        horizon: T

    Returns:
        log_V: (T+1, n_states) log value function
        log_Q: (T, n_states, n_actions) log Q-values
    """
    n_states = log_goal.shape[0]
    n_actions = log_reduced_per_t.shape[3]

    log_V = jnp.full((horizon + 1, n_states), LOG_ZERO)
    log_V = log_V.at[horizon].set(log_goal)
    log_Q = jnp.full((horizon, n_states, n_actions), LOG_ZERO)

    def body_fn(rev_t, carry):
        log_V, log_Q = carry
        t = horizon - 1 - rev_t
        # Q(x, a) = logsumexp_{x'} [log_reduced(x', x, a) + log_V(x')]
        log_Q_t = logsumexp(
            log_reduced_per_t[t] + log_V[t + 1][:, None, None], axis=0
        )
        log_Q = log_Q.at[t].set(log_Q_t)
        # V(x) = max_a Q(x, a), then normalize for numerical stability
        log_V_t = jnp.max(log_Q_t, axis=-1)
        log_V_t = log_V_t - logsumexp(log_V_t)
        log_V = log_V.at[t].set(log_V_t)
        return log_V, log_Q

    return lax.fori_loop(0, horizon, body_fn, (log_V, log_Q))


def forward_pass_vbp(log_reduced_per_t, log_q_x0, log_Q, horizon):
    """
    Forward pass: propagate state beliefs under argmax policy.

    Args:
        log_reduced_per_t: (T, x_new, x_old, u)
        log_q_x0: (n_states,) log initial state belief
        log_Q: (T, n_states, n_actions) log Q-values (for argmax policy)
        horizon: T

    Returns:
        log_fwd: (T+1, n_states) log forward messages (occupancies)
    """
    n_states = log_q_x0.shape[0]
    n_actions = log_reduced_per_t.shape[3]
    log_fwd = jnp.zeros((horizon + 1, n_states))
    log_fwd = log_fwd.at[0].set(log_q_x0)

    def body_fn(t, log_fwd):
        # Argmax policy: one-hot over actions per state
        best_actions = jnp.argmax(log_Q[t], axis=-1)  # (n_states,)
        log_policy = safe_log(nn.one_hot(best_actions, n_actions))  # (n_states, u)

        # Propagate: sum over x_old and u (only best action contributes)
        log_terms = (log_reduced_per_t[t]
                     + log_fwd[t][None, :, None]
                     + log_policy[None, :, :])
        log_q_next = logsumexp(log_terms, axis=(1, 2))
        log_q_next = log_q_next - logsumexp(log_q_next)
        return log_fwd.at[t + 1].set(log_q_next)

    return lax.fori_loop(0, horizon, body_fn, log_fwd)


def compute_dyn_to_theta_msgs_vbp(log_T, log_fwd_msgs, log_V, horizon):
    """
    Compute messages from each dynamics factor to its θ variable.

    Uses max over u (VBP) instead of logsumexp over u (soft BP).

    Args:
        log_T: (x_new, x_old, θ, u)
        log_fwd_msgs: (T+1, n_states)
        log_V: (T+1, n_states) value function (backward messages)
        horizon: T

    Returns:
        log_dyn_to_theta: (T, n_static) log-space messages
    """
    log_fwd_t = log_fwd_msgs[:-1][:, None, :, None, None]
    log_V_t1 = log_V[1:][:, :, None, None, None]

    terms = (log_T[None]
             + log_fwd_t
             + log_V_t1)
    # Max over u (ε→0), then sum over x_new and x_old
    return logsumexp(jnp.max(terms, axis=4), axis=(1, 2))


@partial(jax.jit, static_argnums=(4, 5))
def loopy_vbp_planning(
    q_current_state,    # (n_states,)
    q_static_state,     # (n_static,) prior on θ
    transition_tensor,  # (n_states, n_states, n_static, n_actions) probability tensor
    goal,               # (n_states,)
    horizon,            # int (static)
    n_iterations,       # int (static)
) -> jnp.ndarray:
    """
    Plan actions via loopy VBP with θ as a variable in the factor graph.

    Like loopy_bp_planning but uses max over actions (ε→0 limit):
    value iteration backward, argmax forward, max-over-u in dyn→θ messages.

    Args:
        q_current_state: (n_states,) current belief over dynamic state
        q_static_state: (n_static,) prior belief over static configuration θ
        transition_tensor: (n_states, n_states, n_static, n_actions) probability tensor
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: number of loopy iterations (static for JIT)

    Returns:
        action_dist: (n_actions,) distribution over first action
    """
    n_states = q_current_state.shape[0]
    n_actions = transition_tensor.shape[3]

    # Log once at the top
    log_T = safe_log(transition_tensor)
    log_prior_theta = safe_log(q_static_state)
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    # Initialize: cavity = prior
    log_cavity_theta = jnp.tile(log_prior_theta, (horizon, 1))
    log_Q_init = jnp.full((horizon, n_states, n_actions), LOG_ZERO)

    def body_fn(_, carry):
        log_cavity_theta, _, _ = carry

        log_reduced_per_t = compute_reduced_per_t(log_T, log_cavity_theta)

        # Backward first (value iteration), then forward under argmax policy
        log_V, log_Q = backward_pass_vbp(log_reduced_per_t, log_goal, horizon)
        log_fwd = forward_pass_vbp(log_reduced_per_t, log_q0, log_Q, horizon)

        # Messages to θ: max over u
        log_dyn_to_theta = compute_dyn_to_theta_msgs_vbp(
            log_T, log_fwd, log_V, horizon
        )
        new_log_cavity = compute_theta_cavities(log_prior_theta, log_dyn_to_theta)

        return new_log_cavity, log_Q, log_V

    log_V_init = jnp.full((horizon + 1, n_states), LOG_ZERO)
    log_cavity_theta, log_Q, log_V = lax.fori_loop(
        0, n_iterations, body_fn, (log_cavity_theta, log_Q_init, log_V_init)
    )

    # Action selection: greedy policy weighted by belief and value
    # q(a_0) ∝ Σ_{x_0} δ(a, a*(x_0)) · P(x_0) · V(x_0)
    best_actions = jnp.argmax(log_Q[0], axis=-1)  # (n_states,)
    log_weights = log_q0 + log_V[0]
    weights = nn.softmax(log_weights)
    action_dist = jnp.zeros(n_actions).at[best_actions].add(weights)

    return action_dist
