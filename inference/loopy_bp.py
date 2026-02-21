"""Loopy BP planning with θ as a variable in the temporal factor graph.

Instead of marginalizing θ out once upfront, θ participates as a full variable
connected to each transition factor via equality constraints. Iterative message
passing refines the belief on θ while planning, using cavity messages to avoid
double-counting.

All internal computation is in log-space.

Factor graph:

    p(θ)─θ₀──(=)──θ₁──(=)──θ₂── ··· ──(=)──θ_{T-1}
          |         |         |                 |
    p(x₀)─x₀─[dyn₀]─x₁─[dyn₁]─x₂─ ··· ─[dyn_{T-1}]─x_T─goal
                  |         |                    |
                 u₀        u₁                 u_{T-1}
                  |         |                    |
               p(u₀)     p(u₁)              p(u_{T-1})
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .planning import LOG_ZERO, safe_log


def compute_reduced_per_t(log_T, log_cavity_theta):
    """
    Compute per-timestep reduced transition tensors using θ cavity messages.

    Args:
        log_T: (x_new, x_old, θ, u) log-space transition tensor
        log_cavity_theta: (T, θ) log-space per-timestep cavity beliefs

    Returns:
        (T, x_new, x_old, u) log-space per-timestep reduced tensors
    """
    return logsumexp(
        log_T[None] + log_cavity_theta[:, None, None, :, None], axis=3
    )


def forward_pass(log_reduced_per_t, log_q_x0, log_action_prior, horizon):
    """
    Forward pass: propagate state beliefs through time using per-timestep tensors.

    Args:
        log_reduced_per_t: (T, x_new, x_old, u)
        log_q_x0: (n_states,) log initial state belief
        log_action_prior: (n_actions,)
        horizon: T

    Returns:
        log_fwd_msgs: (T+1, n_states) log forward messages
    """
    n_states = log_q_x0.shape[0]
    log_fwd = jnp.zeros((horizon + 1, n_states))
    log_fwd = log_fwd.at[0].set(log_q_x0)

    def body_fn(t, log_fwd):
        log_terms = (log_reduced_per_t[t]
                     + log_fwd[t][None, :, None]
                     + log_action_prior[None, None, :])
        log_q_next = logsumexp(log_terms, axis=(1, 2))
        log_q_next = log_q_next - logsumexp(log_q_next)
        return log_fwd.at[t + 1].set(log_q_next)

    return lax.fori_loop(0, horizon, body_fn, log_fwd)


def backward_pass(log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior, horizon):
    """
    Backward pass: propagate goal backward, compute action marginals.

    Args:
        log_reduced_per_t: (T, x_new, x_old, u)
        log_fwd_msgs: (T+1, n_states)
        log_goal: (n_states,)
        log_action_prior: (n_actions,)
        horizon: T

    Returns:
        log_bwd_msgs: (T+1, n_states)
        q_u: (T, n_actions) action marginals (probability space)
    """
    n_states = log_goal.shape[0]
    n_actions = log_action_prior.shape[0]
    log_bwd = jnp.zeros((horizon + 1, n_states))
    log_bwd = log_bwd.at[horizon].set(log_goal)
    q_u = jnp.zeros((horizon, n_actions))

    def body_fn(carry, t):
        log_bwd, q_u = carry

        # Action marginal
        log_terms = (log_reduced_per_t[t]
                     + log_bwd[t + 1][:, None, None]
                     + log_fwd_msgs[t][None, :, None])
        log_msg_to_u = logsumexp(log_terms, axis=(0, 1))
        q_u_t = jax.nn.softmax(log_msg_to_u + log_action_prior)
        q_u = q_u.at[t].set(q_u_t)

        # Backward message to x_t
        log_terms_bwd = (log_reduced_per_t[t]
                         + log_bwd[t + 1][:, None, None]
                         + log_action_prior[None, None, :])
        log_bwd_t = logsumexp(log_terms_bwd, axis=(0, 2))
        log_bwd_t = log_bwd_t - logsumexp(log_bwd_t)
        log_bwd = log_bwd.at[t].set(log_bwd_t)

        return (log_bwd, q_u), None

    (log_bwd, q_u), _ = lax.scan(
        body_fn, (log_bwd, q_u), jnp.arange(horizon - 1, -1, -1)
    )

    return log_bwd, q_u


def compute_dyn_to_theta_msgs(log_T, log_fwd_msgs, log_bwd_msgs, log_action_prior, horizon):
    """
    Compute messages from each dynamics factor to its θ variable.

    log_T: (x_new, x_old, θ, u)

    Returns:
        log_dyn_to_theta: (T, n_static) log-space messages
    """
    log_fwd_t = log_fwd_msgs[:-1][:, None, :, None, None]
    log_bwd_t1 = log_bwd_msgs[1:][:, :, None, None, None]

    terms = (log_T[None]
             + log_fwd_t
             + log_bwd_t1
             + log_action_prior[None, None, None, None, :])
    return logsumexp(terms, axis=(1, 2, 4))


def compute_theta_cavities(log_prior_theta, log_dyn_to_theta):
    """
    Compute log-space cavity messages for θ via forward-backward on the equality chain.

    Returns:
        log_cavity_theta: (T, n_static) log-space normalized cavity beliefs
    """
    n_static = log_dyn_to_theta.shape[1]

    log_dyn_cumsum = jnp.cumsum(log_dyn_to_theta, axis=0)
    fwd_exclusive = jnp.concatenate([
        jnp.zeros((1, n_static)),
        log_dyn_cumsum[:-1],
    ], axis=0) + log_prior_theta[None, :]

    log_dyn_rev_cumsum = jnp.cumsum(log_dyn_to_theta[::-1], axis=0)[::-1]
    bwd_exclusive = jnp.concatenate([
        log_dyn_rev_cumsum[1:],
        jnp.zeros((1, n_static)),
    ], axis=0)

    log_cavity = fwd_exclusive + bwd_exclusive
    log_cavity = log_cavity - logsumexp(log_cavity, axis=1, keepdims=True)
    return log_cavity


@partial(jax.jit, static_argnums=(4, 5))
def loopy_bp_planning(
    q_current_state,    # (n_states,)
    q_static_state,     # (n_static,) prior on θ
    transition_tensor,  # (n_states, n_states, n_static, n_actions) probability tensor
    goal,               # (n_states,)
    horizon,            # int (static)
    n_iterations,       # int (static)
    action_prior=None,  # (n_actions,) prior over actions. If None, uniform.
) -> jnp.ndarray:
    """
    Plan actions via loopy BP with θ as a variable in the factor graph.

    Args:
        q_current_state: (n_states,) current belief over dynamic state
        q_static_state: (n_static,) prior belief over static configuration θ
        transition_tensor: (n_states, n_states, n_static, n_actions) probability tensor
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: number of loopy BP iterations (static for JIT)
        action_prior: (n_actions,) prior over actions. If None, uniform.

    Returns:
        action_dist: (n_actions,) distribution over first action
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]

    # Log once at the top
    log_T = safe_log(transition_tensor)
    log_prior_theta = safe_log(q_static_state)
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    # Initialize: cavity = prior
    log_cavity_theta = jnp.tile(log_prior_theta, (horizon, 1))
    q_u_init = jnp.zeros((horizon, n_actions))

    def body_fn(_, carry):
        log_cavity_theta, _ = carry

        log_reduced_per_t = compute_reduced_per_t(log_T, log_cavity_theta)
        log_fwd = forward_pass(log_reduced_per_t, log_q0, log_action_prior, horizon)
        log_bwd, q_u = backward_pass(
            log_reduced_per_t, log_fwd, log_goal, log_action_prior, horizon
        )
        log_dyn_to_theta = compute_dyn_to_theta_msgs(
            log_T, log_fwd, log_bwd, log_action_prior, horizon
        )
        new_log_cavity = compute_theta_cavities(log_prior_theta, log_dyn_to_theta)

        return new_log_cavity, q_u

    log_cavity_theta, q_u = lax.fori_loop(
        0, n_iterations, body_fn, (log_cavity_theta, q_u_init)
    )

    return q_u[0]
