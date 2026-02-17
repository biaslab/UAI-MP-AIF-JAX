"""Planning via message passing on temporal factor graph.

All internal computation is in log-space to avoid numerical underflow/overflow.
Accepts probability-space tensors, logs once at the top, returns probabilities.
"""

import jax
import jax.numpy as jnp
from jax import nn, lax
from jax.scipy.special import logsumexp
from functools import partial


LOG_ZERO = -1e12


def safe_log(x):
    """Log that maps 0 → LOG_ZERO instead of -inf."""
    return jnp.where(x > 0, jnp.log(jnp.maximum(x, 1e-30)), LOG_ZERO)


@partial(jax.jit, static_argnums=(4, 5))
def planning(
    q_current_state: jnp.ndarray,
    q_static_state: jnp.ndarray,
    transition_tensor: jnp.ndarray,
    goal: jnp.ndarray,
    horizon: int,
    n_iterations: int = 1,
) -> jnp.ndarray:
    """
    Plan actions via forward-backward message passing. JIT-compiled.

    Args:
        q_current_state: (n_states,) current belief over state
        q_static_state: (n_static,) belief over static configuration
        transition_tensor: (n_states, n_states, n_static, n_actions) probability tensor
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: number of forward-backward passes (static for JIT)

    Returns:
        action_dist: (n_actions,) distribution over first action
    """
    n_states = q_current_state.shape[0]
    n_actions = transition_tensor.shape[3]

    # Log once at the top
    log_T = safe_log(transition_tensor)
    log_reduced = marginalize_static(log_T, safe_log(q_static_state))

    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
    log_action_prior = safe_log(action_prior)
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    # Initialize log-space state beliefs (positions 1..T overwritten by forward pass)
    log_q_state = jnp.concatenate([
        log_q0[None, :],
        jnp.zeros((horizon, n_states)),
    ], axis=0)

    q_u = jnp.tile(action_prior, (horizon, 1))

    def body_fn(_, carry):
        log_q_state, q_u = carry
        log_q_state_new = forward_pass(log_reduced, log_q_state, log_action_prior, horizon)
        q_u_new = backward_pass(log_reduced, log_q_state_new, log_goal, log_action_prior, horizon)
        return log_q_state_new, q_u_new

    log_q_state, q_u = lax.fori_loop(0, n_iterations, body_fn, (log_q_state, q_u))

    return q_u[0]


def marginalize_static(log_T, log_q_static):
    """Marginalize out static_state from log transition tensor.

    Args:
        log_T: (n_states, n_states, n_static, n_actions) log-space
        log_q_static: (n_static,) log-space

    Returns:
        (n_states, n_states, n_actions) log-space reduced tensor
    """
    return logsumexp(log_T + log_q_static[None, None, :, None], axis=2)


def forward_pass(log_reduced, log_q_state, log_action_prior, horizon):
    """
    Forward pass: propagate state beliefs through time in log-space.

    log_reduced: (x_new, x_old, u)
    log_q_state: (T+1, n_states) log beliefs
    log_action_prior: (n_actions,)
    horizon: T
    """
    def body_fn(t, log_q_state):
        log_terms = (log_reduced
                     + log_q_state[t][None, :, None]
                     + log_action_prior[None, None, :])
        log_q_next = logsumexp(log_terms, axis=(1, 2))
        log_q_next = log_q_next - logsumexp(log_q_next)
        return log_q_state.at[t + 1].set(log_q_next)

    return lax.fori_loop(0, horizon, body_fn, log_q_state)


def backward_pass(log_reduced, log_q_state, log_goal, log_action_prior, horizon):
    """
    Backward pass: propagate goal constraint backward to actions.

    log_reduced: (x_new, x_old, u)
    log_q_state: (T+1, n_states) log beliefs from forward pass
    log_goal: (n_states,) log goal distribution
    log_action_prior: (n_actions,)
    horizon: T

    Returns:
        q_u: (T, n_actions) action marginals (probability space)
    """
    def body_fn(carry, t):
        log_bwd, q_u = carry

        # Action marginal: sum over x_new (i) and x_old (j)
        log_terms = (log_reduced
                     + log_bwd[:, None, None]
                     + log_q_state[t][None, :, None])
        log_msg_to_u = logsumexp(log_terms, axis=(0, 1))
        q_u_t = nn.softmax(log_msg_to_u + log_action_prior)
        q_u = q_u.at[t].set(q_u_t)

        # Backward message to x_t: sum over x_new (i) and u (k)
        log_terms_bwd = (log_reduced
                         + log_bwd[:, None, None]
                         + log_action_prior[None, None, :])
        log_bwd_new = logsumexp(log_terms_bwd, axis=(0, 2))
        log_bwd_new = log_bwd_new - logsumexp(log_bwd_new)

        return (log_bwd_new, q_u), None

    n_actions = log_action_prior.shape[0]
    q_u_init = jnp.zeros((horizon, n_actions))
    (_, q_u), _ = lax.scan(body_fn, (log_goal, q_u_init), jnp.arange(horizon - 1, -1, -1))

    return q_u
