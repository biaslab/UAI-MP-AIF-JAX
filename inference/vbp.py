"""Value Belief Propagation (ε→0) planning via value iteration.

In the ε→0 limit, VBP reduces to standard value iteration:
- Backward: V(x) = max_a Q(x,a) where Q(x,a) = E_{x'|x,a}[V(x')]
- Action selection: greedy policy weighted by state belief

All computation in log-space. Accepts probability tensors, logs once at top.
θ is marginalized once upfront (reduced strategy).
"""

import jax
import jax.numpy as jnp
from jax import nn, lax
from jax.scipy.special import logsumexp
from functools import partial

from .planning import safe_log, marginalize_static, LOG_ZERO


@partial(jax.jit, static_argnums=(4, 5))
def vbp_planning(
    q_current_state: jnp.ndarray,
    q_static_state: jnp.ndarray,
    transition_tensor: jnp.ndarray,
    goal: jnp.ndarray,
    horizon: int,
    n_iterations: int = 1,
) -> jnp.ndarray:
    """
    Plan actions via Value Belief Propagation (ε→0 limit). JIT-compiled.

    In the ε→0 limit, backward messages correspond to the value function
    (max over actions) and the policy is deterministic (argmax). θ is
    marginalized once upfront.

    Args:
        q_current_state: (n_states,) current belief over state
        q_static_state: (n_static,) belief over static configuration
        transition_tensor: (n_states, n_states, n_static, n_actions) probability tensor
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: unused, kept for interface consistency (always 1)

    Returns:
        action_dist: (n_actions,) distribution over first action
    """
    n_states = q_current_state.shape[0]
    n_actions = transition_tensor.shape[3]

    # Log once at the top
    log_T = safe_log(transition_tensor)
    log_reduced = marginalize_static(log_T, safe_log(q_static_state))
    # log_reduced: (x_new, x_old, a)

    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    # Backward pass: value iteration
    # V[T] = goal (terminal value)
    # Q(x, a) = logsumexp_{x'} [log p(x'|x,a) + log V(x')]
    # V(x) = max_a Q(x, a)
    log_V = jnp.full((horizon + 1, n_states), LOG_ZERO)
    log_V = log_V.at[horizon].set(log_goal)
    log_Q = jnp.full((horizon, n_states, n_actions), LOG_ZERO)

    def backward_body(rev_t, carry):
        log_V, log_Q = carry
        t = horizon - 1 - rev_t
        # Q(x, a) = log E_{x'|x,a}[V(x')]
        log_Q_t = logsumexp(log_reduced + log_V[t + 1][:, None, None], axis=0)
        log_Q = log_Q.at[t].set(log_Q_t)
        # V(x) = max_a Q(x, a)
        log_V = log_V.at[t].set(jnp.max(log_Q_t, axis=-1))
        return log_V, log_Q

    log_V, log_Q = lax.fori_loop(0, horizon, backward_body, (log_V, log_Q))

    # Action selection: greedy policy weighted by state belief
    # For each state, find the best action
    best_actions = jnp.argmax(log_Q[0], axis=-1)  # (n_states,)
    # action_dist[a] = Σ_x p(x_0) * δ(a, argmax_a' Q(x, a'))
    q0_probs = nn.softmax(log_q0)
    action_dist = jnp.zeros(n_actions).at[best_actions].add(q0_probs)

    return action_dist
