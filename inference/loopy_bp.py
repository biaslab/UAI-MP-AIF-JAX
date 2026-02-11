"""Loopy BP planning with θ as a variable in the temporal factor graph.

Instead of marginalizing θ out once upfront, θ participates as a full variable
connected to each transition factor via equality constraints. Iterative message
passing refines the belief on θ while planning, using cavity messages to avoid
double-counting.

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
from functools import partial

from .messages import EPSILON
from .planning import marginalize_static_indexed


def compute_reduced_per_t(transition_idx, cavity_theta, n_states):
    """
    Compute per-timestep reduced transition tensors using θ cavity messages.

    Args:
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        cavity_theta: (T, n_static) per-timestep cavity beliefs on θ
        n_states: number of dynamic states

    Returns:
        (T, n_states, n_states, n_actions) per-timestep reduced tensors
    """
    def reduce_single(cavity_t):
        return marginalize_static_indexed(transition_idx, cavity_t, n_states)

    return jax.vmap(reduce_single)(cavity_theta)


def forward_pass(reduced_per_t, q_x0, action_prior, horizon):
    """
    Forward pass: propagate state beliefs through time using per-timestep tensors.

    Args:
        reduced_per_t: (T, n_states, n_states, n_actions)
        q_x0: (n_states,) initial state belief
        action_prior: (n_actions,) prior over actions
        horizon: T

    Returns:
        fwd_msgs: (T+1, n_states) forward messages
    """
    n_states = q_x0.shape[0]
    fwd_msgs = jnp.zeros((horizon + 1, n_states))
    fwd_msgs = fwd_msgs.at[0].set(q_x0)

    def body_fn(t, fwd_msgs):
        q_next = jnp.einsum("ijk,j,k->i", reduced_per_t[t], fwd_msgs[t], action_prior)
        q_next = q_next / (q_next.sum() + EPSILON)
        return fwd_msgs.at[t + 1].set(q_next)

    return lax.fori_loop(0, horizon, body_fn, fwd_msgs)


def backward_pass(reduced_per_t, fwd_msgs, goal, action_prior, horizon):
    """
    Backward pass: propagate goal backward, compute action marginals.

    Args:
        reduced_per_t: (T, n_states, n_states, n_actions)
        fwd_msgs: (T+1, n_states) forward messages
        goal: (n_states,) goal distribution
        action_prior: (n_actions,) prior over actions
        horizon: T

    Returns:
        bwd_msgs: (T+1, n_states) backward messages
        q_u: (T, n_actions) action marginals
    """
    n_states = goal.shape[0]
    n_actions = action_prior.shape[0]
    bwd_msgs = jnp.zeros((horizon + 1, n_states))
    bwd_msgs = bwd_msgs.at[horizon].set(goal)
    q_u = jnp.zeros((horizon, n_actions))

    def body_fn(carry, t):
        bwd_msgs, q_u = carry
        bwd_t1 = bwd_msgs[t + 1]

        # Action marginal: μ_{dyn_t→u_t}
        msg_to_u = jnp.einsum("ijk,i,j->k", reduced_per_t[t], bwd_t1, fwd_msgs[t])
        q_u_t = msg_to_u * action_prior
        q_u_t = q_u_t / (q_u_t.sum() + EPSILON)
        q_u = q_u.at[t].set(q_u_t)

        # Backward message to x_t
        bwd_t = jnp.einsum("ijk,i,k->j", reduced_per_t[t], bwd_t1, action_prior)
        bwd_t = bwd_t / (bwd_t.sum() + EPSILON)
        bwd_msgs = bwd_msgs.at[t].set(bwd_t)

        return (bwd_msgs, q_u), None

    (bwd_msgs, q_u), _ = lax.scan(
        body_fn, (bwd_msgs, q_u), jnp.arange(horizon - 1, -1, -1)
    )

    return bwd_msgs, q_u


def compute_dyn_to_theta_msgs(transition_idx, fwd_msgs, bwd_msgs, action_prior, horizon):
    """
    Compute messages from each dynamics factor to its θ variable.

    For timestep t:
        log_msg[θ] = log Σ_{x_t, u_t, x_{t+1}} I[T_idx[x_t, θ, u_t]==x_{t+1}]
                      · fwd_msg[t](x_t) · p(u_t) · bwd_msg[t+1](x_{t+1})

    Args:
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        fwd_msgs: (T+1, n_states) forward messages
        bwd_msgs: (T+1, n_states) backward messages
        action_prior: (n_actions,) prior over actions
        horizon: T

    Returns:
        log_dyn_to_theta: (T, n_static) log-space messages
    """
    def compute_msg_t(t):
        fwd_t = fwd_msgs[t]
        bwd_t1 = bwd_msgs[t + 1]

        # Gather backward message at transition targets
        bwd_gathered = bwd_t1[transition_idx]  # (n_states, n_static, n_actions)

        # Weight: fwd[old] * action_prior[action] * bwd[new]
        weights = fwd_t[:, None, None] * action_prior[None, None, :] * bwd_gathered

        # Sum over old states and actions → message to θ
        msg = weights.sum(axis=(0, 2))  # (n_static,)

        return jnp.log(msg + EPSILON)

    return jax.vmap(compute_msg_t)(jnp.arange(horizon))


def compute_theta_cavities(log_prior_theta, log_dyn_to_theta):
    """
    Compute cavity messages for θ via forward-backward on the equality chain.

    The θ chain has T copies connected by equality factors. Each node t receives
    a message from dyn_t. The cavity for dyn_t excludes dyn_t's own message.

    cavity[t] = prior + Σ_{s≠t} log_dyn_to_theta[s]

    Computed efficiently using prefix sums.

    Args:
        log_prior_theta: (n_static,) log prior on θ
        log_dyn_to_theta: (T, n_static) per-timestep messages from dynamics

    Returns:
        cavity_theta: (T, n_static) normalized cavity beliefs
    """
    n_static = log_dyn_to_theta.shape[1]

    # Forward exclusive prefix sum:
    #   fwd_excl[0] = log_prior
    #   fwd_excl[t] = log_prior + Σ_{s=0}^{t-1} log_dyn_to_theta[s]
    log_dyn_cumsum = jnp.cumsum(log_dyn_to_theta, axis=0)
    fwd_exclusive = jnp.concatenate([
        jnp.zeros((1, n_static)),
        log_dyn_cumsum[:-1],
    ], axis=0) + log_prior_theta[None, :]

    # Backward exclusive suffix sum:
    #   bwd_excl[T-1] = 0
    #   bwd_excl[t] = Σ_{s=t+1}^{T-1} log_dyn_to_theta[s]
    log_dyn_rev_cumsum = jnp.cumsum(log_dyn_to_theta[::-1], axis=0)[::-1]
    bwd_exclusive = jnp.concatenate([
        log_dyn_rev_cumsum[1:],
        jnp.zeros((1, n_static)),
    ], axis=0)

    # Cavity = all contributions except dyn_t's own message
    log_cavity = fwd_exclusive + bwd_exclusive
    cavity_theta = jax.nn.softmax(log_cavity, axis=1)

    return cavity_theta


@partial(jax.jit, static_argnums=(4, 5))
def loopy_bp_planning_indexed(
    q_current_state,    # (n_states,)
    q_static_state,     # (n_static,) prior on θ
    transition_idx,     # (n_states, n_static, n_actions)
    goal,               # (n_states,)
    horizon,            # int (static)
    n_iterations,       # int (static)
) -> jnp.ndarray:
    """
    Plan actions via loopy BP with θ as a variable in the factor graph.

    Each outer iteration:
      1. Compute per-timestep reduced tensors from θ cavities
      2. Forward pass on x-chain
      3. Backward pass on x-chain (+ action marginals)
      4. Compute dyn_t → θ_t messages
      5. Forward-backward on θ equality chain → new cavities

    Args:
        q_current_state: (n_states,) current belief over dynamic state
        q_static_state: (n_static,) prior belief over static configuration θ
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: number of loopy BP iterations (static for JIT)

    Returns:
        action_dist: (n_actions,) distribution over first action
    """
    n_states = q_current_state.shape[0]
    n_actions = transition_idx.shape[2]
    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])

    log_prior_theta = jnp.log(q_static_state + EPSILON)

    # Initialize: all timesteps use the prior as cavity
    cavity_theta = jnp.tile(q_static_state, (horizon, 1))  # (T, n_static)
    q_u_init = jnp.zeros((horizon, n_actions))

    def body_fn(_, carry):
        cavity_theta, _ = carry

        # Step 1: Per-timestep reduced tensors
        reduced_per_t = compute_reduced_per_t(
            transition_idx, cavity_theta, n_states
        )

        # Step 2: Forward pass
        fwd_msgs = forward_pass(reduced_per_t, q_current_state, action_prior, horizon)

        # Step 3: Backward pass + action marginals
        bwd_msgs, q_u = backward_pass(
            reduced_per_t, fwd_msgs, goal, action_prior, horizon
        )

        # Step 4: dyn_t → θ_t messages
        log_dyn_to_theta = compute_dyn_to_theta_msgs(
            transition_idx, fwd_msgs, bwd_msgs, action_prior, horizon
        )

        # Step 5: θ cavities for next iteration
        new_cavity_theta = compute_theta_cavities(log_prior_theta, log_dyn_to_theta)

        return new_cavity_theta, q_u

    cavity_theta, q_u = lax.fori_loop(
        0, n_iterations, body_fn, (cavity_theta, q_u_init)
    )

    return q_u[0]
