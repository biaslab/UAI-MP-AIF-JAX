"""Planning via message passing on temporal factor graph."""

import jax
import jax.numpy as jnp
from jax import nn, lax
from functools import partial

from .messages import EPSILON, forward_message_indexed


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
        transition_tensor: (n_states, n_states, n_static, n_actions)
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: number of forward-backward passes (static for JIT)
        
    Returns:
        action_dist: (n_actions,) distribution over first action
    """
    n_states = q_current_state.shape[0]
    n_actions = transition_tensor.shape[3]
    
    reduced_tensor = marginalize_static(transition_tensor, q_static_state)
    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
    
    q_u = jnp.tile(action_prior, (horizon, 1))
    q_state = jnp.concatenate([
        q_current_state[None, :],
        jnp.ones((horizon, n_states)) / n_states
    ], axis=0)
    
    def body_fn(_, carry):
        q_state, q_u = carry
        q_state_new = forward_pass(reduced_tensor, q_state, action_prior, horizon)
        q_u_new = backward_pass(reduced_tensor, q_state_new, goal, action_prior, horizon)
        return q_state_new, q_u_new

    q_state, q_u = lax.fori_loop(0, n_iterations, body_fn, (q_state, q_u))

    return q_u[0]


def marginalize_static(
    transition_tensor: jnp.ndarray, q_static: jnp.ndarray
) -> jnp.ndarray:
    """Marginalize out static_state from transition tensor."""
    return jnp.einsum("ijkl,k->ijl", transition_tensor, q_static)


def forward_pass(
    reduced_tensor: jnp.ndarray,
    q_state: jnp.ndarray,
    action_prior: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    """
    Forward pass: propagate state beliefs through time.

    The BP message μ_{u→dyn_t} = p(u) since u_t only connects to p(u_t) and dyn_t.

    reduced_tensor: (n_states, n_states, n_actions)
    q_state: (T+1, n_states) beliefs
    action_prior: (n_actions,) prior over actions
    horizon: T
    """
    def body_fn(t, q_state):
        q_next = jnp.einsum("ijk,j,k->i", reduced_tensor, q_state[t], action_prior)
        q_next = q_next / (q_next.sum() + EPSILON)
        return q_state.at[t + 1].set(q_next)

    return lax.fori_loop(0, horizon, body_fn, q_state)


def backward_pass(
    reduced_tensor: jnp.ndarray,
    q_state: jnp.ndarray,
    goal: jnp.ndarray,
    action_prior: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    """
    Backward pass: propagate goal constraint backward to actions.
    
    reduced_tensor: (n_states, n_states, n_actions)
    q_state: (T+1, n_states) beliefs from forward pass
    goal: (n_states,) goal distribution
    action_prior: (n_actions,) prior over actions
    horizon: T
    """
    def body_fn(carry, t):
        backward_msg, q_u = carry
        
        msg_to_u = jnp.einsum("ijk,i,j->k", reduced_tensor, backward_msg, q_state[t])
        q_u_t = msg_to_u * action_prior
        q_u_t = q_u_t / (q_u_t.sum() + EPSILON)
        q_u = q_u.at[t].set(q_u_t)
        
        backward_msg = jnp.einsum("ijk,i,k->j", reduced_tensor, backward_msg, action_prior)
        backward_msg = backward_msg / (backward_msg.sum() + EPSILON)
        
        return (backward_msg, q_u), None
    
    n_actions = action_prior.shape[0]
    q_u_init = jnp.zeros((horizon, n_actions))
    (_, q_u), _ = lax.scan(body_fn, (goal, q_u_init), jnp.arange(horizon - 1, -1, -1))
    
    return q_u


# =============================================================================
# Index-based planning (memory-efficient)
# =============================================================================


@partial(jax.jit, static_argnums=(4, 5))
def planning_indexed(
    q_current_state: jnp.ndarray,
    q_static_state: jnp.ndarray,
    transition_idx: jnp.ndarray,
    goal: jnp.ndarray,
    horizon: int,
    n_iterations: int = 1,
) -> jnp.ndarray:
    """
    Plan actions via forward-backward message passing using index-based tensors.
    
    Args:
        q_current_state: (n_states,) current belief over state
        q_static_state: (n_static,) belief over static configuration
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: number of forward-backward passes (static for JIT)
        
    Returns:
        action_dist: (n_actions,) distribution over first action
    """
    n_states = q_current_state.shape[0]
    n_actions = transition_idx.shape[2]
    
    # Marginalize static using scatter-add (avoids huge intermediate)
    reduced_tensor = marginalize_static_indexed(transition_idx, q_static_state, n_states)
    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
    
    q_u = jnp.tile(action_prior, (horizon, 1))
    q_state = jnp.concatenate([
        q_current_state[None, :],
        jnp.ones((horizon, n_states)) / n_states
    ], axis=0)
    
    def body_fn(_, carry):
        q_state, q_u = carry
        q_state_new = forward_pass(reduced_tensor, q_state, action_prior, horizon)
        q_u_new = backward_pass(reduced_tensor, q_state_new, goal, action_prior, horizon)
        return q_state_new, q_u_new

    q_state, q_u = lax.fori_loop(0, n_iterations, body_fn, (q_state, q_u))

    return q_u[0]


def marginalize_static_indexed(
    transition_idx: jnp.ndarray, 
    q_static: jnp.ndarray, 
    n_states: int
) -> jnp.ndarray:
    """
    Marginalize out static_state from index-based transition tensor.
    
    Computes: reduced[new, old, action] = sum over static of 
              I[transition_idx[old, static, action] == new] * q_static[static]
    
    This uses scatter-add to avoid materializing the full transition tensor.
    
    Args:
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        q_static: (n_static,) belief over static state
        n_states: number of states
        
    Returns:
        reduced: (n_states, n_states, n_actions) marginalized transition
    """
    n_old = transition_idx.shape[0]
    n_static = transition_idx.shape[1]
    n_actions = transition_idx.shape[2]
    
    # For each action, scatter-add q_static weights based on transition indices
    def marginalize_action(action):
        # transition_idx[:, :, action] has shape (n_old, n_static)
        # For each (old, static), add q_static[static] to result[next_idx[old,static], old]
        next_idx = transition_idx[:, :, action]  # (n_old, n_static)
        
        # Create index arrays for scatter
        old_idx = jnp.arange(n_old)[:, None]  # (n_old, 1)
        old_idx = jnp.broadcast_to(old_idx, (n_old, n_static))  # (n_old, n_static)
        
        # Weights are q_static broadcast over old states
        weights = jnp.broadcast_to(q_static[None, :], (n_old, n_static))  # (n_old, n_static)
        
        # Scatter-add: result[next_idx[o,s], o] += weights[o,s]
        result = jnp.zeros((n_states, n_old))
        result = result.at[next_idx.ravel(), old_idx.ravel()].add(weights.ravel())
        
        return result
    
    # Vectorize over actions
    reduced = jax.vmap(marginalize_action)(jnp.arange(n_actions))  # (n_actions, n_states, n_old)
    
    # Transpose to (n_states, n_old, n_actions)
    return jnp.transpose(reduced, (1, 2, 0))
