"""Message passing primitives for discrete factor graphs."""

import jax
import jax.numpy as jnp
from jax import nn

EPSILON = 1e-10


# =============================================================================
# Index-based message passing (memory-efficient)
# =============================================================================


def forward_message_indexed(
    next_state_idx: jnp.ndarray,
    q_old: jnp.ndarray,
    q_static: jnp.ndarray,
    action_idx: int,
    n_states: int,
) -> jnp.ndarray:
    """
    Forward message using index-based transition representation.
    
    Instead of einsum("ijkl,j,k,l->i", T, q_old, q_static, action),
    we use scatter-add which avoids large intermediate tensors.
    
    Args:
        next_state_idx: (n_states, n_static, n_actions) -> new_state index
        q_old: (n_states,) belief over old state
        q_static: (n_static,) belief over static state
        action_idx: scalar action index
        n_states: number of states (for output size)
        
    Returns:
        msg: (n_states,) unnormalized forward message
    """
    # Select the transition for this action: (n_states, n_static) -> new_state
    next_idx = next_state_idx[:, :, action_idx]
    
    # Compute outer product of beliefs (small: n_states × n_static)
    weights = jnp.outer(q_old, q_static)
    
    # Scatter-add: for each (old, static), add weight to msg[next_idx[old, static]]
    msg = jnp.zeros(n_states).at[next_idx.ravel()].add(weights.ravel())
    
    return msg / (msg.sum() + EPSILON)


def backward_obs_message_indexed(
    obs_idx: jnp.ndarray,
    vision_obs: jnp.ndarray,
    q_static: jnp.ndarray,
) -> jnp.ndarray:
    """
    Backward observation message using index-based representation.
    
    For each FOV position, gather the probability of the correct cell type.
    
    Args:
        obs_idx: (7, 7, n_states, n_static) -> cell_type index (uint8)
        vision_obs: (7, 7, 11) soft observation (probability per cell type)
        q_static: (n_static,) belief over static state
        
    Returns:
        msg: (n_states,) unnormalized message (log-space summed, then exp)
    """
    n_states = obs_idx.shape[2]
    n_static = obs_idx.shape[3]
    
    # Reshape for vectorized gather
    obs_idx_flat = obs_idx.reshape(49, n_states, n_static)  # (49, n_states, n_static)
    vision_flat = vision_obs.reshape(49, 11)  # (49, 11)
    
    # For each FOV position, gather P(observed | state, static)
    # likelihood[fov, state, static] = vision_flat[fov, obs_idx_flat[fov, state, static]]
    def gather_likelihood(fov_idx):
        # obs_idx_flat[fov_idx]: (n_states, n_static) cell type indices
        # vision_flat[fov_idx]: (11,) probabilities
        cell_types = obs_idx_flat[fov_idx]  # (n_states, n_static)
        probs = vision_flat[fov_idx]  # (11,)
        return probs[cell_types]  # (n_states, n_static)
    
    # Vectorize over FOV positions
    likelihoods = jax.vmap(gather_likelihood)(jnp.arange(49))  # (49, n_states, n_static)
    
    # Product over FOV positions (sum in log space for numerical stability)
    log_likelihood = jnp.log(likelihoods + EPSILON).sum(axis=0)  # (n_states, n_static)
    
    # Marginalize over static
    log_msg = jax.scipy.special.logsumexp(
        log_likelihood + jnp.log(q_static + EPSILON)[None, :],
        axis=1
    )
    
    return log_msg  # Return log-space message


def backward_obs_message_to_static_indexed(
    obs_idx: jnp.ndarray,
    vision_obs: jnp.ndarray,
    q_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Backward observation message to static variable using index-based representation.
    
    Args:
        obs_idx: (7, 7, n_states, n_static) -> cell_type index (uint8)
        vision_obs: (7, 7, 11) soft observation
        q_state: (n_states,) belief over state
        
    Returns:
        log_msg: (n_static,) log-space message
    """
    n_states = obs_idx.shape[2]
    n_static = obs_idx.shape[3]
    
    obs_idx_flat = obs_idx.reshape(49, n_states, n_static)
    vision_flat = vision_obs.reshape(49, 11)
    
    def gather_likelihood(fov_idx):
        cell_types = obs_idx_flat[fov_idx]
        probs = vision_flat[fov_idx]
        return probs[cell_types]
    
    likelihoods = jax.vmap(gather_likelihood)(jnp.arange(49))
    log_likelihood = jnp.log(likelihoods + EPSILON).sum(axis=0)  # (n_states, n_static)
    
    # Marginalize over state
    log_msg = jax.scipy.special.logsumexp(
        log_likelihood + jnp.log(q_state + EPSILON)[:, None],
        axis=0
    )
    
    return log_msg


def backward_ori_message_indexed(
    ori_idx: jnp.ndarray,
    ori_obs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Backward orientation message using index-based representation.
    
    Args:
        ori_idx: (n_states,) -> orientation index for each state
        ori_obs: (4,) soft orientation observation
        
    Returns:
        msg: (n_states,) message (probability of observed orientation for each state)
    """
    # For each state, get the probability of its orientation being observed
    return ori_obs[ori_idx]


def transition_message_to_static_indexed(
    next_state_idx: jnp.ndarray,
    q_new: jnp.ndarray,
    q_old: jnp.ndarray,
    action_idx: int,
) -> jnp.ndarray:
    """
    Message from transition factor to static variable using indices.
    
    Computes: sum over (new, old) of T[new, old, static, action] * q_new[new] * q_old[old]
    
    Since T is one-hot in 'new', this becomes:
    msg[static] = sum over old of q_new[next_idx[old, static, action]] * q_old[old]
    
    Args:
        next_state_idx: (n_states, n_static, n_actions) -> new_state index
        q_new: (n_states,) belief over new state
        q_old: (n_states,) belief over old state
        action_idx: scalar action index
        
    Returns:
        msg: (n_static,) unnormalized message
    """
    # next_idx[old, static] = new state for this action
    next_idx = next_state_idx[:, :, action_idx]  # (n_old, n_static)
    
    # For each (old, static), get q_new[next_idx[old, static]] * q_old[old]
    # q_new_gathered[old, static] = q_new[next_idx[old, static]]
    q_new_gathered = q_new[next_idx]  # (n_old, n_static)
    
    # Weighted sum over old states
    msg = jnp.einsum("os,o->s", q_new_gathered, q_old)
    
    return msg


# =============================================================================
# Original tensor-based message passing (for reference)
# =============================================================================


def forward_message_2d(tensor: jnp.ndarray, q_in: jnp.ndarray) -> jnp.ndarray:
    """
    Forward message through a 2D transition tensor.
    
    tensor: (n_out, n_in)
    q_in: (n_in,)
    returns: (n_out,) normalized
    """
    msg = jnp.einsum("ij,j->i", tensor, q_in)
    return msg / (msg.sum() + EPSILON)


def forward_message_3d(
    tensor: jnp.ndarray, q_in1: jnp.ndarray, q_in2: jnp.ndarray
) -> jnp.ndarray:
    """
    Forward message through a 3D transition tensor.
    
    tensor: (n_out, n_in1, n_in2)
    q_in1: (n_in1,), q_in2: (n_in2,)
    returns: (n_out,) normalized
    """
    msg = jnp.einsum("ijk,j,k->i", tensor, q_in1, q_in2)
    return msg / (msg.sum() + EPSILON)


def forward_message_4d(
    tensor: jnp.ndarray,
    q_in1: jnp.ndarray,
    q_in2: jnp.ndarray,
    q_in3: jnp.ndarray,
) -> jnp.ndarray:
    """
    Forward message through a 4D transition tensor.
    
    tensor: (n_out, n_in1, n_in2, n_in3)
    q_in1: (n_in1,), q_in2: (n_in2,), q_in3: (n_in3,)
    returns: (n_out,) normalized
    """
    msg = jnp.einsum("ijkl,j,k,l->i", tensor, q_in1, q_in2, q_in3)
    return msg / (msg.sum() + EPSILON)


def backward_message_2d(
    tensor: jnp.ndarray, obs_onehot: jnp.ndarray
) -> jnp.ndarray:
    """
    Backward message from clamped observation through 2D tensor.
    
    tensor: (n_obs, n_state)
    obs_onehot: (n_obs,)
    returns: (n_state,) unnormalized (will be combined with other messages)
    """
    return jnp.einsum("ij,i->j", tensor, obs_onehot)


def backward_message_3d(
    tensor: jnp.ndarray, obs_onehot: jnp.ndarray, q_other: jnp.ndarray
) -> jnp.ndarray:
    """
    Backward message from clamped observation through 3D tensor.
    
    tensor: (n_obs, n_state, n_other)
    obs_onehot: (n_obs,)
    q_other: (n_other,)
    returns: (n_state,) unnormalized
    """
    return jnp.einsum("ijk,i,k->j", tensor, obs_onehot, q_other)


def backward_message_to_other_3d(
    tensor: jnp.ndarray, obs_onehot: jnp.ndarray, q_state: jnp.ndarray
) -> jnp.ndarray:
    """
    Backward message to the 'other' variable through 3D tensor.
    
    tensor: (n_obs, n_state, n_other)
    obs_onehot: (n_obs,)
    q_state: (n_state,)
    returns: (n_other,) unnormalized
    """
    return jnp.einsum("ijk,i,j->k", tensor, obs_onehot, q_state)


def combine_messages(messages: list[jnp.ndarray]) -> jnp.ndarray:
    """
    Combine multiple messages into a posterior via normalized product.
    
    messages: list of (n,) arrays (same shape)
    returns: (n,) normalized posterior
    """
    log_msgs = [jnp.log(msg + EPSILON) for msg in messages]
    log_q = sum(log_msgs)
    return nn.softmax(log_q)


def combine_messages_log(log_messages: list[jnp.ndarray]) -> jnp.ndarray:
    """
    Combine log-space messages into a posterior.
    
    log_messages: list of (n,) log-probability arrays
    returns: (n,) normalized posterior
    """
    log_q = sum(log_messages)
    return nn.softmax(log_q)
