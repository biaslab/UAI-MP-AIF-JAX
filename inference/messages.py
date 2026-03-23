"""Message passing primitives for discrete factor graphs."""

import jax
import jax.numpy as jnp
from jax import nn
from jax.scipy.special import logsumexp

EPSILON = 1e-10
LOG_ZERO = -1e12


def safe_log(x):
    """Log that maps 0 → LOG_ZERO instead of -inf."""
    return jnp.where(x > 0, jnp.log(jnp.maximum(x, 1e-30)), LOG_ZERO)


def safe_log_div(log_num, log_den):
    """log(num/den) where 0/0 and nonzero/0 → LOG_ZERO (0 in probability space)."""
    valid = (log_num > LOG_ZERO / 2) & (log_den > LOG_ZERO / 2)
    return jnp.where(valid, log_num - log_den, LOG_ZERO)


# =============================================================================
# Tensor-based message passing
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


def marginalize_static(log_T, log_q_static):
    """Marginalize out static_state from log transition tensor.

    Args:
        log_T: (n_states, n_states, n_static, n_actions) log-space
        log_q_static: (n_static,) log-space

    Returns:
        (n_states, n_states, n_actions) log-space reduced tensor
    """
    return logsumexp(log_T + log_q_static[None, None, :, None], axis=2)


def combine_messages_log(log_messages: list[jnp.ndarray]) -> jnp.ndarray:
    """
    Combine log-space messages into a posterior.
    
    log_messages: list of (n,) log-probability arrays
    returns: (n,) normalized posterior
    """
    log_q = sum(log_messages)
    return nn.softmax(log_q)
