"""State inference for MiniGrid via loopy belief propagation."""

import jax
import jax.numpy as jnp
from jax import nn, lax
from functools import partial

from .messages import (
    EPSILON,
    forward_message_4d,
    backward_message_3d,
    backward_message_to_other_3d,
    backward_message_2d,
    forward_message_indexed,
    backward_obs_message_indexed,
    backward_obs_message_to_static_indexed,
    backward_ori_message_indexed,
    transition_message_to_static_indexed,
)


@partial(jax.jit, static_argnums=(8,))
def state_inference_step(
    q_old_state: jnp.ndarray,
    q_static_state: jnp.ndarray,
    transition_tensor: jnp.ndarray,
    obs_tensors: jnp.ndarray,
    ori_tensor: jnp.ndarray,
    vision_obs: jnp.ndarray,
    ori_obs: jnp.ndarray,
    action_onehot: jnp.ndarray,
    n_iterations: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run loopy BP for state inference. JIT-compiled for performance.
    
    Args:
        q_old_state: (n_states,) prior belief over previous state
        q_static_state: (n_static,) belief over static configuration
        transition_tensor: (n_states, n_states, n_static, n_actions)
        obs_tensors: (7, 7, 11, n_states, n_static) observation model
        ori_tensor: (4, n_states) orientation observation model
        vision_obs: (7, 7, 11) one-hot observations per FOV cell
        ori_obs: (4,) one-hot orientation observation
        action_onehot: (n_actions,) one-hot previous action
        n_iterations: number of loopy BP iterations (static for JIT)
        
    Returns:
        q_current_state: (n_states,) posterior over current state
        q_static_state: (n_static,) updated belief over static state
    """

    def body_fn(_, carry):
        q_current, q_static = carry
        return _single_iteration(
            q_old_state,
            q_current,
            q_static,
            transition_tensor,
            obs_tensors,
            ori_tensor,
            vision_obs,
            ori_obs,
            action_onehot,
        )
    
    init = (q_old_state, q_static_state)
    q_current, q_static = lax.fori_loop(0, n_iterations, body_fn, init)
    
    return q_current, q_static


def _single_iteration(
    q_old_state: jnp.ndarray,
    q_current: jnp.ndarray,
    q_static: jnp.ndarray,
    transition_tensor: jnp.ndarray,
    obs_tensors: jnp.ndarray,
    ori_tensor: jnp.ndarray,
    vision_obs: jnp.ndarray,
    ori_obs: jnp.ndarray,
    action_onehot: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single iteration of loopy BP. Vectorized for GPU."""
    n_states = q_old_state.shape[0]
    n_static = q_static.shape[0]
    
    msg_trans = forward_message_4d(
        transition_tensor, q_old_state, q_static, action_onehot
    )
    
    n_fov = obs_tensors.shape[0] * obs_tensors.shape[1]
    obs_flat = obs_tensors.reshape(n_fov, 11, n_states, n_static)
    vision_flat = vision_obs.reshape(n_fov, 11)
    log_msgs_vision = jax.vmap(
        lambda t, v: jnp.log(backward_message_3d(t, v, q_static) + EPSILON)
    )(obs_flat, vision_flat)
    log_msg_vision = log_msgs_vision.sum(axis=0)
    
    msg_ori = backward_message_2d(ori_tensor, ori_obs)
    
    log_q_current = (
        jnp.log(msg_trans + EPSILON)
        + log_msg_vision
        + jnp.log(msg_ori + EPSILON)
    )
    q_current_new = nn.softmax(log_q_current)
    
    log_msgs_static = jax.vmap(
        lambda t, v: jnp.log(backward_message_to_other_3d(t, v, q_current_new) + EPSILON)
    )(obs_flat, vision_flat)
    log_msg_static = log_msgs_static.sum(axis=0)
    
    msg_trans_static = jnp.einsum(
        "ijkl,i,j,l->k",
        transition_tensor,
        q_current_new,
        q_old_state,
        action_onehot,
    )
    log_msg_static = log_msg_static + jnp.log(msg_trans_static + EPSILON)
    
    q_static_new = nn.softmax(log_msg_static)
    
    return q_current_new, q_static_new


# =============================================================================
# Index-based state inference (memory-efficient)
# =============================================================================


@partial(jax.jit, static_argnums=(8,))
def state_inference_step_indexed(
    q_old_state: jnp.ndarray,
    q_static_state: jnp.ndarray,
    transition_idx: jnp.ndarray,
    obs_idx: jnp.ndarray,
    ori_idx: jnp.ndarray,
    vision_obs: jnp.ndarray,
    ori_obs: jnp.ndarray,
    action_idx: int,
    n_iterations: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run loopy BP for state inference using index-based tensors. Memory-efficient.
    
    Args:
        q_old_state: (n_states,) prior belief over previous state
        q_static_state: (n_static,) belief over static configuration
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        obs_idx: (7, 7, n_states, n_static) -> cell_type index
        ori_idx: (n_states,) -> orientation index
        vision_obs: (7, 7, 11) soft observations per FOV cell
        ori_obs: (4,) soft orientation observation
        action_idx: scalar action index (not one-hot!)
        n_iterations: number of loopy BP iterations (static for JIT)
        
    Returns:
        q_current_state: (n_states,) posterior over current state
        q_static_state: (n_static,) updated belief over static state
    """
    n_states = q_old_state.shape[0]

    def body_fn(_, carry):
        q_current, q_static = carry
        return _single_iteration_indexed(
            q_old_state,
            q_current,
            q_static,
            transition_idx,
            obs_idx,
            ori_idx,
            vision_obs,
            ori_obs,
            action_idx,
            n_states,
        )
    
    init = (q_old_state, q_static_state)
    q_current, q_static = lax.fori_loop(0, n_iterations, body_fn, init)
    
    return q_current, q_static


def _single_iteration_indexed(
    q_old_state: jnp.ndarray,
    q_current: jnp.ndarray,
    q_static: jnp.ndarray,
    transition_idx: jnp.ndarray,
    obs_idx: jnp.ndarray,
    ori_idx: jnp.ndarray,
    vision_obs: jnp.ndarray,
    ori_obs: jnp.ndarray,
    action_idx: int,
    n_states: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single iteration of loopy BP using index-based tensors."""
    
    # Forward message from transition (prediction)
    msg_trans = forward_message_indexed(
        transition_idx, q_old_state, q_static, action_idx, n_states
    )
    
    # Backward message from vision observations (already in log space)
    log_msg_vision = backward_obs_message_indexed(obs_idx, vision_obs, q_static)
    
    # Backward message from orientation observation
    msg_ori = backward_ori_message_indexed(ori_idx, ori_obs)
    
    # Combine messages for current state posterior
    log_q_current = (
        jnp.log(msg_trans + EPSILON)
        + log_msg_vision
        + jnp.log(msg_ori + EPSILON)
    )
    q_current_new = nn.softmax(log_q_current)
    
    # Messages to static state
    log_msg_static_obs = backward_obs_message_to_static_indexed(
        obs_idx, vision_obs, q_current_new
    )
    
    msg_trans_static = transition_message_to_static_indexed(
        transition_idx, q_current_new, q_old_state, action_idx
    )
    
    log_msg_static = log_msg_static_obs + jnp.log(msg_trans_static + EPSILON)
    q_static_new = nn.softmax(log_msg_static)
    
    return q_current_new, q_static_new
