"""Reduced region-extended planning with fixed θ.

Like region_extended_loopy_bp.py but treats θ as known (fixed at q_static_state).
Skips all θ backward messages and cavity computation, but keeps observation factors
and kernel reparametrization.

All internal computation is in log-space. Accepts probability-space tensors.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .planning import LOG_ZERO, safe_log
from .messages import safe_log_div
from .region_extended_loopy_bp import (
    compute_log_reduced,
    forward_pass,
    backward_pass,
    compute_obs_to_x_msgs,
    compute_dyn_region_beliefs,
    compute_obs_region_beliefs,
    compute_dyn_channels,
    compute_obs_channels,
    anneal_log_channel,
)
from environments.minigrid import N_CELL_TYPES


@partial(jax.jit, static_argnums=(5, 6, 7))
def reduced_region_extended_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta (used as fixed cavity)
    transition_tensor,    # (n_states, n_states, n_static, n_actions)
    observation_tensor,   # (fov_w, fov_h, N_CELL_TYPES, n_states, n_static)
    goal,                 # (n_states,)
    horizon,              # int (static)
    n_iterations,         # int (static)
    anneal=False,         # bool (static) - anneal channel influence over iterations
) -> jnp.ndarray:
    """
    Plan actions via reduced region-extended BP with fixed θ.

    Same as region_extended_loopy_bp_planning but θ is fixed at q_static_state.
    Kernel reparametrization still evolves across iterations.

    Returns:
        action_dist: (n_actions,)
        log_dyn_channels: (T, n_states, n_states, n_actions)
        log_obs_channels: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]
    fov_w, fov_h = observation_tensor.shape[0], observation_tensor.shape[1]
    n_fov = fov_w * fov_h

    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
    log_action_prior = safe_log(action_prior)

    # Log once at top
    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)  # (x_old, x_new, θ, u)
    log_B_flat = safe_log(observation_tensor.reshape(n_fov, N_CELL_TYPES, n_states, n_static))
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    # Fixed cavities: θ = q_static_state tiled over time (log-space, normalized)
    log_cavity_fixed = safe_log(q_static_state)
    log_cavity_dyn = jnp.tile(log_cavity_fixed, (horizon, 1))
    log_cavity_obs = jnp.tile(log_cavity_fixed, (horizon + 1, 1))

    # Initialize carry
    q_u_init = jnp.zeros((horizon, n_actions))
    log_dyn_channels_init = jnp.zeros((horizon, n_states, n_states, n_actions))
    log_obs_channels_init = jnp.zeros((horizon + 1, n_fov, N_CELL_TYPES, n_states, n_static))

    def body_fn(i, carry):
        q_u, log_dyn_channels, log_obs_channels = carry

        # Inline kernels (with optional channel annealing)
        if anneal and n_iterations > 1:
            alpha = i / (n_iterations - 1)
            scaled_dyn = anneal_log_channel(log_dyn_channels, alpha, cond_axis=2)
            scaled_obs = anneal_log_channel(log_obs_channels, alpha, cond_axis=2)
        else:
            scaled_dyn = log_dyn_channels
            scaled_obs = log_obs_channels
        log_dyn_kernels = safe_log_div(log_T_kernel[None], scaled_dyn[:, :, :, None, :])
        log_obs_kernels = safe_log_div(log_B_flat[None], scaled_obs)

        # Reduced tensors
        log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

        # obs->x messages
        log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs)

        # Forward pass
        log_fwd_msgs = forward_pass(
            log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon
        )

        # Backward pass + action marginals
        log_bwd_msgs, q_u = backward_pass(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
            log_obs_to_x, horizon
        )

        # Region beliefs (using FIXED cavities)
        log_dyn_regions = compute_dyn_region_beliefs(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_dyn, log_action_prior
        )
        log_obs_regions = compute_obs_region_beliefs(
            log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_obs
        )

        # Channels from region beliefs
        new_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
        new_log_obs_channels = compute_obs_channels(log_obs_regions)

        return q_u, new_log_dyn_channels, new_log_obs_channels

    q_u, log_dyn_channels, log_obs_channels = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, log_dyn_channels_init, log_obs_channels_init)
    )

    return q_u[0], log_dyn_channels, log_obs_channels
