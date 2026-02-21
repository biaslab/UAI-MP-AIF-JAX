"""Reduced dyn-channel planning with fixed theta.

Like dyn_channel_loopy_bp.py but treats theta as known (fixed at q_static_state).
Skips all theta backward messages and cavity computation. Observation factors
use the raw observation tensor (no obs channels). Only dynamics factors get
kernel reparameterization.

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
    compute_dyn_channels,
    damp_log_channel,
)


@partial(jax.jit, static_argnums=(5, 6))
def reduced_dyn_channel_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta (used as fixed cavity)
    transition_tensor,    # (n_states, n_states, n_static, n_actions)
    observation_tensor,   # (n_channels, n_obs_types, n_states, n_static)
    goal,                 # (n_states,)
    horizon,              # int (static)
    n_iterations,         # int (static)
    damping=1.0,          # float - channel update damping (1.0 = no damping)
    action_prior=None,    # (n_actions,) prior over actions. If None, uniform.
) -> tuple:
    """
    Plan actions via reduced dyn-channel BP with fixed theta.

    Same as dyn_channel_loopy_bp_planning but theta is fixed at q_static_state.
    Only dynamics factors get kernel reparameterization. Observation factors
    use the raw B tensor with fixed cavity.

    Returns:
        action_dist: (n_actions,)
        log_dyn_channels: (T, n_states, n_states, n_actions)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]
    n_fov = observation_tensor.shape[0]
    n_obs_types = observation_tensor.shape[1]

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    # Log once at top
    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)  # (x_old, x_new, theta, u)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    # Fixed cavities: theta = q_static_state tiled over time
    log_cavity_fixed = safe_log(q_static_state)
    log_cavity_dyn = jnp.tile(log_cavity_fixed, (horizon, 1))
    log_cavity_obs = jnp.tile(log_cavity_fixed, (horizon + 1, 1))

    # Tile obs tensor over time: kernel = raw B (no obs channels)
    log_B_tiled = jnp.broadcast_to(
        log_B_flat[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static)
    )

    # Precompute obs->x messages (both B and cavity are constant)
    log_obs_to_x = compute_obs_to_x_msgs(log_B_tiled, log_cavity_obs)

    # Initialize carry
    q_u_init = jnp.zeros((horizon, n_actions))
    log_dyn_channels_init = jnp.zeros((horizon, n_states, n_states, n_actions))

    def body_fn(i, carry):
        q_u, log_dyn_channels = carry

        # Dyn kernels (factor / channel in log-space)
        log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])

        # Reduced tensors (using FIXED cavity)
        log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

        # Forward pass (with precomputed obs->x)
        log_fwd_msgs = forward_pass(
            log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon
        )

        # Backward pass + action marginals
        log_bwd_msgs, q_u = backward_pass(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
            log_obs_to_x, horizon
        )

        # Dyn region beliefs -> extract dyn channels -> damped update
        log_dyn_regions = compute_dyn_region_beliefs(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_dyn, log_action_prior
        )
        raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
        new_log_dyn_channels = damp_log_channel(
            log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)

        return q_u, new_log_dyn_channels

    q_u, log_dyn_channels = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, log_dyn_channels_init)
    )

    return q_u[0], log_dyn_channels
