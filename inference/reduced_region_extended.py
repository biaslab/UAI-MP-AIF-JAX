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
    compute_pref_to_x_msgs,
    compute_dyn_region_beliefs,
    compute_obs_region_beliefs,
    compute_dyn_channels,
    compute_obs_channels,
    damp_log_channel,
)


@partial(jax.jit, static_argnums=(5, 6))
def reduced_region_extended_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta (used as fixed cavity)
    transition_tensor,    # (n_states, n_states, n_static, n_actions)
    observation_tensor,   # (n_channels, n_obs_types, n_states, n_static)
    goal,                 # (n_states,) or (n_states, n_static) preference
    horizon,              # int (static)
    n_iterations,         # int (static)
    damping=1.0,          # float - channel update damping (1.0 = no damping)
    action_prior=None,    # (n_actions,) prior over actions. If None, uniform.
) -> jnp.ndarray:
    """
    Plan actions via reduced region-extended BP with fixed θ.

    Same as region_extended_loopy_bp_planning but θ is fixed at q_static_state.
    Kernel reparametrization still evolves across iterations.

    When goal is 2D, preference is applied at every timestep with fixed cavity.

    Returns:
        action_dist: (n_actions,)
        log_dyn_channels: (T, n_states, n_states, n_actions)
        log_obs_channels: (T+1, n_channels, n_obs_types, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]
    n_fov = observation_tensor.shape[0]
    n_obs_types = observation_tensor.shape[1]
    has_pref = goal.ndim == 2

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    # Log once at top
    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)  # (x_old, x_new, θ, u)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)

    if has_pref:
        log_C = safe_log(goal)                            # (n_states, n_static)
        log_goal = jnp.zeros(n_states)                    # uniform terminal
    else:
        log_C = None
        log_goal = safe_log(goal)

    # Fixed cavities: θ = q_static_state tiled over time (log-space, normalized)
    log_cavity_fixed = safe_log(q_static_state)
    log_cavity_dyn = jnp.tile(log_cavity_fixed, (horizon, 1))
    log_cavity_obs = jnp.tile(log_cavity_fixed, (horizon + 1, 1))

    # Precompute factor_reduced: logsumexp_θ(log_T + log_cavity) → (x_old, x_new, u)
    # Then transpose to (x_new, x_old, u) to match compute_log_reduced output layout
    log_factor_reduced = logsumexp(
        log_T_kernel + log_cavity_fixed[None, None, :, None], axis=2
    ).transpose(1, 0, 2)  # (x_new, x_old, u)

    # Precompute pref_to_x with fixed cavity (constant across iterations)
    if has_pref:
        log_pref_to_x = compute_pref_to_x_msgs(log_C, log_cavity_obs)  # (T+1, n_states)

    # Initialize carry
    q_u_init = jnp.zeros((horizon, n_actions))

    # Initial dyn channels: r(x_new | x_old, u) from θ-marginalized transition
    log_dyn_ch0 = log_factor_reduced  # already (x_new, x_old, u)
    log_dyn_ch0 = log_dyn_ch0 - logsumexp(log_dyn_ch0, axis=0, keepdims=True)  # normalize over x_new
    log_dyn_ch0 = log_dyn_ch0.transpose(1, 0, 2)  # (x_old, x_new, u)
    log_dyn_channels_init = jnp.broadcast_to(log_dyn_ch0[None], (horizon, n_states, n_states, n_actions))

    # Initial obs channels: r(y | x, θ) = B(y | x, θ)
    log_obs_ch0 = log_B_flat - logsumexp(log_B_flat, axis=1, keepdims=True)
    log_obs_channels_init = jnp.broadcast_to(log_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static))

    def body_fn(i, carry):
        q_u, log_dyn_channels, log_obs_channels = carry

        # Inline kernels (factor / channel in log-space)
        log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
        log_obs_kernels = log_B_flat[None] + log_obs_channels

        # Reduced tensors: factor_reduced / channel (avoids per-iteration logsumexp over θ)
        log_reduced_per_t = safe_log_div(
            log_factor_reduced[None],
            log_dyn_channels.transpose(0, 2, 1, 3)  # (T, x_new, x_old, u)
        )

        # obs->x messages
        log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs)

        # Combine with preference if 2D goal
        log_local_to_x = log_obs_to_x + log_pref_to_x if has_pref else log_obs_to_x

        # Forward pass
        log_fwd_msgs = forward_pass(
            log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon
        )

        # Backward pass + action marginals
        log_bwd_msgs, q_u = backward_pass(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
            log_local_to_x, horizon
        )

        # Region beliefs (using FIXED cavities, combined local_to_x)
        log_dyn_regions = compute_dyn_region_beliefs(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
            log_cavity_dyn, log_action_prior
        )
        log_obs_regions = compute_obs_region_beliefs(
            log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
            log_cavity_obs
        )

        # Channels from region beliefs (with damping)
        raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
        raw_log_obs_channels = compute_obs_channels(log_obs_regions)

        new_log_dyn_channels = damp_log_channel(
            log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)
        new_log_obs_channels = damp_log_channel(
            log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)

        return q_u, new_log_dyn_channels, new_log_obs_channels

    q_u, log_dyn_channels, log_obs_channels = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, log_dyn_channels_init, log_obs_channels_init)
    )

    return q_u[0], log_dyn_channels, log_obs_channels
