"""Precise info-seeking planning: VBP action channels + obs channel reparameterization.

Combines two reparameterizations:
- Obs channel from region_extended: obs kernels use r(y|x,θ) and r(y|x)
- VBP action channel from vbp_channel: r(u|x) enters dynamics kernel in NUMERATOR

Modified dynamics kernel: κ_t(x_old, x_new, θ, u) = T(x_old, x_new, θ, u) · r_t(u|x_old)
Obs kernel: B(y|x,θ) · r(y|x,θ) · r(y|x,θ)/r(y|x) (same as region-extended)

All internal computation is in log-space. Accepts probability-space tensors.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .messages import LOG_ZERO, safe_log
from .messages import safe_log_div
from .region_extended_loopy_bp import (
    compute_log_reduced,
    forward_pass,
    backward_pass,
    compute_obs_to_x_msgs,
    compute_obs_to_theta_msgs,
    compute_pref_to_x_msgs,
    compute_pref_to_theta_msgs,
    compute_theta_cavities_extended,
    compute_dyn_to_theta_msgs,
    compute_dyn_region_beliefs,
    compute_obs_region_beliefs,
    compute_obs_channels,
    compute_marginal_obs_channels,
    damp_log_channel,
)
from .vbp_channel import (
    compute_dyn_kernels_vbp,
    compute_pair_marginal,
    compute_action_channel,
)


@partial(jax.jit, static_argnums=(5, 6))
def precise_info_seeking_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta
    transition_tensor,    # (n_states, n_states, n_static, n_actions)
    observation_tensor,   # (n_channels, n_obs_types, n_states, n_static)
    goal,                 # (n_states,) or (n_states, n_static) preference
    horizon,              # int (static)
    n_iterations,         # int (static)
    damping=1.0,          # float - channel update damping (1.0 = no damping)
    action_prior=None,    # (n_actions,) prior over actions. If None, uniform.
) -> tuple:
    """
    Plan actions via precise info-seeking: VBP action channels + obs channels.

    Returns:
        action_dist: (n_actions,)
        log_action_channels: (T, n_states, n_actions)
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
    log_T = safe_log(transition_tensor)                   # (x_new, x_old, θ, u)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)           # (x_old, x_new, θ, u)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)

    if has_pref:
        log_C = safe_log(goal)                            # (n_states, n_static)
        log_goal = jnp.zeros(n_states)                    # uniform terminal
    else:
        log_C = None
        log_goal = safe_log(goal)

    # Initialize messages
    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    log_obs_to_theta = jnp.zeros((horizon + 1, n_static))
    log_pref_to_theta_init = jnp.zeros((horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))

    # Initial action channels: uniform r(u|x) = 1/n_actions (from vbp_channel)
    log_r_ux_init = jnp.full((horizon, n_states, n_actions), -jnp.log(n_actions))

    # Initial obs channels: r(y | x, θ) = B(y | x, θ) (from region_extended)
    log_obs_ch0 = log_B_flat - logsumexp(log_B_flat, axis=1, keepdims=True)
    log_obs_channels_init = jnp.broadcast_to(log_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static))

    # Initial marginal obs channels: r(y | x) by marginalizing θ with prior
    log_marginal_obs_ch0 = logsumexp(
        log_B_flat + log_prior_theta[None, None, None, :], axis=3)
    log_marginal_obs_ch0 = log_marginal_obs_ch0 - logsumexp(log_marginal_obs_ch0, axis=1, keepdims=True)
    log_marginal_obs_channels_init = jnp.broadcast_to(
        log_marginal_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states))

    if has_pref:
        def body_fn(i, carry):
            (log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta, _,
             log_r_ux, log_obs_channels, log_marginal_obs_channels) = carry

            # Step 1: 3-way theta cavities
            log_cavity_dyn, log_cavity_obs, log_cavity_pref = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta
            )

            # Step 2: Dyn kernels via VBP (channel in NUMERATOR)
            log_dyn_kernels = compute_dyn_kernels_vbp(log_T_kernel, log_r_ux)

            # Step 3: Obs kernels via region-extended
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            # Step 4: Reduced tensors from dyn kernels
            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

            # Step 5: obs->x and pref->x messages
            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs)
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_cavity_pref)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            # Step 6: Forward pass
            log_fwd_msgs = forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon
            )

            # Step 7: Backward pass
            log_bwd_msgs, q_u = backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_local_to_x, horizon
            )

            # Step 8: dyn->theta messages
            new_log_dyn_to_theta = compute_dyn_to_theta_msgs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_action_prior, horizon
            )

            # Step 9: obs->theta messages (include pref_to_x in x belief)
            new_log_obs_to_theta = compute_obs_to_theta_msgs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_extra_to_x=log_pref_to_x
            )

            # Step 10: pref->theta messages (include obs_to_x in x belief)
            new_log_pref_to_theta = compute_pref_to_theta_msgs(
                log_C, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            # Step 11: Region beliefs
            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_cavity_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_cavity_obs
            )

            # Step 12: Action channels from dyn region beliefs (VBP style)
            log_pair = compute_pair_marginal(log_dyn_regions)
            raw_log_r_ux = compute_action_channel(log_pair)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            # Step 13: Obs channels from obs region beliefs (region-extended style)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)

            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            return (new_log_dyn_to_theta, new_log_obs_to_theta, new_log_pref_to_theta,
                    q_u, new_log_r_ux, new_log_obs_channels,
                    new_log_marginal_obs_channels)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta_init,
             q_u_init, log_r_ux_init, log_obs_channels_init,
             log_marginal_obs_channels_init)
        )
        _, _, _, q_u, log_r_ux, log_obs_channels, _ = result
    else:
        def body_fn(i, carry):
            (log_dyn_to_theta, log_obs_to_theta, _,
             log_r_ux, log_obs_channels, log_marginal_obs_channels) = carry

            # Step 1: theta cavities
            log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta
            )

            # Step 2: Dyn kernels via VBP (channel in NUMERATOR)
            log_dyn_kernels = compute_dyn_kernels_vbp(log_T_kernel, log_r_ux)

            # Step 3: Obs kernels via region-extended
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            # Step 4: Reduced tensors from dyn kernels
            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

            # Step 5: obs->x messages
            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs)

            # Step 6: Forward pass
            log_fwd_msgs = forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon
            )

            # Step 7: Backward pass + action marginals
            log_bwd_msgs, q_u = backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_obs_to_x, horizon
            )

            # Step 8: dyn->theta messages
            new_log_dyn_to_theta = compute_dyn_to_theta_msgs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_action_prior, horizon
            )

            # Step 9: obs->theta messages
            new_log_obs_to_theta = compute_obs_to_theta_msgs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            # Step 10: Region beliefs
            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_cavity_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_cavity_obs
            )

            # Step 11: Action channels from dyn region beliefs (VBP style)
            log_pair = compute_pair_marginal(log_dyn_regions)
            raw_log_r_ux = compute_action_channel(log_pair)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            # Step 12: Obs channels from obs region beliefs (region-extended style)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)

            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            return (new_log_dyn_to_theta, new_log_obs_to_theta, q_u,
                    new_log_r_ux, new_log_obs_channels,
                    new_log_marginal_obs_channels)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, log_obs_to_theta, q_u_init,
             log_r_ux_init, log_obs_channels_init,
             log_marginal_obs_channels_init)
        )
        _, _, q_u, log_r_ux, log_obs_channels, _ = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_r_ux, log_obs_channels
