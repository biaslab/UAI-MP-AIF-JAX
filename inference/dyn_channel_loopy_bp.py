"""Dyn+action channel loopy BP planning with observation factors (theta inferred).

Like region_extended_loopy_bp.py but dynamics factors get both:
- Dynamics channel reparameterization: r(x'|x,u) in the denominator
- VBP action channel reparameterization: r(u|x) in the numerator

Kernel: kappa_t = T * r(u|x) / r(x'|x,u)

Observation factors contribute obs->x and obs->theta messages via the raw
observation tensor (no obs kernels, no obs channels, no obs region beliefs).

All internal computation is in log-space. Accepts probability-space tensors.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .messages import (
    LOG_ZERO, safe_log, safe_log_div,
    sparse_reduced_dyn_channel, sparse_dyn_to_theta_dyn_channel,
    sparse_dyn_channels_and_pair_dyn_channel,
)
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
    compute_dyn_channels_and_pair_marginal,
    damp_log_channel,
)


@partial(jax.jit, static_argnums=(5, 6))
def dyn_channel_loopy_bp_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta
    transition_tensor,    # (n_states, n_states, n_static, n_actions)
    observation_tensor,   # (n_channels, n_obs_types, n_states, n_static)
    goal,                 # (n_states,) or (n_states, n_static) preference
    horizon,              # int (static)
    n_iterations,         # int (static)
    damping=1.0,          # float - channel update damping (1.0 = no damping)
    action_prior=None,    # (n_actions,) prior over actions. If None, uniform.
    T_idx=None,           # (S, A, θ) int32 sparse transition index
) -> tuple:
    """
    Plan actions via dyn+action channel loopy BP with observation factors.

    Observation factors use the raw observation tensor (no obs channels).
    Dynamics factors get kernel reparameterization via both:
    - Dynamics channels r(x'|x,u) in the denominator
    - VBP action channels r(u|x) in the numerator

    Returns:
        action_dist: (n_actions,)
        log_dyn_channels: (T, n_states, n_states, n_actions)
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

    use_sparse = T_idx is not None

    # Log once at top
    if not use_sparse:
        log_T = safe_log(transition_tensor)               # (x_new, x_old, theta, u)
        log_T_kernel = log_T.transpose(1, 0, 2, 3)       # (x_old, x_new, theta, u)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)

    if has_pref:
        log_C = safe_log(goal)                            # (n_states, n_static)
        log_goal = jnp.zeros(n_states)                    # uniform terminal
    else:
        log_C = None
        log_goal = safe_log(goal)

    # Tile obs tensor over time: kernel = raw B (no obs channels)
    log_B_tiled = jnp.broadcast_to(
        log_B_flat[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static)
    )

    # Initialize messages
    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    log_obs_to_theta = jnp.zeros((horizon + 1, n_static))
    log_pref_to_theta_init = jnp.zeros((horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))

    # Initial dyn channels: uniform r(x_new | x_old, u) = 1/n_states
    log_dyn_channels_init = jnp.full((horizon, n_states, n_states, n_actions), -jnp.log(n_states))

    # Initial action channels: uniform r(u|x) = 1/n_actions
    log_r_ux_init = jnp.full((horizon, n_states, n_actions), -jnp.log(n_actions))

    if has_pref:
        def body_fn(i, carry):
            (log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta, _,
             log_dyn_channels, log_r_ux) = carry

            # Step 1: 3-way theta cavities
            log_cavity_dyn, log_cavity_obs, log_cavity_pref = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta
            )

            # Steps 2-3: Reduced tensors (kernel = T * r(u|x) / r(x'|x,u))
            if use_sparse:
                log_reduced_per_t = sparse_reduced_dyn_channel(
                    T_idx, log_cavity_dyn, log_dyn_channels, log_r_ux, n_states)
            else:
                log_dyn_kernels = (
                    safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
                    + log_r_ux[:, :, None, None, :])
                log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

            # Step 4: obs->x and pref->x messages
            log_obs_to_x = compute_obs_to_x_msgs(log_B_tiled, log_cavity_obs)
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_cavity_pref)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            # Step 5: Forward pass (with combined local messages)
            log_fwd_msgs = forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon
            )

            # Step 6: Backward pass (uniform terminal, pref enters via local_to_x)
            log_bwd_msgs, q_u = backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_local_to_x, horizon
            )

            # Step 7: dyn->theta messages
            if use_sparse:
                new_log_dyn_to_theta = sparse_dyn_to_theta_dyn_channel(
                    T_idx, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                    log_action_prior, log_dyn_channels, log_r_ux, n_states)
            else:
                new_log_dyn_to_theta = compute_dyn_to_theta_msgs(
                    log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                    log_action_prior, horizon)

            # Step 8: obs->theta messages (include pref_to_x in x belief)
            new_log_obs_to_theta = compute_obs_to_theta_msgs(
                log_B_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_extra_to_x=log_pref_to_x
            )

            # Step 9: pref->theta messages (include obs_to_x in x belief)
            new_log_pref_to_theta = compute_pref_to_theta_msgs(
                log_C, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            # Step 10: Dyn region beliefs -> dyn channels + action channels -> damped update
            if use_sparse:
                raw_log_dyn_channels, log_pair = sparse_dyn_channels_and_pair_dyn_channel(
                    T_idx, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                    log_cavity_dyn, log_action_prior, log_dyn_channels, log_r_ux,
                    n_states)
            else:
                log_dyn_regions = compute_dyn_region_beliefs(
                    log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                    log_cavity_dyn, log_action_prior)
                raw_log_dyn_channels, log_pair = compute_dyn_channels_and_pair_marginal(log_dyn_regions)
            raw_log_r_ux = log_pair - logsumexp(log_pair, axis=2, keepdims=True)
            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            return (new_log_dyn_to_theta, new_log_obs_to_theta, new_log_pref_to_theta,
                    q_u, new_log_dyn_channels, new_log_r_ux)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta_init,
             q_u_init, log_dyn_channels_init, log_r_ux_init)
        )
        _, _, _, q_u, log_dyn_channels, _ = result
    else:
        def body_fn(i, carry):
            log_dyn_to_theta, log_obs_to_theta, _, log_dyn_channels, log_r_ux = carry

            # Step 1: theta cavities
            log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta
            )

            # Steps 2-3: Reduced tensors
            if use_sparse:
                log_reduced_per_t = sparse_reduced_dyn_channel(
                    T_idx, log_cavity_dyn, log_dyn_channels, log_r_ux, n_states)
            else:
                log_dyn_kernels = (
                    safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
                    + log_r_ux[:, :, None, None, :])
                log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

            # Step 4: obs->x messages (raw B, no kernels)
            log_obs_to_x = compute_obs_to_x_msgs(log_B_tiled, log_cavity_obs)

            # Step 5: Forward pass
            log_fwd_msgs = forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon
            )

            # Step 6: Backward pass + action marginals
            log_bwd_msgs, q_u = backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_obs_to_x, horizon
            )

            # Step 7: dyn->theta messages
            if use_sparse:
                new_log_dyn_to_theta = sparse_dyn_to_theta_dyn_channel(
                    T_idx, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                    log_action_prior, log_dyn_channels, log_r_ux, n_states)
            else:
                new_log_dyn_to_theta = compute_dyn_to_theta_msgs(
                    log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                    log_action_prior, horizon)

            # Step 8: obs->theta messages (raw B)
            new_log_obs_to_theta = compute_obs_to_theta_msgs(
                log_B_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            # Step 9: Dyn region beliefs -> dyn channels + action channels -> damped update
            if use_sparse:
                raw_log_dyn_channels, log_pair = sparse_dyn_channels_and_pair_dyn_channel(
                    T_idx, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                    log_cavity_dyn, log_action_prior, log_dyn_channels, log_r_ux,
                    n_states)
            else:
                log_dyn_regions = compute_dyn_region_beliefs(
                    log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                    log_cavity_dyn, log_action_prior)
                raw_log_dyn_channels, log_pair = compute_dyn_channels_and_pair_marginal(log_dyn_regions)
            raw_log_r_ux = log_pair - logsumexp(log_pair, axis=2, keepdims=True)
            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            return (new_log_dyn_to_theta, new_log_obs_to_theta, q_u,
                    new_log_dyn_channels, new_log_r_ux)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, log_obs_to_theta, q_u_init,
             log_dyn_channels_init, log_r_ux_init)
        )
        _, _, q_u, log_dyn_channels, _ = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_dyn_channels
