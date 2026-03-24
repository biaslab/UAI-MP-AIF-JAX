"""Active Inference planning: VBP action channels + dyn channels + obs channels.

Combines three channel reparameterizations:
- VBP action channel from vbp_channel: r(u|x) enters dynamics kernel in NUMERATOR
- Dyn channel from region_extended: r(x'|x,u) enters dynamics kernel in DENOMINATOR
- Obs channel from region_extended: obs kernels use r(y|x,θ) and r(y|x,θ)/r(y|x)

Optimized: precomputes θ-marginalized dynamics base before the iteration loop,
replacing per-iteration 5D tensor construction with 4D element-wise operations.

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
    forward_pass,
    backward_pass,
    compute_obs_to_x_msgs,
    compute_pref_to_x_msgs,
    damp_log_channel,
)


def compute_dyn_kernels_aif(log_T_kernel, log_r_ux, log_dyn_channels):
    """Compute AIF dynamics kernels: T * r(u|x) / r(x'|x,u).

    Combines VBP action channel (numerator) with region-extended
    dynamics channel (denominator).

    Note: not used by active_inference_planning (which precomputes the
    θ-marginalized base), but still needed by convergence.py.

    Args:
        log_T_kernel: (x_old, x_new, theta, u) log transition kernel
        log_r_ux: (T, n_states, n_actions) log action channels
        log_dyn_channels: (T, x_old, x_new, u) log dynamics channels

    Returns:
        (T, x_old, x_new, theta, u) log dynamics kernels
    """
    # T / r(x'|x,u) — denominator channel
    kernel = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
    # * r(u|x) — numerator channel
    return kernel + log_r_ux[:, :, None, None, :]


@partial(jax.jit, static_argnums=(5, 6))
def active_inference_planning(
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
    Plan actions via Active Inference: VBP action channels + dyn channels + obs channels.

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

    # Precompute prior broadcast for obs messages
    log_prior_obs = jnp.broadcast_to(log_prior_theta[None, :], (horizon + 1, n_static))

    # === Precompute θ-marginalized dynamics base (major optimization) ===
    # log_base[xo, xn, u] = logsumexp_θ(log_T_kernel[xo,xn,θ,u] + log_prior_theta[θ])
    # This is constant across iterations — θ prior is fixed, not a cavity.
    log_base = logsumexp(
        log_T_kernel + log_prior_theta[None, None, :, None], axis=2
    )  # (x_old, x_new, u)

    q_u_init = jnp.zeros((horizon, n_actions))
    log_fwd_prev_init = jnp.zeros((horizon + 1, n_states))
    log_bwd_prev_init = jnp.zeros((horizon + 1, n_states))

    # Initial action channels: uniform r(u|x) = 1/n_actions (from vbp_channel)
    log_r_ux_init = jnp.full((horizon, n_states, n_actions), -jnp.log(n_actions))

    # Initial channels: uniform
    log_dyn_channels_init = jnp.full((horizon, n_states, n_states, n_actions), -jnp.log(n_states))
    log_obs_channels_init = jnp.full((horizon + 1, n_fov, n_obs_types, n_states, n_static), -jnp.log(n_obs_types))
    log_marginal_obs_channels_init = jnp.full((horizon + 1, n_fov, n_obs_types, n_states), -jnp.log(n_obs_types))

    if has_pref:
        def body_fn(i, carry):
            (_, log_r_ux, log_dyn_channels, log_obs_channels, log_marginal_obs_channels,
             log_fwd_prev, log_bwd_prev) = carry

            # === 4D base with channel adjustment (replaces 5D kernel + logsumexp over θ) ===
            dyn_valid = log_dyn_channels > LOG_ZERO / 2
            neg_dyn_ch = jnp.where(dyn_valid, -log_dyn_channels, 0.0)
            base_4d = log_base[None] + neg_dyn_ch + log_r_ux[:, :, None, :]
            base_4d = jnp.where(dyn_valid, base_4d, LOG_ZERO)

            # Step 1: Reduced tensors from precomputed base
            log_reduced_per_t = base_4d.transpose(0, 2, 1, 3)  # (T, xn, xo, u)

            # Step 2: Obs kernels (unchanged)
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            # Step 3: obs->x and pref->x messages (use prior instead of cavity)
            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_prior_obs)
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_prior_obs)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            # Step 4: Forward pass (with inertial damping)
            log_fwd_msgs = forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping
            )

            # Step 5: Backward pass (with inertial damping)
            log_bwd_msgs, q_u = backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_local_to_x, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping
            )

            # Step 6: Dyn channels + action channels from 4D theta_marg
            fwd_local = log_fwd_msgs[:-1] + log_local_to_x[:-1]
            bwd_local = log_bwd_msgs[1:] + log_local_to_x[1:]
            log_theta_marg = (base_4d
                              + fwd_local[:, :, None, None]
                              + bwd_local[:, None, :, None]
                              + log_action_prior[None, None, None, :])
            raw_log_dyn_channels = log_theta_marg - logsumexp(log_theta_marg, axis=2, keepdims=True)
            log_pair = logsumexp(log_theta_marg, axis=2)
            raw_log_r_ux = log_pair - logsumexp(log_pair, axis=2, keepdims=True)

            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)
            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)

            # Step 7: Obs channels directly from kernels (skips 5D obs region beliefs)
            raw_log_obs_channels = log_obs_kernels - logsumexp(
                log_obs_kernels, axis=2, keepdims=True)
            kernel_with_prior = log_obs_kernels + log_prior_obs[:, None, None, None, :]
            theta_marg_obs = logsumexp(kernel_with_prior, axis=4)
            raw_log_marginal_obs_channels = theta_marg_obs - logsumexp(
                theta_marg_obs, axis=2, keepdims=True)

            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            return (q_u, new_log_r_ux, new_log_dyn_channels, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_r_ux_init, log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init)
        )
        q_u, _, log_dyn_channels, log_obs_channels, _, _, _ = result
    else:
        def body_fn(i, carry):
            (_, log_r_ux, log_dyn_channels, log_obs_channels, log_marginal_obs_channels,
             log_fwd_prev, log_bwd_prev) = carry

            # === 4D base with channel adjustment (replaces 5D kernel + logsumexp over θ) ===
            dyn_valid = log_dyn_channels > LOG_ZERO / 2
            neg_dyn_ch = jnp.where(dyn_valid, -log_dyn_channels, 0.0)
            base_4d = log_base[None] + neg_dyn_ch + log_r_ux[:, :, None, :]
            base_4d = jnp.where(dyn_valid, base_4d, LOG_ZERO)

            # Step 1: Reduced tensors from precomputed base
            log_reduced_per_t = base_4d.transpose(0, 2, 1, 3)  # (T, xn, xo, u)

            # Step 2: Obs kernels (unchanged)
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            # Step 3: obs->x messages (use prior instead of cavity)
            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_prior_obs)

            # Step 4: Forward pass (with inertial damping)
            log_fwd_msgs = forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping
            )

            # Step 5: Backward pass + action marginals (with inertial damping)
            log_bwd_msgs, q_u = backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_obs_to_x, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping
            )

            # Step 6: Dyn channels + action channels from 4D theta_marg
            fwd_local = log_fwd_msgs[:-1] + log_obs_to_x[:-1]
            bwd_local = log_bwd_msgs[1:] + log_obs_to_x[1:]
            log_theta_marg = (base_4d
                              + fwd_local[:, :, None, None]
                              + bwd_local[:, None, :, None]
                              + log_action_prior[None, None, None, :])
            raw_log_dyn_channels = log_theta_marg - logsumexp(log_theta_marg, axis=2, keepdims=True)
            log_pair = logsumexp(log_theta_marg, axis=2)
            raw_log_r_ux = log_pair - logsumexp(log_pair, axis=2, keepdims=True)

            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)
            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)

            # Step 7: Obs channels directly from kernels (skips 5D obs region beliefs)
            raw_log_obs_channels = log_obs_kernels - logsumexp(
                log_obs_kernels, axis=2, keepdims=True)
            kernel_with_prior = log_obs_kernels + log_prior_obs[:, None, None, None, :]
            theta_marg_obs = logsumexp(kernel_with_prior, axis=4)
            raw_log_marginal_obs_channels = theta_marg_obs - logsumexp(
                theta_marg_obs, axis=2, keepdims=True)

            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            return (q_u, new_log_r_ux, new_log_dyn_channels, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_r_ux_init, log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init)
        )
        q_u, _, log_dyn_channels, log_obs_channels, _, _, _ = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_dyn_channels, log_obs_channels
