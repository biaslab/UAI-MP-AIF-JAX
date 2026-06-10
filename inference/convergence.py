"""Convergence variants of planning functions that return per-iteration VFE traces.

Each convergence function mirrors its original planning function but additionally
accumulates a VFE value at each iteration. The VFE formulas differ by method:

- Loopy BP: Bethe free energy
- Region-extended: Region-based free energy with observation factors
- Nuijten: Same as region-based but with EFE action priors
- Dyn-channel: Region-based free energy with dynamic channels

All internal computation is in log-space.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .messages import LOG_ZERO, safe_log, safe_log_div
from .vbp_channel import compute_dyn_kernels_vbp, compute_pair_marginal, compute_action_channel
from .active_inference import compute_dyn_kernels_aif
from .loopy_bp import (
    compute_reduced_per_t,
    forward_pass as loopy_forward_pass,
    backward_pass as loopy_backward_pass,
    compute_dyn_to_theta_msgs as loopy_dyn_to_theta,
    compute_theta_cavities,
)
from .region_extended_loopy_bp import (
    compute_log_reduced,
    forward_pass as re_forward_pass,
    backward_pass as re_backward_pass,
    compute_dyn_to_theta_msgs as re_dyn_to_theta,
    compute_obs_to_x_msgs,
    compute_obs_to_theta_msgs,
    compute_pref_to_x_msgs,
    compute_pref_to_theta_msgs,
    compute_theta_cavities_extended,
    compute_dyn_region_beliefs,
    compute_obs_region_beliefs,
    compute_dyn_channels,
    compute_obs_channels,
    compute_marginal_obs_channels,
    damp_log_channel,
)
from .nuijten_mp import (
    compute_obs_region_beliefs_original,
    compute_efe_action_prior,
    compute_obs_efe_to_x,
    compute_obs_efe_to_theta,
    forward_pass_nuijten,
    backward_pass_nuijten,
    compute_dyn_to_theta_msgs_nuijten,
    compute_dyn_region_beliefs_nuijten,
)


# =============================================================================
# VFE helpers
# =============================================================================


def _entropy(log_q):
    """Shannon entropy H[q] = -sum q log q from normalized log-probs.

    Args:
        log_q: (...,) log-probabilities (normalized over last axis or flat).

    Returns:
        scalar entropy value
    """
    q = jnp.exp(log_q)
    return -jnp.sum(jnp.where(q > 1e-30, q * log_q, 0.0))


def _entropy_unnorm(log_b):
    """Shannon entropy from unnormalized log beliefs (normalizes first).

    Normalizes over ALL dimensions (treats as flat distribution).

    Args:
        log_b: arbitrary shape, unnormalized log-beliefs

    Returns:
        scalar entropy of the normalized distribution
    """
    flat = log_b.reshape(-1)
    log_q = flat - logsumexp(flat)
    q = jnp.exp(log_q)
    return -jnp.sum(jnp.where(q > 1e-30, q * log_q, 0.0))


def _energy(log_region, log_factor):
    """Average energy U = -sum b(x) log f(x) from unnormalized log region beliefs.

    Normalizes the region belief, then computes -sum q * log_factor.
    Entries where log_factor == LOG_ZERO are skipped (structural zeros).

    Args:
        log_region: unnormalized log region beliefs
        log_factor: log factor values (same shape)

    Returns:
        scalar energy value
    """
    flat_region = log_region.reshape(-1)
    log_q = flat_region - logsumexp(flat_region)
    q = jnp.exp(log_q)
    flat_factor = log_factor.reshape(-1)
    valid = flat_factor > LOG_ZERO / 2
    return -jnp.sum(jnp.where(valid, q * flat_factor, 0.0))


# =============================================================================
# Bethe VFE for loopy BP
# =============================================================================


def compute_bethe_vfe_loopy(log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs,
                             q_u, log_cavity_theta, log_prior_theta,
                             log_action_prior):
    """Bethe VFE for loopy BP with theta as variable.

    Like standard Bethe but:
    - Factor beliefs include theta (via cavity)
    - Energy uses full log_T (not reduced)
    - H[q(theta)] subtracted with counting number (1 - T)
    """
    horizon = q_u.shape[0]

    def per_t_vfe(t):
        # Factor belief with theta
        log_b = (log_T.transpose(1, 0, 2, 3)  # (x_old, x_new, theta, u)
                 + log_fwd_msgs[t][:, None, None, None]
                 + log_bwd_msgs[t + 1][None, :, None, None]
                 + log_cavity_theta[t][None, None, :, None]
                 + safe_log(q_u[t])[None, None, None, :])

        # Energy uses original log_T (x_new, x_old, theta, u) → transpose
        log_factor = log_T.transpose(1, 0, 2, 3)
        U_t = _energy(log_b, log_factor)
        H_t = _entropy_unnorm(log_b)
        return U_t - H_t

    factor_vfe = jax.vmap(per_t_vfe)(jnp.arange(horizon)).sum()

    # Singleton entropies: intermediate x_t
    def x_entropy(t):
        log_q_x = log_fwd_msgs[t] + log_bwd_msgs[t]
        log_q_x = log_q_x - logsumexp(log_q_x)
        return _entropy(log_q_x)

    singleton_H = jnp.where(
        horizon > 1,
        jax.vmap(x_entropy)(jnp.arange(1, horizon)).sum(),
        0.0,
    )

    # Theta entropy with counting number (1 - T)
    log_q_theta = log_cavity_theta.mean(axis=0)
    log_q_theta = log_q_theta - logsumexp(log_q_theta)
    H_theta = _entropy(log_q_theta)
    theta_counting = 1.0 - horizon
    theta_term = theta_counting * H_theta

    return factor_vfe - singleton_H - theta_term


# =============================================================================
# Channel correction helpers (conditional entropies from region beliefs)
# =============================================================================


def _action_cond_entropy(log_dyn_regions, horizon):
    """Σ_t H[q(u|x)] from dyn region beliefs.

    H[q(u|x)] = H[q(x,u)] - H[q(x)] where q(x,u) is the pair marginal.
    """
    def per_t(t):
        log_pair = logsumexp(log_dyn_regions[t], axis=(1, 2))  # (x_old, u)
        return _entropy_unnorm(log_pair) - _entropy_unnorm(logsumexp(log_pair, axis=1))

    return jax.vmap(per_t)(jnp.arange(horizon)).sum()


def _dyn_cond_entropy(log_dyn_regions, horizon):
    """Σ_t H[q(x'|x,u)] from dyn region beliefs.

    H[q(x'|x,u)] = H[q(x,x',u)] - H[q(x,u)].
    """
    def per_t(t):
        log_xxu = logsumexp(log_dyn_regions[t], axis=2)  # (x_old, x_new, u) marginalize θ
        log_xu = logsumexp(log_xxu, axis=1)               # (x_old, u)
        return _entropy_unnorm(log_xxu) - _entropy_unnorm(log_xu)

    return jax.vmap(per_t)(jnp.arange(horizon)).sum()


def _obs_cond_entropy(log_obs_regions, horizon):
    """Σ_{t,k} (2·H[q(y|x,θ)] − H[q(y|x)]) from obs region beliefs.

    log_obs_regions: (T+1, n_fov, n_obs_types, n_states, n_static)
    """
    n_fov = log_obs_regions.shape[1]

    def per_t(t):
        def per_k(k):
            regions = log_obs_regions[t, k]  # (n_obs_types, n_states, n_static)
            # H[q(y|x,θ)] = H[q(y,x,θ)] - H[q(x,θ)]
            H_y_given_xtheta = (_entropy_unnorm(regions)
                                - _entropy_unnorm(logsumexp(regions, axis=0)))
            # H[q(y|x)] from marginal q(y,x) = Σ_θ q(y,x,θ)
            log_yx = logsumexp(regions, axis=2)  # (n_obs_types, n_states)
            H_y_given_x = (_entropy_unnorm(log_yx)
                           - _entropy_unnorm(logsumexp(log_yx, axis=0)))
            return 2.0 * H_y_given_xtheta - H_y_given_x

        return jax.vmap(per_k)(jnp.arange(n_fov)).sum()

    return jax.vmap(per_t)(jnp.arange(horizon + 1)).sum()


# =============================================================================
# VFE functions for each method (all build on compute_bethe_vfe_loopy)
# =============================================================================


def compute_region_extended_vfe(log_T, log_reduced_per_t, log_fwd_msgs,
                                 log_bwd_msgs, q_u, log_cavity_dyn,
                                 log_prior_theta, log_action_prior,
                                 log_dyn_regions, log_obs_regions):
    """VFE for region-extended: F_p + obs_correction − dyn_cond_entropy."""
    horizon = q_u.shape[0]
    F_p = compute_bethe_vfe_loopy(log_T, log_reduced_per_t, log_fwd_msgs,
                                   log_bwd_msgs, q_u, log_cavity_dyn,
                                   log_prior_theta, log_action_prior)
    return (F_p
            + _obs_cond_entropy(log_obs_regions, horizon)
            - _dyn_cond_entropy(log_dyn_regions, horizon))


def compute_dyn_channel_vfe(log_T, log_reduced_per_t, log_fwd_msgs,
                             log_bwd_msgs, q_u, log_cavity_dyn,
                             log_prior_theta, log_action_prior,
                             log_dyn_regions):
    """VFE for dyn-channel: F_p − dyn_cond_entropy."""
    horizon = q_u.shape[0]
    F_p = compute_bethe_vfe_loopy(log_T, log_reduced_per_t, log_fwd_msgs,
                                   log_bwd_msgs, q_u, log_cavity_dyn,
                                   log_prior_theta, log_action_prior)
    return F_p - _dyn_cond_entropy(log_dyn_regions, horizon)


def compute_vbp_channel_vfe(log_T, log_reduced_per_t, log_fwd_msgs,
                             log_bwd_msgs, q_u, log_cavity_dyn,
                             log_prior_theta, log_action_prior,
                             log_dyn_regions):
    """VFE for vbp-channel: F_p + action_cond_entropy."""
    horizon = q_u.shape[0]
    F_p = compute_bethe_vfe_loopy(log_T, log_reduced_per_t, log_fwd_msgs,
                                   log_bwd_msgs, q_u, log_cavity_dyn,
                                   log_prior_theta, log_action_prior)
    return F_p + _action_cond_entropy(log_dyn_regions, horizon)


def compute_precise_info_seeking_vfe(log_T, log_reduced_per_t, log_fwd_msgs,
                                      log_bwd_msgs, q_u, log_cavity_dyn,
                                      log_prior_theta, log_action_prior,
                                      log_dyn_regions, log_obs_regions):
    """VFE for precise-info-seeking: F_p + action_cond_entropy + obs_correction."""
    horizon = q_u.shape[0]
    F_p = compute_bethe_vfe_loopy(log_T, log_reduced_per_t, log_fwd_msgs,
                                   log_bwd_msgs, q_u, log_cavity_dyn,
                                   log_prior_theta, log_action_prior)
    return (F_p
            + _action_cond_entropy(log_dyn_regions, horizon)
            + _obs_cond_entropy(log_obs_regions, horizon))


# =============================================================================
# Nuijten VFE (same as region-extended but with EFE action prior factors)
# =============================================================================


def compute_nuijten_vfe(log_dyn_regions, obs_regions,
                         log_fwd_msgs, log_bwd_msgs, q_u,
                         log_T_kernel_tiled, log_B_flat,
                         log_cavity_dyn, log_cavity_obs,
                         log_prior_theta, action_prior_per_t):
    """VFE for Nuijten methods — region-extended with additional EFE prior factors.

    Same structure as region_extended_vfe but adds:
    - Energy from EFE action prior factors: U_prior(t) = -sum q(u_t) * log pi_t(u_t)
    """
    horizon = q_u.shape[0]
    n_fov = obs_regions.shape[1]

    # Use the region-extended VFE as base
    # But obs_regions here is in probability space (from compute_obs_region_beliefs_original)
    # Convert to log-space for the VFE computation
    log_obs_regions = safe_log(obs_regions)

    # --- Energy terms ---
    def dyn_energy_t(t):
        return _energy(log_dyn_regions[t], log_T_kernel_tiled[t])

    U_dyn = jax.vmap(dyn_energy_t)(jnp.arange(horizon)).sum()

    def obs_energy_t(t):
        def obs_energy_k(k):
            return _energy(log_obs_regions[t, k], log_B_flat[k])
        return jax.vmap(obs_energy_k)(jnp.arange(n_fov)).sum()

    U_obs = jax.vmap(obs_energy_t)(jnp.arange(horizon + 1)).sum()

    # EFE action prior energy
    def prior_u_energy(t):
        log_pi = safe_log(action_prior_per_t[t])
        log_q_u = safe_log(q_u[t])
        q_ut = jnp.exp(log_q_u)
        return -jnp.sum(q_ut * log_pi)

    U_prior = jax.vmap(prior_u_energy)(jnp.arange(horizon)).sum()

    # --- Singleton entropies ---
    def x_entropy(t):
        log_q_x = log_fwd_msgs[t] + log_bwd_msgs[t]
        log_q_x = log_q_x - logsumexp(log_q_x)
        return _entropy(log_q_x)

    H_x = jax.vmap(x_entropy)(jnp.arange(horizon + 1)).sum()

    def u_entropy(t):
        return _entropy(safe_log(q_u[t]))

    H_u = jax.vmap(u_entropy)(jnp.arange(horizon)).sum()

    all_cavities = jnp.concatenate([log_cavity_dyn, log_cavity_obs], axis=0)
    log_q_theta = all_cavities.mean(axis=0)
    log_q_theta = log_q_theta - logsumexp(log_q_theta)
    H_theta = _entropy(log_q_theta)

    # --- Intersection terms ---
    def dyn_intersection_t(t):
        log_joint_xu = logsumexp(log_dyn_regions[t], axis=(1, 2))
        H_xu = _entropy_unnorm(log_joint_xu)
        log_joint_xxu = logsumexp(log_dyn_regions[t], axis=2)
        H_xxu = _entropy_unnorm(log_joint_xxu)
        return H_xu - H_xxu

    dyn_intersection = jax.vmap(dyn_intersection_t)(jnp.arange(horizon)).sum()

    def obs_intersection_t(t):
        def obs_intersection_k(k):
            H_yxtheta = _entropy_unnorm(log_obs_regions[t, k])
            log_xtheta = logsumexp(log_obs_regions[t, k], axis=0)
            H_xtheta = _entropy_unnorm(log_xtheta)
            return H_yxtheta - H_xtheta
        return jax.vmap(obs_intersection_k)(jnp.arange(n_fov)).sum()

    obs_intersection = jax.vmap(obs_intersection_t)(jnp.arange(horizon + 1)).sum()

    return (U_dyn + U_obs + U_prior) - (H_x + H_u + H_theta) + dyn_intersection + obs_intersection


# =============================================================================
# Convergence planning functions
# =============================================================================


@partial(jax.jit, static_argnums=(4, 5))
def loopy_bp_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    goal,
    horizon,
    n_iterations,
    action_prior=None,
):
    """Loopy BP planning with per-iteration VFE trace.

    Returns:
        action_dist: (n_actions,)
        vfe_trace: (n_iterations,)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]

    log_T = safe_log(transition_tensor)
    log_prior_theta = safe_log(q_static_state)
    log_q0 = safe_log(q_current_state)
    has_pref = goal.ndim == 2

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    log_cavity_theta = jnp.tile(log_prior_theta, (horizon, 1))
    q_u_init = jnp.zeros((horizon, n_actions))
    vfe_trace = jnp.zeros(n_iterations)

    if has_pref:
        # 2D goal: per-config preference C(x, θ)
        log_C = safe_log(goal)
        log_dyn_to_theta_init = jnp.zeros((horizon, n_static))

        def body_fn(i, carry):
            log_cavity_theta, _, log_dyn_to_theta_prev, vfe_trace = carry

            log_reduced_per_t = compute_reduced_per_t(log_T, log_cavity_theta)

            # Full theta belief from previous iteration's messages
            log_q_theta = log_prior_theta + log_dyn_to_theta_prev.sum(axis=0)
            log_q_theta = log_q_theta - logsumexp(log_q_theta)

            # Per-step preference: marginalize C(x,θ) with full theta belief
            log_pref = logsumexp(log_C + log_q_theta[None, :], axis=1)
            log_pref = log_pref - logsumexp(log_pref)

            # Forward pass with per-step preference injected
            log_fwd = jnp.zeros((horizon + 1, n_states))
            log_fwd = log_fwd.at[0].set(log_q0)

            def fwd_body(t, lf):
                log_terms = (log_reduced_per_t[t]
                             + (lf[t] + log_pref)[None, :, None]
                             + log_action_prior[None, None, :])
                log_q_next = logsumexp(log_terms, axis=(1, 2))
                log_q_next = log_q_next - logsumexp(log_q_next)
                return lf.at[t + 1].set(log_q_next)

            log_fwd = lax.fori_loop(0, horizon, fwd_body, log_fwd)

            # Backward pass with per-step preference, uniform terminal
            def bwd_body(carry_bwd, t):
                log_bwd_val, q_u, log_bwd_arr = carry_bwd
                log_bwd_with_pref = log_bwd_val + log_pref

                # Action marginal
                log_terms = (log_reduced_per_t[t]
                             + log_bwd_with_pref[:, None, None]
                             + (log_fwd[t] + log_pref)[None, :, None])
                log_msg_to_u = logsumexp(log_terms, axis=(0, 1))
                q_u_t = jax.nn.softmax(log_msg_to_u + log_action_prior)
                q_u = q_u.at[t].set(q_u_t)

                # Backward message to x_t
                log_terms_bwd = (log_reduced_per_t[t]
                                 + log_bwd_with_pref[:, None, None]
                                 + log_action_prior[None, None, :])
                log_bwd_t = logsumexp(log_terms_bwd, axis=(0, 2))
                log_bwd_t = log_bwd_t - logsumexp(log_bwd_t)
                log_bwd_arr = log_bwd_arr.at[t].set(log_bwd_t)

                return (log_bwd_t, q_u, log_bwd_arr), None

            log_bwd_init = jnp.zeros((horizon + 1, n_states))
            (_, q_u, log_bwd), _ = lax.scan(
                bwd_body,
                (jnp.zeros(n_states), jnp.zeros((horizon, n_actions)), log_bwd_init),
                jnp.arange(horizon - 1, -1, -1),
            )

            # dyn_to_theta: include preference in x beliefs
            log_fwd_t = (log_fwd[:-1] + log_pref[None, :])[:, None, :, None, None]
            log_bwd_t1 = (log_bwd[1:] + log_pref[None, :])[:, :, None, None, None]
            terms = (log_T[None]
                     + log_fwd_t
                     + log_bwd_t1
                     + log_action_prior[None, None, None, None, :])
            new_dyn_to_theta = logsumexp(terms, axis=(1, 2, 4))

            new_log_cavity = compute_theta_cavities(log_prior_theta, new_dyn_to_theta)

            vfe = compute_bethe_vfe_loopy(log_T, log_reduced_per_t, log_fwd, log_bwd,
                                           q_u, log_cavity_theta, log_prior_theta,
                                           log_action_prior)
            vfe_trace = vfe_trace.at[i].set(vfe)

            return new_log_cavity, q_u, new_dyn_to_theta, vfe_trace

        log_cavity_theta, q_u, _, vfe_trace = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_cavity_theta, q_u_init, log_dyn_to_theta_init, vfe_trace)
        )
    else:
        # 1D goal: original terminal-goal behavior
        log_goal = safe_log(goal)

        def body_fn(i, carry):
            log_cavity_theta, _, vfe_trace = carry

            log_reduced_per_t = compute_reduced_per_t(log_T, log_cavity_theta)
            log_fwd = loopy_forward_pass(log_reduced_per_t, log_q0, log_action_prior, horizon)
            log_bwd, q_u = loopy_backward_pass(
                log_reduced_per_t, log_fwd, log_goal, log_action_prior, horizon
            )
            log_dyn_to_theta = loopy_dyn_to_theta(
                log_T, log_fwd, log_bwd, log_action_prior, horizon
            )
            new_log_cavity = compute_theta_cavities(log_prior_theta, log_dyn_to_theta)

            vfe = compute_bethe_vfe_loopy(log_T, log_reduced_per_t, log_fwd, log_bwd,
                                           q_u, log_cavity_theta, log_prior_theta,
                                           log_action_prior)
            vfe_trace = vfe_trace.at[i].set(vfe)

            return new_log_cavity, q_u, vfe_trace

        log_cavity_theta, q_u, vfe_trace = lax.fori_loop(
            0, n_iterations, body_fn, (log_cavity_theta, q_u_init, vfe_trace)
        )

    return q_u[0], vfe_trace


@partial(jax.jit, static_argnums=(5, 6))
def region_extended_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    observation_tensor,
    goal,
    horizon,
    n_iterations,
    damping=1.0,
    action_prior=None,
):
    """Region-extended loopy BP planning with per-iteration VFE trace.

    Returns:
        action_dist: (n_actions,)
        log_dyn_channels: (T, n_states, n_states, n_actions)
        log_obs_channels: (T+1, n_channels, n_obs_types, n_states, n_static)
        vfe_trace: (n_iterations,)
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

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)

    if has_pref:
        log_C = safe_log(goal)                            # (n_states, n_static)
        log_goal = jnp.zeros(n_states)                    # uniform terminal
    else:
        log_C = None
        log_goal = safe_log(goal)

    log_prior_dyn = jnp.broadcast_to(log_prior_theta[None, :], (horizon, n_static))
    log_prior_obs = jnp.broadcast_to(log_prior_theta[None, :], (horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))

    # Initial dyn channels: r(x_new | x_old, u) from θ-marginalized transition
    log_dyn_ch0 = logsumexp(log_T + log_prior_theta[None, None, :, None], axis=2)
    log_dyn_ch0 = log_dyn_ch0 - logsumexp(log_dyn_ch0, axis=0, keepdims=True)
    log_dyn_ch0 = log_dyn_ch0.transpose(1, 0, 2)
    log_dyn_channels_init = jnp.broadcast_to(log_dyn_ch0[None], (horizon, n_states, n_states, n_actions))

    # Initial obs channels: r(y | x, θ) = B(y | x, θ)
    log_obs_ch0 = log_B_flat - logsumexp(log_B_flat, axis=1, keepdims=True)
    log_obs_channels_init = jnp.broadcast_to(log_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static))

    # Initial marginal obs channels: r(y | x) by marginalizing θ with prior weighting
    log_marginal_obs_ch0 = logsumexp(
        log_B_flat + log_prior_theta[None, None, None, :], axis=3)
    log_marginal_obs_ch0 = log_marginal_obs_ch0 - logsumexp(log_marginal_obs_ch0, axis=1, keepdims=True)
    log_marginal_obs_channels_init = jnp.broadcast_to(
        log_marginal_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states))

    vfe_trace = jnp.zeros(n_iterations)
    log_fwd_prev_init = jnp.zeros((horizon + 1, n_states))
    log_bwd_prev_init = jnp.zeros((horizon + 1, n_states))

    if has_pref:
        def body_fn(i, carry):
            (_, log_dyn_channels, log_obs_channels, log_marginal_obs_channels,
             log_fwd_prev, log_bwd_prev,
             vfe_trace) = carry

            log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_prior_dyn)

            # obs->x and pref->x messages
            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_prior_obs)
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_prior_obs)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping
            )
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_local_to_x, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping
            )

            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_prior_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_prior_obs
            )

            raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)

            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)
            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            vfe = compute_region_extended_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_prior_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions, log_obs_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (q_u, new_log_dyn_channels, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs,
                    vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init,
             vfe_trace)
        )
        q_u, log_dyn_channels, log_obs_channels, _, _, _, vfe_trace = result
    else:
        def body_fn(i, carry):
            (_, log_dyn_channels,
             log_obs_channels, log_marginal_obs_channels,
             log_fwd_prev, log_bwd_prev,
             vfe_trace) = carry

            log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_prior_dyn)
            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_prior_obs)

            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping
            )
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_obs_to_x, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping
            )

            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_prior_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_prior_obs
            )

            raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)

            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)
            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            vfe = compute_region_extended_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_prior_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions, log_obs_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (q_u, new_log_dyn_channels, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs,
                    vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init,
             vfe_trace)
        )
        q_u, log_dyn_channels, log_obs_channels, _, _, _, vfe_trace = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_dyn_channels, log_obs_channels, vfe_trace


@partial(jax.jit, static_argnums=(5, 6))
def nuijten_mp_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    observation_tensor,
    goal,
    horizon,
    n_iterations,
    action_prior=None,
):
    """Nuijten MP planning (theta inferred) with per-iteration VFE trace.

    Returns:
        action_dist: (n_actions,)
        log_dyn_region_beliefs: (T, x_old, x_new, theta, u)
        obs_region_beliefs: (T+1, n_channels, n_obs_types, n_states, n_static)
        vfe_trace: (n_iterations,)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]
    n_fov = observation_tensor.shape[0]
    n_obs_types = observation_tensor.shape[1]
    has_pref = goal.ndim == 2

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    action_mask = (action_prior > 0).astype(jnp.float32)

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_T_kernel_tiled = jnp.broadcast_to(
        log_T_kernel[None], (horizon, n_states, n_states, n_static, n_actions)
    )
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)

    if has_pref:
        log_C = safe_log(goal)                            # (n_states, n_static)
        log_goal = jnp.zeros(n_states)                    # uniform terminal
    else:
        log_C = None
        log_goal = safe_log(goal)

    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    action_prior_init = jnp.tile(action_prior, (horizon, 1))
    log_dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, n_obs_types, n_states, n_static))
    vfe_trace = jnp.zeros(n_iterations)

    if has_pref:
        log_pref_to_theta_init = jnp.zeros((horizon + 1, n_static))

        def body_fn(i, carry):
            log_dyn_to_theta, _, action_prior_per_t, _, obs_regions, log_pref_to_theta, vfe_trace = carry

            log_action_prior_per_t = safe_log(action_prior_per_t)

            log_obs_to_x = compute_obs_efe_to_x(obs_regions)
            log_obs_to_theta = compute_obs_efe_to_theta(obs_regions)

            # 3-way theta cavities
            log_cavity_dyn, log_cavity_obs, log_cavity_pref = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta
            )

            log_reduced_per_t = compute_log_reduced(log_T_kernel_tiled, log_cavity_dyn)

            # pref->x messages and combined local messages
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_cavity_pref)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            log_fwd_msgs = forward_pass_nuijten(
                log_reduced_per_t, log_q0, log_action_prior_per_t, log_local_to_x, horizon
            )
            log_bwd_msgs, q_u = backward_pass_nuijten(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior_per_t,
                log_local_to_x, horizon
            )

            new_log_dyn_to_theta = compute_dyn_to_theta_msgs_nuijten(
                log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_action_prior_per_t, horizon
            )

            # pref->theta messages
            new_log_pref_to_theta = compute_pref_to_theta_msgs(
                log_C, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            log_dyn_regions = compute_dyn_region_beliefs_nuijten(
                log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_cavity_dyn, log_action_prior_per_t
            )
            obs_regions = compute_obs_region_beliefs_original(
                log_B_flat, log_fwd_msgs, log_bwd_msgs, log_cavity_obs
            )

            new_action_prior = compute_efe_action_prior(log_dyn_regions, action_mask)

            vfe = compute_nuijten_vfe(
                log_dyn_regions, obs_regions,
                log_fwd_msgs, log_bwd_msgs, q_u,
                log_T_kernel_tiled, log_B_flat,
                log_cavity_dyn, log_cavity_obs,
                log_prior_theta, action_prior_per_t,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (new_log_dyn_to_theta, q_u, new_action_prior, log_dyn_regions,
                    obs_regions, new_log_pref_to_theta, vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, q_u_init, action_prior_init,
             log_dyn_regions_init, obs_regions_init, log_pref_to_theta_init, vfe_trace)
        )
        _, q_u, _, log_dyn_region_beliefs, obs_region_beliefs, _, vfe_trace = result
    else:
        def body_fn(i, carry):
            log_dyn_to_theta, _, action_prior_per_t, _, obs_regions, vfe_trace = carry

            log_action_prior_per_t = safe_log(action_prior_per_t)

            log_obs_to_x = compute_obs_efe_to_x(obs_regions)
            log_obs_to_theta = compute_obs_efe_to_theta(obs_regions)

            log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta
            )

            log_reduced_per_t = compute_log_reduced(log_T_kernel_tiled, log_cavity_dyn)

            log_fwd_msgs = forward_pass_nuijten(
                log_reduced_per_t, log_q0, log_action_prior_per_t, log_obs_to_x, horizon
            )
            log_bwd_msgs, q_u = backward_pass_nuijten(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior_per_t,
                log_obs_to_x, horizon
            )

            new_log_dyn_to_theta = compute_dyn_to_theta_msgs_nuijten(
                log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_action_prior_per_t, horizon
            )

            log_dyn_regions = compute_dyn_region_beliefs_nuijten(
                log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_cavity_dyn, log_action_prior_per_t
            )
            obs_regions = compute_obs_region_beliefs_original(
                log_B_flat, log_fwd_msgs, log_bwd_msgs, log_cavity_obs
            )

            new_action_prior = compute_efe_action_prior(log_dyn_regions, action_mask)

            vfe = compute_nuijten_vfe(
                log_dyn_regions, obs_regions,
                log_fwd_msgs, log_bwd_msgs, q_u,
                log_T_kernel_tiled, log_B_flat,
                log_cavity_dyn, log_cavity_obs,
                log_prior_theta, action_prior_per_t,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return new_log_dyn_to_theta, q_u, new_action_prior, log_dyn_regions, obs_regions, vfe_trace

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, q_u_init, action_prior_init,
             log_dyn_regions_init, obs_regions_init, vfe_trace)
        )
        _, q_u, _, log_dyn_region_beliefs, obs_region_beliefs, vfe_trace = result

    return q_u[0], log_dyn_region_beliefs, obs_region_beliefs, vfe_trace


# =============================================================================
# Dyn-channel convergence (theta inferred)
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def dyn_channel_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    observation_tensor,
    goal,
    horizon,
    n_iterations,
    damping=1.0,
    action_prior=None,
):
    """Dyn-channel loopy BP planning with per-iteration VFE trace.

    Returns:
        action_dist: (n_actions,)
        log_dyn_channels: (T, n_states, n_states, n_actions)
        vfe_trace: (n_iterations,)
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

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
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

    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    log_obs_to_theta = jnp.zeros((horizon + 1, n_static))
    log_pref_to_theta_init = jnp.zeros((horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))

    # Initial dyn channels: r(x_new | x_old, u) from θ-marginalized transition
    log_dyn_ch0 = logsumexp(log_T + log_prior_theta[None, None, :, None], axis=2)
    log_dyn_ch0 = log_dyn_ch0 - logsumexp(log_dyn_ch0, axis=0, keepdims=True)
    log_dyn_ch0 = log_dyn_ch0.transpose(1, 0, 2)
    log_dyn_channels_init = jnp.broadcast_to(log_dyn_ch0[None], (horizon, n_states, n_states, n_actions))

    vfe_trace = jnp.zeros(n_iterations)

    if has_pref:
        def body_fn(i, carry):
            (log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta, _,
             log_dyn_channels, vfe_trace) = carry

            # Step 1: 3-way theta cavities
            log_cavity_dyn, log_cavity_obs, log_cavity_pref = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta
            )

            # Step 2: Dyn kernels
            log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])

            # Step 3: Reduced tensors
            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

            # Step 4: obs->x and pref->x messages
            log_obs_to_x = compute_obs_to_x_msgs(log_B_tiled, log_cavity_obs)
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_cavity_pref)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            # Step 5: Forward pass
            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon
            )

            # Step 6: Backward pass
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_local_to_x, horizon
            )

            # Step 7: dyn->theta messages
            new_log_dyn_to_theta = re_dyn_to_theta(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_action_prior, horizon
            )

            # Step 8: obs->theta messages
            new_log_obs_to_theta = compute_obs_to_theta_msgs(
                log_B_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_extra_to_x=log_pref_to_x
            )

            # Step 9: pref->theta messages
            new_log_pref_to_theta = compute_pref_to_theta_msgs(
                log_C, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            # Step 10: Dyn region beliefs -> extract dyn channels -> damped update
            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_cavity_dyn, log_action_prior
            )
            raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)

            vfe = compute_dyn_channel_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_cavity_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (new_log_dyn_to_theta, new_log_obs_to_theta, new_log_pref_to_theta,
                    q_u, new_log_dyn_channels, vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta_init,
             q_u_init, log_dyn_channels_init, vfe_trace)
        )
        _, _, _, q_u, log_dyn_channels, vfe_trace = result
    else:
        def body_fn(i, carry):
            (log_dyn_to_theta, log_obs_to_theta, _, log_dyn_channels,
             vfe_trace) = carry

            log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta
            )

            log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])

            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)
            log_obs_to_x = compute_obs_to_x_msgs(log_B_tiled, log_cavity_obs)

            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon
            )
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_obs_to_x, horizon
            )

            new_log_dyn_to_theta = re_dyn_to_theta(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_action_prior, horizon
            )
            new_log_obs_to_theta = compute_obs_to_theta_msgs(
                log_B_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_cavity_dyn, log_action_prior
            )

            raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)

            vfe = compute_dyn_channel_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_cavity_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (new_log_dyn_to_theta, new_log_obs_to_theta, q_u,
                    new_log_dyn_channels, vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, log_obs_to_theta, q_u_init,
             log_dyn_channels_init, vfe_trace)
        )
        _, _, q_u, log_dyn_channels, vfe_trace = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_dyn_channels, vfe_trace


# =============================================================================
# VBP-channel convergence
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def vbp_channel_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    observation_tensor,
    goal,
    horizon,
    n_iterations,
    damping=1.0,
    action_prior=None,
):
    """VBP channel planning with per-iteration VFE trace.

    Returns:
        action_dist: (n_actions,)
        log_r_ux: (T, n_states, n_actions)
        vfe_trace: (n_iterations,)
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

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)

    if has_pref:
        log_C = safe_log(goal)
        log_goal = jnp.zeros(n_states)
    else:
        log_C = None
        log_goal = safe_log(goal)

    # Tile obs tensor over time: kernel = raw B (no obs channels)
    log_B_tiled = jnp.broadcast_to(
        log_B_flat[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static)
    )

    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    log_obs_to_theta = jnp.zeros((horizon + 1, n_static))
    log_pref_to_theta_init = jnp.zeros((horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))

    # Initial action channels: uniform r(u|x) = 1/n_actions
    log_r_ux_init = jnp.full((horizon, n_states, n_actions), -jnp.log(n_actions))

    vfe_trace = jnp.zeros(n_iterations)

    if has_pref:
        def body_fn(i, carry):
            (log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta, _,
             log_r_ux, vfe_trace) = carry

            log_cavity_dyn, log_cavity_obs, log_cavity_pref = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta
            )

            log_dyn_kernels = compute_dyn_kernels_vbp(log_T_kernel, log_r_ux)
            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

            log_obs_to_x = compute_obs_to_x_msgs(log_B_tiled, log_cavity_obs)
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_cavity_pref)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon
            )
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_local_to_x, horizon
            )

            new_log_dyn_to_theta = re_dyn_to_theta(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_action_prior, horizon
            )
            new_log_obs_to_theta = compute_obs_to_theta_msgs(
                log_B_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_extra_to_x=log_pref_to_x
            )
            new_log_pref_to_theta = compute_pref_to_theta_msgs(
                log_C, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_cavity_dyn, log_action_prior
            )
            log_pair = compute_pair_marginal(log_dyn_regions)
            raw_log_r_ux = compute_action_channel(log_pair)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            vfe = compute_vbp_channel_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_cavity_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (new_log_dyn_to_theta, new_log_obs_to_theta, new_log_pref_to_theta,
                    q_u, new_log_r_ux, vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta_init,
             q_u_init, log_r_ux_init, vfe_trace)
        )
        _, _, _, q_u, log_r_ux, vfe_trace = result
    else:
        def body_fn(i, carry):
            log_dyn_to_theta, log_obs_to_theta, _, log_r_ux, vfe_trace = carry

            log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta
            )

            log_dyn_kernels = compute_dyn_kernels_vbp(log_T_kernel, log_r_ux)
            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

            log_obs_to_x = compute_obs_to_x_msgs(log_B_tiled, log_cavity_obs)

            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon
            )
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_obs_to_x, horizon
            )

            new_log_dyn_to_theta = re_dyn_to_theta(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_action_prior, horizon
            )
            new_log_obs_to_theta = compute_obs_to_theta_msgs(
                log_B_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_cavity_dyn, log_action_prior
            )
            log_pair = compute_pair_marginal(log_dyn_regions)
            raw_log_r_ux = compute_action_channel(log_pair)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            vfe = compute_vbp_channel_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_cavity_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (new_log_dyn_to_theta, new_log_obs_to_theta, q_u,
                    new_log_r_ux, vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, log_obs_to_theta, q_u_init,
             log_r_ux_init, vfe_trace)
        )
        _, _, q_u, log_r_ux, vfe_trace = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_r_ux, vfe_trace


# =============================================================================
# Precise info-seeking convergence
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def precise_info_seeking_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    observation_tensor,
    goal,
    horizon,
    n_iterations,
    damping=1.0,
    action_prior=None,
):
    """Precise info-seeking planning with per-iteration VFE trace.

    Returns:
        action_dist: (n_actions,)
        log_r_ux: (T, n_states, n_actions)
        log_obs_channels: (T+1, n_channels, n_obs_types, n_states, n_static)
        vfe_trace: (n_iterations,)
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

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)

    if has_pref:
        log_C = safe_log(goal)
        log_goal = jnp.zeros(n_states)
    else:
        log_C = None
        log_goal = safe_log(goal)

    log_prior_dyn = jnp.broadcast_to(log_prior_theta[None, :], (horizon, n_static))
    log_prior_obs = jnp.broadcast_to(log_prior_theta[None, :], (horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))

    # Initial action channels: uniform r(u|x) = 1/n_actions
    log_r_ux_init = jnp.full((horizon, n_states, n_actions), -jnp.log(n_actions))

    # Initial obs channels: r(y | x, θ) = B(y | x, θ)
    log_obs_ch0 = log_B_flat - logsumexp(log_B_flat, axis=1, keepdims=True)
    log_obs_channels_init = jnp.broadcast_to(
        log_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static))

    # Initial marginal obs channels: r(y | x) by marginalizing θ with prior
    log_marginal_obs_ch0 = logsumexp(
        log_B_flat + log_prior_theta[None, None, None, :], axis=3)
    log_marginal_obs_ch0 = log_marginal_obs_ch0 - logsumexp(
        log_marginal_obs_ch0, axis=1, keepdims=True)
    log_marginal_obs_channels_init = jnp.broadcast_to(
        log_marginal_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states))

    vfe_trace = jnp.zeros(n_iterations)
    log_fwd_prev_init = jnp.zeros((horizon + 1, n_states))
    log_bwd_prev_init = jnp.zeros((horizon + 1, n_states))

    if has_pref:
        def body_fn(i, carry):
            (_, log_r_ux, log_obs_channels, log_marginal_obs_channels,
             log_fwd_prev, log_bwd_prev,
             vfe_trace) = carry

            log_dyn_kernels = compute_dyn_kernels_vbp(log_T_kernel, log_r_ux)
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_prior_dyn)

            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_prior_obs)
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_prior_obs)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping
            )
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_local_to_x, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping
            )

            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_prior_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_prior_obs
            )

            # Action channels (VBP style)
            log_pair = compute_pair_marginal(log_dyn_regions)
            raw_log_r_ux = compute_action_channel(log_pair)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            # Obs channels (region-extended style)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)
            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            vfe = compute_precise_info_seeking_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_prior_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions, log_obs_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (q_u, new_log_r_ux, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs,
                    vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_r_ux_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init,
             vfe_trace)
        )
        q_u, log_r_ux, log_obs_channels, _, _, _, vfe_trace = result
    else:
        def body_fn(i, carry):
            (_, log_r_ux, log_obs_channels, log_marginal_obs_channels,
             log_fwd_prev, log_bwd_prev,
             vfe_trace) = carry

            log_dyn_kernels = compute_dyn_kernels_vbp(log_T_kernel, log_r_ux)
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_prior_dyn)
            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_prior_obs)

            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping
            )
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_obs_to_x, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping
            )

            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_prior_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_prior_obs
            )

            # Action channels (VBP style)
            log_pair = compute_pair_marginal(log_dyn_regions)
            raw_log_r_ux = compute_action_channel(log_pair)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            # Obs channels (region-extended style)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)
            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            vfe = compute_precise_info_seeking_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_prior_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions, log_obs_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (q_u, new_log_r_ux, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs,
                    vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_r_ux_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init,
             vfe_trace)
        )
        q_u, log_r_ux, log_obs_channels, _, _, _, vfe_trace = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_r_ux, log_obs_channels, vfe_trace


# =============================================================================
# Active Inference VFE and convergence
# =============================================================================


def compute_active_inference_vfe(log_T, log_reduced_per_t, log_fwd_msgs,
                                  log_bwd_msgs, q_u, log_cavity_dyn,
                                  log_prior_theta, log_action_prior,
                                  log_dyn_regions, log_obs_regions):
    """VFE for Active Inference: F_p + action_cond_entropy + obs_correction - dyn_cond_entropy."""
    horizon = q_u.shape[0]
    F_p = compute_bethe_vfe_loopy(log_T, log_reduced_per_t, log_fwd_msgs,
                                   log_bwd_msgs, q_u, log_cavity_dyn,
                                   log_prior_theta, log_action_prior)
    return (F_p
            + _action_cond_entropy(log_dyn_regions, horizon)
            + _obs_cond_entropy(log_obs_regions, horizon)
            - _dyn_cond_entropy(log_dyn_regions, horizon))


def active_inference_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    observation_tensor,
    goal,
    horizon,
    n_iterations,
    damping=1.0,
    action_prior=None,
):
    """Active Inference planning with per-iteration VFE trace.

    Returns:
        action_dist: (n_actions,)
        log_dyn_channels: (T, n_states, n_states, n_actions)
        log_obs_channels: (T+1, n_channels, n_obs_types, n_states, n_static)
        vfe_trace: (n_iterations,)
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

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)

    if has_pref:
        log_C = safe_log(goal)
        log_goal = jnp.zeros(n_states)
    else:
        log_C = None
        log_goal = safe_log(goal)

    log_prior_dyn = jnp.broadcast_to(log_prior_theta[None, :], (horizon, n_static))
    log_prior_obs = jnp.broadcast_to(log_prior_theta[None, :], (horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))

    # Initial action channels: uniform r(u|x) = 1/n_actions
    log_r_ux_init = jnp.full((horizon, n_states, n_actions), -jnp.log(n_actions))

    # Initial dyn channels: r(x_new | x_old, u) from θ-marginalized transition
    log_dyn_ch0 = logsumexp(log_T + log_prior_theta[None, None, :, None], axis=2)
    log_dyn_ch0 = log_dyn_ch0 - logsumexp(log_dyn_ch0, axis=0, keepdims=True)
    log_dyn_ch0 = log_dyn_ch0.transpose(1, 0, 2)
    log_dyn_channels_init = jnp.broadcast_to(log_dyn_ch0[None], (horizon, n_states, n_states, n_actions))

    # Initial obs channels: r(y | x, θ) = B(y | x, θ)
    log_obs_ch0 = log_B_flat - logsumexp(log_B_flat, axis=1, keepdims=True)
    log_obs_channels_init = jnp.broadcast_to(
        log_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static))

    # Initial marginal obs channels: r(y | x) by marginalizing θ with prior
    log_marginal_obs_ch0 = logsumexp(
        log_B_flat + log_prior_theta[None, None, None, :], axis=3)
    log_marginal_obs_ch0 = log_marginal_obs_ch0 - logsumexp(
        log_marginal_obs_ch0, axis=1, keepdims=True)
    log_marginal_obs_channels_init = jnp.broadcast_to(
        log_marginal_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states))

    vfe_trace = jnp.zeros(n_iterations)
    log_fwd_prev_init = jnp.zeros((horizon + 1, n_states))
    log_bwd_prev_init = jnp.zeros((horizon + 1, n_states))

    if has_pref:
        def body_fn(i, carry):
            (_, log_r_ux, log_dyn_channels, log_obs_channels, log_marginal_obs_channels,
             log_fwd_prev, log_bwd_prev,
             vfe_trace) = carry

            log_dyn_kernels = compute_dyn_kernels_aif(log_T_kernel, log_r_ux, log_dyn_channels)
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_prior_dyn)

            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_prior_obs)
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_prior_obs)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping
            )
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_local_to_x, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping
            )

            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_prior_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_prior_obs
            )

            # Action channels (VBP style)
            log_pair = compute_pair_marginal(log_dyn_regions)
            raw_log_r_ux = compute_action_channel(log_pair)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            # Dyn channels (region-extended style)
            raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)

            # Obs channels (region-extended style)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)
            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            vfe = compute_active_inference_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_prior_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions, log_obs_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (q_u, new_log_r_ux, new_log_dyn_channels, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs,
                    vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_r_ux_init, log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init,
             vfe_trace)
        )
        q_u, _, log_dyn_channels, log_obs_channels, _, _, _, vfe_trace = result
    else:
        def body_fn(i, carry):
            (_, log_r_ux, log_dyn_channels, log_obs_channels, log_marginal_obs_channels,
             log_fwd_prev, log_bwd_prev,
             vfe_trace) = carry

            log_dyn_kernels = compute_dyn_kernels_aif(log_T_kernel, log_r_ux, log_dyn_channels)
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_prior_dyn)
            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_prior_obs)

            log_fwd_msgs = re_forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping
            )
            log_bwd_msgs, q_u = re_backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_obs_to_x, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping
            )

            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_prior_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_prior_obs
            )

            # Action channels (VBP style)
            log_pair = compute_pair_marginal(log_dyn_regions)
            raw_log_r_ux = compute_action_channel(log_pair)
            new_log_r_ux = damp_log_channel(
                log_r_ux, raw_log_r_ux, damping, cond_axis=2)

            # Dyn channels (region-extended style)
            raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)

            # Obs channels (region-extended style)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)
            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2)

            vfe = compute_active_inference_vfe(
                log_T, log_reduced_per_t, log_fwd_msgs, log_bwd_msgs, q_u,
                log_prior_dyn, log_prior_theta, log_action_prior,
                log_dyn_regions, log_obs_regions,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (q_u, new_log_r_ux, new_log_dyn_channels, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs,
                    vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_r_ux_init, log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init,
             vfe_trace)
        )
        q_u, _, log_dyn_channels, log_obs_channels, _, _, _, vfe_trace = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_dyn_channels, log_obs_channels, vfe_trace
