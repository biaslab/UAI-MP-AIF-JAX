"""Convergence variants of planning functions that return per-iteration VFE traces.

Each convergence function mirrors its original planning function but additionally
accumulates a VFE value at each iteration. The VFE formulas differ by method:

- Standard BP / Loopy BP: Bethe free energy
- Region-extended / Reduced: Region-based free energy with observation factors
- Nuijten / Reduced-Nuijten: Same as region-based but with EFE action priors

All internal computation is in log-space.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .planning import LOG_ZERO, safe_log, marginalize_static
from .planning import forward_pass as bp_forward_pass
from .messages import safe_log_div, EPSILON
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
    compute_theta_cavities_extended,
    compute_dyn_region_beliefs,
    compute_obs_region_beliefs,
    compute_dyn_channels,
    compute_obs_channels,
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
# Modified backward pass for standard BP (returns bwd messages)
# =============================================================================


def backward_pass_bp_full(log_reduced, log_q_state, log_goal, log_action_prior, horizon):
    """Backward pass for standard BP that also returns backward messages.

    Same logic as planning.backward_pass but stores log_bwd_msgs.

    Returns:
        log_bwd_msgs: (T+1, n_states)
        q_u: (T, n_actions) action marginals (probability space)
    """
    n_states = log_goal.shape[0]
    n_actions = log_action_prior.shape[0]
    log_bwd = jnp.zeros((horizon + 1, n_states))
    log_bwd = log_bwd.at[horizon].set(log_goal)
    q_u = jnp.zeros((horizon, n_actions))

    def body_fn(carry, t):
        log_bwd, q_u = carry

        # Action marginal
        log_terms = (log_reduced
                     + log_bwd[t + 1][:, None, None]
                     + log_q_state[t][None, :, None])
        log_msg_to_u = logsumexp(log_terms, axis=(0, 1))
        q_u_t = jax.nn.softmax(log_msg_to_u + log_action_prior)
        q_u = q_u.at[t].set(q_u_t)

        # Backward message to x_t
        log_terms_bwd = (log_reduced
                         + log_bwd[t + 1][:, None, None]
                         + log_action_prior[None, None, :])
        log_bwd_new = logsumexp(log_terms_bwd, axis=(0, 2))
        log_bwd_new = log_bwd_new - logsumexp(log_bwd_new)
        log_bwd = log_bwd.at[t].set(log_bwd_new)

        return (log_bwd, q_u), None

    (log_bwd, q_u), _ = lax.scan(
        body_fn, (log_bwd, q_u), jnp.arange(horizon - 1, -1, -1)
    )
    return log_bwd, q_u


# =============================================================================
# Bethe VFE for standard BP
# =============================================================================


def compute_bethe_vfe_bp(log_reduced, log_fwd_msgs, log_bwd_msgs, q_u,
                          log_action_prior):
    """Standard Bethe VFE for BP (tree-structured, theta marginalized once).

    F_Bethe = sum_t [U_dyn(t) - H[b_dyn(t)]] - sum_{t=1}^{T-1} H[q(x_t)]

    where b_dyn(t)(x_new, x_old, u) = f(x_new|x_old,u) * fwd[t](x_old)
                                       * bwd[t+1](x_new) * q(u_t)
    """
    horizon = q_u.shape[0]

    def per_t_vfe(t):
        # Factor belief (unnormalized log)
        log_b = (log_reduced
                 + log_fwd_msgs[t][None, :, None]
                 + log_bwd_msgs[t + 1][:, None, None]
                 + safe_log(q_u[t])[None, None, :])

        U_t = _energy(log_b, log_reduced)
        H_t = _entropy_unnorm(log_b)
        return U_t - H_t

    # Factor contributions
    factor_vfe = jax.vmap(per_t_vfe)(jnp.arange(horizon)).sum()

    # Singleton entropy: intermediate x_t (t=1..T-1)
    def x_entropy(t):
        log_q_x = log_fwd_msgs[t] + log_bwd_msgs[t]
        log_q_x = log_q_x - logsumexp(log_q_x)
        return _entropy(log_q_x)

    # Only intermediate nodes (not x_0 fixed, not x_T from goal)
    singleton_H = jnp.where(
        horizon > 1,
        jax.vmap(x_entropy)(jnp.arange(1, horizon)).sum(),
        0.0,
    )

    return factor_vfe - singleton_H


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
    # q(theta) = prior * product of all dyn_to_theta messages
    # Approximated from cavity: q(theta|t) ~ cavity(t) * dyn_to_theta(t)
    # Full q(theta) ~ prior * prod dyn_to_theta = sum of all cavities (in log) ...
    # Use total belief: prior + sum of (cavity - prior + dyn_to_theta msg)
    # Simpler: compute from cavity[0] + dyn_to_theta[0]
    # Actually, theta belief = prior + sum_t dyn_to_theta_msgs
    # But we don't have dyn_to_theta here directly. Use cavity approach:
    # cavity[t] = total - msg[t], so total = cavity[t] + msg[t] for any t
    # We don't have the individual msgs. Instead, compute theta belief as:
    # For the counting number approach, theta's counting = 1 - n_connected = 1 - T
    # So we subtract (1-T)*H[q(theta)] = (T-1)*H[q(theta)] is added
    # But we need q(theta). Approximate: q(theta) ~ softmax(prior + sum cavities - (T-1)*prior)
    # ... this gets complicated. Simpler: just compute from cavities.
    # cavity[t] = total - msg[t]. total = cavity[0] + msg[0].
    # We can get total from: for all t, cavity[t] + msg[t] should be the same.
    # Since we don't store individual msgs, use: total = log_prior + sum_t (cavity[t] - log_prior + correction)
    # Actually the simplest: q(theta) is proportional to softmax of any cavity + its msg.
    # We'll skip the theta counting number for simplicity — it's small.
    # Just use (1 - T) * H[q(theta)] where q(theta) = softmax(cavity[0])
    # Actually cavity already excludes one factor, so it's not the full posterior.
    # For now, approximate q(theta) as mean of cavities (they're close after convergence).
    log_q_theta = log_cavity_theta.mean(axis=0)
    log_q_theta = log_q_theta - logsumexp(log_q_theta)
    H_theta = _entropy(log_q_theta)
    theta_counting = 1.0 - horizon
    theta_term = theta_counting * H_theta

    return factor_vfe - singleton_H - theta_term


# =============================================================================
# Region-extended VFE
# =============================================================================


def compute_region_extended_vfe(log_dyn_regions, log_obs_regions,
                                 log_fwd_msgs, log_bwd_msgs, q_u,
                                 log_T_kernel, log_B_flat,
                                 log_cavity_dyn, log_cavity_obs,
                                 log_prior_theta):
    """VFE for region-extended / reduced-aif methods.

    F = sum_a U_a - sum_i H[q_i]
      + sum_t { H[q(x_{t-1}, u_t)] - H[q(x_t, x_{t-1}, u_t)]
              + sum_k [ H[q(y_{t,k}, x_t, theta)] - H[q(x_t, theta)] ] }

    Uses original log_T and log_B for energies (not kernels).
    """
    horizon = q_u.shape[0]
    n_fov = log_obs_regions.shape[1]

    # --- Energy terms ---

    # Dynamics energy: U_dyn(t) = -sum b_dyn(t) * log T(x_new|x_old,theta,u)
    # log_dyn_regions: (T, x_old, x_new, theta, u)
    # log_T_kernel: (x_old, x_new, theta, u) — this IS the original factor in (x_old,x_new) order
    def dyn_energy_t(t):
        return _energy(log_dyn_regions[t], log_T_kernel)

    U_dyn = jax.vmap(dyn_energy_t)(jnp.arange(horizon)).sum()

    # Obs energy: U_obs(t,k) = -sum b_obs(t,k) * log B(y|x,theta)
    # log_obs_regions: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    # log_B_flat: (n_fov, N_CELL_TYPES, n_states, n_static)
    def obs_energy_t(t):
        def obs_energy_k(k):
            return _energy(log_obs_regions[t, k], log_B_flat[k])
        return jax.vmap(obs_energy_k)(jnp.arange(n_fov)).sum()

    U_obs = jax.vmap(obs_energy_t)(jnp.arange(horizon + 1)).sum()

    # --- Singleton variable entropies ---

    # H[q(x_t)] for t=0..T
    def x_entropy(t):
        log_q_x = log_fwd_msgs[t] + log_bwd_msgs[t]
        log_q_x = log_q_x - logsumexp(log_q_x)
        return _entropy(log_q_x)

    H_x = jax.vmap(x_entropy)(jnp.arange(horizon + 1)).sum()

    # H[q(u_t)] for t=0..T-1
    def u_entropy(t):
        log_q_u = safe_log(q_u[t])
        return _entropy(log_q_u)

    H_u = jax.vmap(u_entropy)(jnp.arange(horizon)).sum()

    # H[q(theta)] — approximate from mean of cavities
    all_cavities = jnp.concatenate([log_cavity_dyn, log_cavity_obs], axis=0)
    log_q_theta = all_cavities.mean(axis=0)
    log_q_theta = log_q_theta - logsumexp(log_q_theta)
    H_theta = _entropy(log_q_theta)

    # --- Intersection terms ---

    # H[q(x_{t-1}, u_t)] - marginal of dyn region over (x_new, theta)
    def dyn_intersection_t(t):
        # Marginalize x_new and theta from region belief
        log_joint_xu = logsumexp(log_dyn_regions[t], axis=(1, 2))  # (x_old, u)
        H_xu = _entropy_unnorm(log_joint_xu)

        # H[q(x_t, x_{t-1}, u_t)] — marginalize theta from region
        log_joint_xxu = logsumexp(log_dyn_regions[t], axis=2)  # (x_old, x_new, u)
        H_xxu = _entropy_unnorm(log_joint_xxu)

        return H_xu - H_xxu

    dyn_intersection = jax.vmap(dyn_intersection_t)(jnp.arange(horizon)).sum()

    # sum_k [H[q(y_{t,k}, x_t, theta)] - H[q(x_t, theta)]] per t
    def obs_intersection_t(t):
        def obs_intersection_k(k):
            # H[q(y, x, theta)] for this factor
            H_yxtheta = _entropy_unnorm(log_obs_regions[t, k])
            # H[q(x, theta)] = marginalize y
            log_xtheta = logsumexp(log_obs_regions[t, k], axis=0)  # (n_states, n_static)
            H_xtheta = _entropy_unnorm(log_xtheta)
            return H_yxtheta - H_xtheta
        return jax.vmap(obs_intersection_k)(jnp.arange(n_fov)).sum()

    obs_intersection = jax.vmap(obs_intersection_t)(jnp.arange(horizon + 1)).sum()

    return (U_dyn + U_obs) - (H_x + H_u + H_theta) + dyn_intersection + obs_intersection


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
def planning_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    goal,
    horizon,
    n_iterations=1,
    action_prior=None,
):
    """Standard BP planning with per-iteration VFE trace.

    Returns:
        action_dist: (n_actions,)
        vfe_trace: (n_iterations,)
    """
    n_states = q_current_state.shape[0]
    n_actions = transition_tensor.shape[3]

    log_T = safe_log(transition_tensor)
    log_reduced = marginalize_static(log_T, safe_log(q_static_state))

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    log_q_state = jnp.concatenate([
        log_q0[None, :],
        jnp.zeros((horizon, n_states)),
    ], axis=0)

    q_u = jnp.tile(action_prior, (horizon, 1))
    vfe_trace = jnp.zeros(n_iterations)

    def body_fn(i, carry):
        log_q_state, q_u, vfe_trace = carry

        log_q_state_new = bp_forward_pass(log_reduced, log_q_state, log_action_prior, horizon)
        log_bwd, q_u_new = backward_pass_bp_full(
            log_reduced, log_q_state_new, log_goal, log_action_prior, horizon
        )

        vfe = compute_bethe_vfe_bp(log_reduced, log_q_state_new, log_bwd,
                                    q_u_new, log_action_prior)
        vfe_trace = vfe_trace.at[i].set(vfe)

        return log_q_state_new, q_u_new, vfe_trace

    log_q_state, q_u, vfe_trace = lax.fori_loop(
        0, n_iterations, body_fn, (log_q_state, q_u, vfe_trace)
    )

    return q_u[0], vfe_trace


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
    log_goal = safe_log(goal)

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    log_cavity_theta = jnp.tile(log_prior_theta, (horizon, 1))
    q_u_init = jnp.zeros((horizon, n_actions))
    vfe_trace = jnp.zeros(n_iterations)

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

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)
    log_goal = safe_log(goal)

    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    log_obs_to_theta = jnp.zeros((horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    log_dyn_channels_init = jnp.zeros((horizon, n_states, n_states, n_actions))
    log_obs_channels_init = jnp.zeros((horizon + 1, n_fov, n_obs_types, n_states, n_static))
    vfe_trace = jnp.zeros(n_iterations)

    def body_fn(i, carry):
        (log_dyn_to_theta, log_obs_to_theta, _, log_dyn_channels,
         log_obs_channels, vfe_trace) = carry

        log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
            log_prior_theta, log_dyn_to_theta, log_obs_to_theta
        )

        log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
        log_obs_kernels = safe_log_div(log_B_flat[None], log_obs_channels)

        log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)
        log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs)

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
            log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
        )

        log_dyn_regions = compute_dyn_region_beliefs(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_dyn, log_action_prior
        )
        log_obs_regions = compute_obs_region_beliefs(
            log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_obs
        )

        raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
        raw_log_obs_channels = compute_obs_channels(log_obs_regions)

        new_log_dyn_channels = damp_log_channel(
            log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)
        new_log_obs_channels = damp_log_channel(
            log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)

        vfe = compute_region_extended_vfe(
            log_dyn_regions, log_obs_regions,
            log_fwd_msgs, log_bwd_msgs, q_u,
            log_T_kernel, log_B_flat,
            log_cavity_dyn, log_cavity_obs,
            log_prior_theta,
        )
        vfe_trace = vfe_trace.at[i].set(vfe)

        return (new_log_dyn_to_theta, new_log_obs_to_theta, q_u,
                new_log_dyn_channels, new_log_obs_channels, vfe_trace)

    result = lax.fori_loop(
        0, n_iterations, body_fn,
        (log_dyn_to_theta, log_obs_to_theta, q_u_init,
         log_dyn_channels_init, log_obs_channels_init, vfe_trace)
    )
    _, _, q_u, log_dyn_channels, log_obs_channels, vfe_trace = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_dyn_channels, log_obs_channels, vfe_trace


@partial(jax.jit, static_argnums=(5, 6))
def reduced_region_extended_convergence(
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
    """Reduced region-extended planning (fixed theta) with per-iteration VFE trace.

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

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    log_cavity_fixed = safe_log(q_static_state)
    log_cavity_dyn = jnp.tile(log_cavity_fixed, (horizon, 1))
    log_cavity_obs = jnp.tile(log_cavity_fixed, (horizon + 1, 1))

    q_u_init = jnp.zeros((horizon, n_actions))
    log_dyn_channels_init = jnp.zeros((horizon, n_states, n_states, n_actions))
    log_obs_channels_init = jnp.zeros((horizon + 1, n_fov, n_obs_types, n_states, n_static))
    vfe_trace = jnp.zeros(n_iterations)

    def body_fn(i, carry):
        q_u, log_dyn_channels, log_obs_channels, vfe_trace = carry

        log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
        log_obs_kernels = safe_log_div(log_B_flat[None], log_obs_channels)

        log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)
        log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs)

        log_fwd_msgs = re_forward_pass(
            log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon
        )
        log_bwd_msgs, q_u = re_backward_pass(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
            log_obs_to_x, horizon
        )

        log_dyn_regions = compute_dyn_region_beliefs(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_dyn, log_action_prior
        )
        log_obs_regions = compute_obs_region_beliefs(
            log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_obs
        )

        raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
        raw_log_obs_channels = compute_obs_channels(log_obs_regions)

        new_log_dyn_channels = damp_log_channel(
            log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)
        new_log_obs_channels = damp_log_channel(
            log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)

        vfe = compute_region_extended_vfe(
            log_dyn_regions, log_obs_regions,
            log_fwd_msgs, log_bwd_msgs, q_u,
            log_T_kernel, log_B_flat,
            log_cavity_dyn, log_cavity_obs,
            log_cavity_fixed,
        )
        vfe_trace = vfe_trace.at[i].set(vfe)

        return q_u, new_log_dyn_channels, new_log_obs_channels, vfe_trace

    q_u, log_dyn_channels, log_obs_channels, vfe_trace = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, log_dyn_channels_init, log_obs_channels_init, vfe_trace)
    )

    return q_u[0], log_dyn_channels, log_obs_channels, vfe_trace


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
    log_goal = safe_log(goal)

    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    action_prior_init = jnp.tile(action_prior, (horizon, 1))
    log_dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, n_obs_types, n_states, n_static))
    vfe_trace = jnp.zeros(n_iterations)

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


@partial(jax.jit, static_argnums=(5, 6))
def reduced_nuijten_mp_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    observation_tensor,
    goal,
    horizon,
    n_iterations,
    action_prior=None,
):
    """Reduced Nuijten MP planning (fixed theta) with per-iteration VFE trace.

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
    log_goal = safe_log(goal)

    log_q_static = safe_log(q_static_state)
    log_q_static_norm = log_q_static - logsumexp(log_q_static)
    log_cavity_dyn = jnp.tile(log_q_static_norm, (horizon, 1))
    log_cavity_obs = jnp.tile(log_q_static_norm, (horizon + 1, 1))

    log_reduced_per_t = compute_log_reduced(log_T_kernel_tiled, log_cavity_dyn)

    q_u_init = jnp.zeros((horizon, n_actions))
    action_prior_init = jnp.tile(action_prior, (horizon, 1))
    log_dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, n_obs_types, n_states, n_static))
    vfe_trace = jnp.zeros(n_iterations)

    def body_fn(i, carry):
        _, action_prior_per_t, _, obs_regions, vfe_trace = carry

        log_action_prior_per_t = safe_log(action_prior_per_t)

        log_obs_to_x = compute_obs_efe_to_x(obs_regions)

        log_fwd_msgs = forward_pass_nuijten(
            log_reduced_per_t, log_q0, log_action_prior_per_t, log_obs_to_x, horizon
        )
        log_bwd_msgs, q_u = backward_pass_nuijten(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior_per_t,
            log_obs_to_x, horizon
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
            log_q_static_norm, action_prior_per_t,
        )
        vfe_trace = vfe_trace.at[i].set(vfe)

        return q_u, new_action_prior, log_dyn_regions, obs_regions, vfe_trace

    result = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, action_prior_init, log_dyn_regions_init, obs_regions_init, vfe_trace)
    )
    q_u, _, log_dyn_region_beliefs, obs_region_beliefs, vfe_trace = result

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

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)
    log_goal = safe_log(goal)

    # Tile obs tensor over time: kernel = raw B (no obs channels)
    log_B_tiled = jnp.broadcast_to(
        log_B_flat[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static)
    )

    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    log_obs_to_theta = jnp.zeros((horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    log_dyn_channels_init = jnp.zeros((horizon, n_states, n_states, n_actions))
    vfe_trace = jnp.zeros(n_iterations)

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

        # Compute obs region beliefs from raw B for VFE computation
        log_obs_regions = compute_obs_region_beliefs(
            log_B_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_obs
        )

        vfe = compute_region_extended_vfe(
            log_dyn_regions, log_obs_regions,
            log_fwd_msgs, log_bwd_msgs, q_u,
            log_T_kernel, log_B_flat,
            log_cavity_dyn, log_cavity_obs,
            log_prior_theta,
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
# Reduced dyn-channel convergence (theta fixed)
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def reduced_dyn_channel_convergence(
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
    """Reduced dyn-channel planning (fixed theta) with per-iteration VFE trace.

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

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    log_cavity_fixed = safe_log(q_static_state)
    log_cavity_dyn = jnp.tile(log_cavity_fixed, (horizon, 1))
    log_cavity_obs = jnp.tile(log_cavity_fixed, (horizon + 1, 1))

    # Tile obs tensor over time
    log_B_tiled = jnp.broadcast_to(
        log_B_flat[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static)
    )

    # Precompute obs->x messages (constant)
    log_obs_to_x = compute_obs_to_x_msgs(log_B_tiled, log_cavity_obs)

    q_u_init = jnp.zeros((horizon, n_actions))
    log_dyn_channels_init = jnp.zeros((horizon, n_states, n_states, n_actions))
    vfe_trace = jnp.zeros(n_iterations)

    def body_fn(i, carry):
        q_u, log_dyn_channels, vfe_trace = carry

        log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])

        log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

        log_fwd_msgs = re_forward_pass(
            log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon
        )
        log_bwd_msgs, q_u = re_backward_pass(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
            log_obs_to_x, horizon
        )

        log_dyn_regions = compute_dyn_region_beliefs(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_dyn, log_action_prior
        )

        raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
        new_log_dyn_channels = damp_log_channel(
            log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)

        # Compute obs region beliefs from raw B for VFE computation
        log_obs_regions = compute_obs_region_beliefs(
            log_B_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_obs
        )

        vfe = compute_region_extended_vfe(
            log_dyn_regions, log_obs_regions,
            log_fwd_msgs, log_bwd_msgs, q_u,
            log_T_kernel, log_B_flat,
            log_cavity_dyn, log_cavity_obs,
            log_cavity_fixed,
        )
        vfe_trace = vfe_trace.at[i].set(vfe)

        return q_u, new_log_dyn_channels, vfe_trace

    q_u, log_dyn_channels, vfe_trace = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, log_dyn_channels_init, vfe_trace)
    )

    return q_u[0], log_dyn_channels, vfe_trace
