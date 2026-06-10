"""Nuijten message passing: region beliefs with EFE-based action priors (full log-space).

Computes region beliefs using original transition/observation tensors — no
channels, no kernels. Region beliefs produce EFE-based per-timestep action
priors that feed back into message passing, forming a fixed-point iteration.

θ-inferred variant: loopy BP iterations refine θ cavities via dyn→θ messages
  and action priors via EFE from region beliefs.
θ-fixed variant: iterative — action prior evolves via EFE even with fixed θ.

Factor graph (same as region_extended_loopy_bp.py):

    p(theta)---theta (single variable, connected to all factors)
                |
    p(x0)--x0--[dyn0]--x1--[dyn1]--x2-- ... --[dyn_{T-1}]--x_T--goal
            |           |           |                        |
         [obs0]_k    [obs1]_k    [obs2]_k               [obsT]_k   (k=1..n_fov)
            |           |           |                        |
          y0,k        y1,k        y2,k                    yT,k     (uniform)
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .messages import (
    safe_log, EPSILON,
    sparse_reduced, sparse_dyn_to_theta, sparse_efe_action_prior,
)
from .region_extended_loopy_bp import (
    compute_log_reduced,
    compute_theta_cavities_extended,
    compute_pref_to_x_msgs,
    compute_pref_to_theta_msgs,
)


# =============================================================================
# Observation region beliefs using original (deterministic) factors
# =============================================================================


def compute_obs_region_beliefs_original(log_B_flat, log_fwd_msgs, log_bwd_msgs, log_cavity_obs):
    """
    Compute region beliefs for observation factors using original p(y|x,θ).

    Args:
        log_B_flat: (n_fov, N_CELL_TYPES, n_states, n_static) log obs tensor
        log_fwd_msgs: (T+1, n_states) log forward messages
        log_bwd_msgs: (T+1, n_states) log backward messages
        log_cavity_obs: (T+1, n_static) log cavity beliefs for obs factors

    Returns:
        region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static) normalized
    """
    log_x_belief = log_fwd_msgs + log_bwd_msgs  # (T+1, n_states)
    log_x_belief = log_x_belief - logsumexp(log_x_belief, axis=1, keepdims=True)

    # Broadcast to (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    log_belief = (log_B_flat[None]
                  + log_x_belief[:, None, None, :, None]
                  + log_cavity_obs[:, None, None, None, :])

    # Normalize per t over all (n_fov, N_CELL_TYPES, n_states, n_static)
    T_plus_1 = log_belief.shape[0]
    flat = log_belief.reshape(T_plus_1, -1)
    return jax.nn.softmax(flat, axis=1).reshape(log_belief.shape)


# =============================================================================
# EFE-based action prior from region beliefs
# =============================================================================


def compute_efe_action_prior(log_dyn_region_beliefs, action_mask):
    """
    Compute EFE-based per-timestep action prior from dynamics region beliefs.

    π_t(u) = softmax( H(x_new | x_old, θ, u) )

    Vectorized — no nested vmaps.

    Args:
        log_dyn_region_beliefs: (T, x_old, x_new, θ, u) unnormalized log beliefs
        action_mask: (n_actions,) binary — 1 for valid actions

    Returns:
        action_prior_per_t: (T, n_actions) probability-space action priors
    """
    # Move u to front for per-(t,u) softmax: (T, u, x_old, x_new, θ)
    log_b = log_dyn_region_beliefs.transpose(0, 4, 1, 2, 3)
    shape = log_b.shape
    log_b_flat = log_b.reshape(shape[0] * shape[1], -1)
    q = jax.nn.softmax(log_b_flat, axis=1).reshape(shape)  # (T, u, x_old, x_new, θ)

    # q(x_old, θ | u, t) = sum_{x_new} q
    q_marg = q.sum(axis=3, keepdims=True)  # (T, u, x_old, 1, θ)

    # H(x_new | x_old, θ, u, t) = -sum q log(q / q_marg)
    q_cond = q / (q_marg + EPSILON)
    efe = -(q * jnp.log(q_cond + EPSILON)).sum(axis=(2, 3, 4))  # (T, u)

    efe = jnp.where(action_mask[None] > 0, efe, -jnp.inf)
    return jax.nn.softmax(efe, axis=1)


def compute_obs_efe_to_x(obs_region_beliefs):
    """
    Message to x_t from observation factors via EFE.

    obs_to_x[t](x) = softmax(-sum_k H_k(y | theta, x))

    Vectorized — no nested vmaps.

    Args:
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static) normalized

    Returns:
        log_obs_to_x: (T+1, n_states) log-normalized messages
    """
    # Normalize per (t, k, x) over (y=axis2, θ=axis4)
    Z = obs_region_beliefs.sum(axis=(2, 4), keepdims=True) + EPSILON
    q = obs_region_beliefs / Z

    # q(θ | k, x, t) = sum_y q(y, θ | k, x, t)
    q_theta = q.sum(axis=2, keepdims=True)  # (T+1, n_fov, 1, n_states, n_static)

    # H(y | θ, k, x, t) = -sum_{y,θ} q log(q / q_theta)
    q_cond = q / (q_theta + EPSILON)
    H_cond = -(q * jnp.log(q_cond + EPSILON)).sum(axis=(2, 4))  # (T+1, n_fov, n_states)

    # Sum over FOV positions
    total_H = H_cond.sum(axis=1)  # (T+1, n_states)
    log_msg = -total_H
    return log_msg - logsumexp(log_msg, axis=1, keepdims=True)


def compute_obs_efe_to_theta(obs_region_beliefs):
    """
    Message to theta from observation factors via EFE (log-space).

    Vectorized — no nested vmaps.

    Args:
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static) normalized

    Returns:
        log_obs_to_theta: (T+1, n_static) log-space messages
    """
    # Normalize per (t, k, θ) over (y=axis2, x=axis3)
    Z = obs_region_beliefs.sum(axis=(2, 3), keepdims=True) + EPSILON
    q = obs_region_beliefs / Z

    # q(x | k, θ, t) = sum_y q(y, x | k, θ, t)
    q_x = q.sum(axis=2, keepdims=True)  # (T+1, n_fov, 1, n_states, n_static)

    # H(y | x, k, θ, t) = -sum_{y,x} q log(q / q_x)
    q_cond = q / (q_x + EPSILON)
    H_cond = -(q * jnp.log(q_cond + EPSILON)).sum(axis=(2, 3))  # (T+1, n_fov, n_static)

    # Sum over FOV positions
    total_H = H_cond.sum(axis=1)  # (T+1, n_static)
    return jnp.log(jax.nn.softmax(-total_H, axis=1) + EPSILON)


# =============================================================================
# Local forward/backward functions with per-timestep action prior (log-space)
# =============================================================================


def forward_pass_nuijten(log_reduced_per_t, log_q_x0, log_action_prior_per_t,
                          log_obs_to_x, horizon):
    """
    Forward pass with per-timestep action prior (log-space).

    Args:
        log_reduced_per_t: (T, n_states, n_states, n_actions)
        log_q_x0: (n_states,) log initial state belief
        log_action_prior_per_t: (T, n_actions) log per-timestep action prior
        log_obs_to_x: (T+1, n_states) log obs messages to x
        horizon: T

    Returns:
        log_fwd_msgs: (T+1, n_states) log forward messages
    """
    n_states = log_q_x0.shape[0]
    log_fwd = jnp.zeros((horizon + 1, n_states))
    log_fwd = log_fwd.at[0].set(log_q_x0)

    def body_fn(t, log_fwd):
        log_x_msg = log_fwd[t] + log_obs_to_x[t]
        terms = (log_reduced_per_t[t]
                 + log_x_msg[None, :, None]
                 + log_action_prior_per_t[t][None, None, :])
        log_q_next = logsumexp(terms, axis=(1, 2))
        log_q_next = log_q_next - logsumexp(log_q_next)
        return log_fwd.at[t + 1].set(log_q_next)

    return lax.fori_loop(0, horizon, body_fn, log_fwd)


def backward_pass_nuijten(log_reduced_per_t, log_fwd_msgs, log_goal,
                           log_action_prior_per_t, log_obs_to_x, horizon):
    """
    Backward pass with per-timestep action prior (log-space).

    Args:
        log_reduced_per_t: (T, n_states, n_states, n_actions)
        log_fwd_msgs: (T+1, n_states) log forward messages
        log_goal: (n_states,) log goal distribution
        log_action_prior_per_t: (T, n_actions) log per-timestep action prior
        log_obs_to_x: (T+1, n_states) log obs messages to x
        horizon: T

    Returns:
        log_bwd_msgs: (T+1, n_states) log backward messages
        q_u: (T, n_actions) action marginals (probability space)
    """
    n_states = log_goal.shape[0]
    n_actions = log_action_prior_per_t.shape[1]
    log_bwd = jnp.zeros((horizon + 1, n_states))
    log_bwd = log_bwd.at[horizon].set(log_goal)
    q_u = jnp.zeros((horizon, n_actions))

    def body_fn(carry, t):
        log_bwd, q_u = carry
        log_bwd_t1 = log_bwd[t + 1] + log_obs_to_x[t + 1]

        log_fwd_t = log_fwd_msgs[t] + log_obs_to_x[t]
        terms = (log_reduced_per_t[t]
                 + log_bwd_t1[:, None, None]
                 + log_fwd_t[None, :, None])
        log_msg_to_u = logsumexp(terms, axis=(0, 1))
        q_u_t = jax.nn.softmax(log_msg_to_u + log_action_prior_per_t[t])
        q_u = q_u.at[t].set(q_u_t)

        terms = (log_reduced_per_t[t]
                 + log_bwd_t1[:, None, None]
                 + log_action_prior_per_t[t][None, None, :])
        log_bwd_t = logsumexp(terms, axis=(0, 2))
        log_bwd_t = log_bwd_t - logsumexp(log_bwd_t)
        log_bwd = log_bwd.at[t].set(log_bwd_t)

        return (log_bwd, q_u), None

    (log_bwd, q_u), _ = lax.scan(
        body_fn, (log_bwd, q_u), jnp.arange(horizon - 1, -1, -1)
    )

    return log_bwd, q_u


def compute_dyn_to_theta_msgs_nuijten(log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs,
                                       log_obs_to_x, log_action_prior_per_t, horizon):
    """
    Compute dyn→θ messages with per-timestep action prior (log-space).

    Args:
        log_T_kernel_tiled: (T, x_old, x_new, θ, u) log original factors
        log_fwd_msgs: (T+1, n_states) log forward messages
        log_bwd_msgs: (T+1, n_states) log backward messages
        log_obs_to_x: (T+1, n_states) log obs messages to x
        log_action_prior_per_t: (T, n_actions) log per-timestep action prior
        horizon: T

    Returns:
        log_dyn_to_theta: (T, n_static) log-space messages
    """
    log_fwd_t = log_fwd_msgs[:-1] + log_obs_to_x[:-1]    # (T, n_states)
    log_bwd_t1 = log_bwd_msgs[1:] + log_obs_to_x[1:]     # (T, n_states)

    terms = (log_T_kernel_tiled
             + log_fwd_t[:, :, None, None, None]
             + log_bwd_t1[:, None, :, None, None]
             + log_action_prior_per_t[:, None, None, None, :])
    return logsumexp(terms, axis=(1, 2, 4))


def compute_dyn_region_beliefs_nuijten(log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs,
                                        log_obs_to_x, log_cavity_dyn, log_action_prior_per_t):
    """
    Compute dynamics region beliefs with per-timestep action prior (log-space).

    Args:
        log_T_kernel_tiled: (T, x_old, x_new, θ, u) log original factors
        log_fwd_msgs: (T+1, n_states) log forward messages
        log_bwd_msgs: (T+1, n_states) log backward messages
        log_obs_to_x: (T+1, n_states) log obs messages to x
        log_cavity_dyn: (T, n_static) log cavity beliefs for dyn factors
        log_action_prior_per_t: (T, n_actions) log per-timestep action prior

    Returns:
        log_region_beliefs: (T, x_old, x_new, θ, u) unnormalized log beliefs
    """
    log_fwd_t = log_fwd_msgs[:-1] + log_obs_to_x[:-1]    # (T, n_states)
    log_bwd_t1 = log_bwd_msgs[1:] + log_obs_to_x[1:]     # (T, n_states)

    return (log_T_kernel_tiled
            + log_fwd_t[:, :, None, None, None]
            + log_bwd_t1[:, None, :, None, None]
            + log_cavity_dyn[:, None, None, :, None]
            + log_action_prior_per_t[:, None, None, None, :])


# =============================================================================
# θ-inferred variant
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def nuijten_mp_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta
    transition_tensor,    # (n_states, n_states, n_static, n_actions) probability
    observation_tensor,   # (n_channels, n_obs_types, n_states, n_static) probability
    goal,                 # (n_states,) or (n_states, n_static) preference
    horizon,              # int (static)
    n_iterations,         # int (static)
    action_prior=None,    # (n_actions,) prior over actions. If None, uniform.
    T_idx=None,           # (S, A, θ) int32 sparse transition index
):
    """
    Plan actions via Nuijten MP with θ as a variable node.

    Accepts full probability tensors, logs once at the top.

    When goal is 2D (n_states, n_static), it is treated as a per-config
    preference factor C(x, θ) applied at every timestep.

    Returns:
        action_dist: (n_actions,) distribution over first action
        log_dyn_region_beliefs: (T, x_old, x_new, θ, u)
        obs_region_beliefs: (T+1, n_channels, n_obs_types, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    use_sparse = T_idx is not None
    n_actions = T_idx.shape[1] if use_sparse else transition_tensor.shape[3]
    n_fov = observation_tensor.shape[0]
    n_obs_types = observation_tensor.shape[1]
    has_pref = goal.ndim == 2

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    action_mask = (action_prior > 0).astype(jnp.float32)

    # Log once at top
    if not use_sparse:
        log_T = safe_log(transition_tensor)
        log_T_kernel = log_T.transpose(1, 0, 2, 3)  # (x_old, x_new, θ, u)
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

    # Initialize carry
    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    action_prior_init = jnp.tile(action_prior, (horizon, 1))
    if not use_sparse:
        log_dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, n_obs_types, n_states, n_static))

    if has_pref and use_sparse:
        log_pref_to_theta_init = jnp.zeros((horizon + 1, n_static))

        def body_fn(_, carry):
            log_dyn_to_theta, _, action_prior_per_t, obs_regions, log_pref_to_theta = carry

            log_action_prior_per_t = safe_log(action_prior_per_t)

            log_obs_to_x = compute_obs_efe_to_x(obs_regions)
            log_obs_to_theta = compute_obs_efe_to_theta(obs_regions)

            log_cavity_dyn, log_cavity_obs, log_cavity_pref = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta
            )

            log_reduced_per_t = sparse_reduced(T_idx, log_cavity_dyn, n_states)

            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_cavity_pref)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            log_fwd_msgs = forward_pass_nuijten(
                log_reduced_per_t, log_q0, log_action_prior_per_t, log_local_to_x, horizon
            )
            log_bwd_msgs, q_u = backward_pass_nuijten(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior_per_t,
                log_local_to_x, horizon
            )

            new_log_dyn_to_theta = sparse_dyn_to_theta(
                T_idx, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_action_prior_per_t, n_states)

            new_log_pref_to_theta = compute_pref_to_theta_msgs(
                log_C, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            obs_regions = compute_obs_region_beliefs_original(
                log_B_flat, log_fwd_msgs, log_bwd_msgs, log_cavity_obs
            )

            new_action_prior = sparse_efe_action_prior(
                T_idx, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_cavity_dyn, log_action_prior_per_t, n_states, action_mask)

            return (new_log_dyn_to_theta, q_u, new_action_prior,
                    obs_regions, new_log_pref_to_theta)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, q_u_init, action_prior_init,
             obs_regions_init, log_pref_to_theta_init)
        )
        _, q_u, _, obs_region_beliefs, _ = result
        log_dyn_region_beliefs = None

    elif has_pref:
        log_pref_to_theta_init = jnp.zeros((horizon + 1, n_static))

        def body_fn(_, carry):
            log_dyn_to_theta, _, action_prior_per_t, _, obs_regions, log_pref_to_theta = carry

            log_action_prior_per_t = safe_log(action_prior_per_t)

            log_obs_to_x = compute_obs_efe_to_x(obs_regions)
            log_obs_to_theta = compute_obs_efe_to_theta(obs_regions)

            log_cavity_dyn, log_cavity_obs, log_cavity_pref = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta
            )

            log_reduced_per_t = compute_log_reduced(log_T_kernel_tiled, log_cavity_dyn)

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

            return (new_log_dyn_to_theta, q_u, new_action_prior, log_dyn_regions,
                    obs_regions, new_log_pref_to_theta)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, q_u_init, action_prior_init, log_dyn_regions_init,
             obs_regions_init, log_pref_to_theta_init)
        )
        _, q_u, _, log_dyn_region_beliefs, obs_region_beliefs, _ = result

    elif use_sparse:
        def body_fn(_, carry):
            log_dyn_to_theta, _, action_prior_per_t, obs_regions = carry

            log_action_prior_per_t = safe_log(action_prior_per_t)

            log_obs_to_x = compute_obs_efe_to_x(obs_regions)
            log_obs_to_theta = compute_obs_efe_to_theta(obs_regions)

            log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta
            )

            log_reduced_per_t = sparse_reduced(T_idx, log_cavity_dyn, n_states)

            log_fwd_msgs = forward_pass_nuijten(
                log_reduced_per_t, log_q0, log_action_prior_per_t, log_obs_to_x, horizon
            )
            log_bwd_msgs, q_u = backward_pass_nuijten(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior_per_t,
                log_obs_to_x, horizon
            )

            new_log_dyn_to_theta = sparse_dyn_to_theta(
                T_idx, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_action_prior_per_t, n_states)

            obs_regions = compute_obs_region_beliefs_original(
                log_B_flat, log_fwd_msgs, log_bwd_msgs, log_cavity_obs
            )

            new_action_prior = sparse_efe_action_prior(
                T_idx, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_cavity_dyn, log_action_prior_per_t, n_states, action_mask)

            return new_log_dyn_to_theta, q_u, new_action_prior, obs_regions

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, q_u_init, action_prior_init, obs_regions_init)
        )
        _, q_u, _, obs_region_beliefs = result
        log_dyn_region_beliefs = None

    else:
        def body_fn(_, carry):
            log_dyn_to_theta, _, action_prior_per_t, _, obs_regions = carry

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

            return new_log_dyn_to_theta, q_u, new_action_prior, log_dyn_regions, obs_regions

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, q_u_init, action_prior_init, log_dyn_regions_init, obs_regions_init)
        )
        _, q_u, _, log_dyn_region_beliefs, obs_region_beliefs = result

    return q_u[0], log_dyn_region_beliefs, obs_region_beliefs
