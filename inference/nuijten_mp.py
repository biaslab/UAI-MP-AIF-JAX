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

from .planning import safe_log
from .messages import EPSILON
from .region_extended_loopy_bp import (
    compute_log_reduced,
    compute_theta_cavities_extended,
)
from environments.minigrid import N_CELL_TYPES


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
    T_plus_1 = log_fwd_msgs.shape[0]

    def compute_single_t(t):
        log_x_belief = log_fwd_msgs[t] + log_bwd_msgs[t]
        log_x_belief = log_x_belief - logsumexp(log_x_belief)

        # q(y, x, θ) ∝ B(y|x,θ) · q(x) · cavity(θ)
        log_belief = (log_B_flat
                      + log_x_belief[None, None, :, None]
                      + log_cavity_obs[t][None, None, None, :])

        return jax.nn.softmax(log_belief.ravel()).reshape(log_belief.shape)

    return jax.vmap(compute_single_t)(jnp.arange(T_plus_1))


# =============================================================================
# EFE-based action prior from region beliefs
# =============================================================================


def compute_efe_action_prior(log_dyn_region_beliefs, action_mask):
    """
    Compute EFE-based per-timestep action prior from dynamics region beliefs.

    π_t(u) = softmax( H(q(x_new, x_old, θ | u)) − H(q(x_old, θ | u)) )

    Accepts unnormalized log beliefs, locally softmax's for entropy.

    Args:
        log_dyn_region_beliefs: (T, x_old, x_new, θ, u) unnormalized log beliefs
        action_mask: (n_actions,) binary — 1 for valid actions

    Returns:
        action_prior_per_t: (T, n_actions) probability-space action priors
    """
    def compute_single_t(log_region_t):
        # log_region_t: (x_old, x_new, θ, u)
        def compute_single_u(u):
            log_joint = log_region_t[:, :, :, u]
            q_joint = jax.nn.softmax(log_joint.ravel()).reshape(log_joint.shape)
            log_q = jnp.log(q_joint + EPSILON)
            H_joint = -(q_joint * log_q).sum()

            q_marg = q_joint.sum(axis=1)  # (x_old, θ)
            log_q_marg = jnp.log(q_marg + EPSILON)
            H_marginal = -(q_marg * log_q_marg).sum()

            return H_joint - H_marginal

        n_actions = log_region_t.shape[3]
        efe = jax.vmap(compute_single_u)(jnp.arange(n_actions))
        efe = jnp.where(action_mask > 0, efe, -jnp.inf)
        return jax.nn.softmax(efe)

    return jax.vmap(compute_single_t)(log_dyn_region_beliefs)


def compute_obs_efe_to_x(obs_region_beliefs):
    """
    Message to x_t from observation factors via EFE.

    obs_to_x[t](x) = softmax(-sum_k H_k(y | theta, x))

    Args:
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static) normalized

    Returns:
        log_obs_to_x: (T+1, n_states) log-normalized messages
    """
    def compute_H_cond(belief_slice):
        # belief_slice: (N_CELL_TYPES, n_static) — q(y, theta) for fixed x, k
        Z = belief_slice.sum() + EPSILON
        q = belief_slice / Z
        log_q = jnp.log(q + EPSILON)
        H_joint = -(q * log_q).sum()

        q_marg_theta = q.sum(axis=0)  # (n_static,)
        log_q_marg = jnp.log(q_marg_theta + EPSILON)
        H_marg = -(q_marg_theta * log_q_marg).sum()

        return H_joint - H_marg

    compute_per_x = jax.vmap(compute_H_cond)

    def compute_per_k(belief_k):
        return compute_per_x(jnp.transpose(belief_k, (1, 0, 2)))

    def compute_per_t(beliefs_t):
        H_per_k = jax.vmap(compute_per_k)(beliefs_t)  # (n_fov, n_states)
        total = H_per_k.sum(axis=0)
        # Return log-normalized
        log_msg = -total
        return log_msg - logsumexp(log_msg)

    return jax.vmap(compute_per_t)(obs_region_beliefs)


def compute_obs_efe_to_theta(obs_region_beliefs):
    """
    Message to theta from observation factors via EFE (log-space).

    Args:
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static) normalized

    Returns:
        log_obs_to_theta: (T+1, n_static) log-space messages
    """
    def compute_H_cond(belief_slice):
        Z = belief_slice.sum() + EPSILON
        q = belief_slice / Z
        log_q = jnp.log(q + EPSILON)
        H_joint = -(q * log_q).sum()

        q_marg_x = q.sum(axis=0)
        log_q_marg = jnp.log(q_marg_x + EPSILON)
        H_marg = -(q_marg_x * log_q_marg).sum()

        return H_joint - H_marg

    compute_per_theta = jax.vmap(compute_H_cond)

    def compute_per_k(belief_k):
        return compute_per_theta(jnp.transpose(belief_k, (2, 0, 1)))

    def compute_per_t(beliefs_t):
        H_per_k = jax.vmap(compute_per_k)(beliefs_t)
        total = H_per_k.sum(axis=0)
        return jnp.log(jax.nn.softmax(-total) + EPSILON)

    return jax.vmap(compute_per_t)(obs_region_beliefs)


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
    def compute_msg_t(t):
        log_fwd_t = log_fwd_msgs[t] + log_obs_to_x[t]
        log_bwd_t1 = log_bwd_msgs[t + 1] + log_obs_to_x[t + 1]

        terms = (log_T_kernel_tiled[t]
                 + log_fwd_t[:, None, None, None]
                 + log_bwd_t1[None, :, None, None]
                 + log_action_prior_per_t[t][None, None, None, :])
        return logsumexp(terms, axis=(0, 1, 3))

    return jax.vmap(compute_msg_t)(jnp.arange(horizon))


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
    T = log_cavity_dyn.shape[0]

    def compute_single_t(t):
        log_fwd_t = log_fwd_msgs[t] + log_obs_to_x[t]
        log_bwd_t1 = log_bwd_msgs[t + 1] + log_obs_to_x[t + 1]

        return (log_T_kernel_tiled[t]
                + log_fwd_t[:, None, None, None]
                + log_bwd_t1[None, :, None, None]
                + log_cavity_dyn[t][None, None, :, None]
                + log_action_prior_per_t[t][None, None, None, :])

    return jax.vmap(compute_single_t)(jnp.arange(T))


# =============================================================================
# θ-inferred variant
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def nuijten_mp_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta
    transition_tensor,    # (n_states, n_states, n_static, n_actions) probability
    observation_tensor,   # (fov_w, fov_h, N_CELL_TYPES, n_states, n_static) probability
    goal,                 # (n_states,)
    horizon,              # int (static)
    n_iterations,         # int (static)
):
    """
    Plan actions via Nuijten MP with θ as a variable node.

    Accepts full probability tensors, logs once at the top.

    Returns:
        action_dist: (n_actions,) distribution over first action
        log_dyn_region_beliefs: (T, x_old, x_new, θ, u)
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]
    fov_w, fov_h = observation_tensor.shape[0], observation_tensor.shape[1]
    n_fov = fov_w * fov_h
    action_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    # Log once at top
    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)  # (x_old, x_new, θ, u)
    log_T_kernel_tiled = jnp.broadcast_to(
        log_T_kernel[None], (horizon, n_states, n_states, n_static, n_actions)
    )
    log_B_flat = safe_log(observation_tensor.reshape(n_fov, N_CELL_TYPES, n_states, n_static))
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)
    log_goal = safe_log(goal)

    # Initialize carry
    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    action_prior_init = jnp.tile(
        jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0]), (horizon, 1)
    )
    log_dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, N_CELL_TYPES, n_states, n_static))

    def body_fn(_, carry):
        log_dyn_to_theta, _, action_prior_per_t, _, obs_regions = carry

        log_action_prior_per_t = safe_log(action_prior_per_t)

        # Obs EFE messages from carried region beliefs
        log_obs_to_x = compute_obs_efe_to_x(obs_regions)
        log_obs_to_theta = compute_obs_efe_to_theta(obs_regions)

        # Step 1: theta cavities (total-minus-self)
        log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
            log_prior_theta, log_dyn_to_theta, log_obs_to_theta
        )

        # Step 2: Per-timestep reduced tensors from original factors
        log_reduced_per_t = compute_log_reduced(log_T_kernel_tiled, log_cavity_dyn)

        # Step 3: Forward pass with per-timestep action prior
        log_fwd_msgs = forward_pass_nuijten(
            log_reduced_per_t, log_q0, log_action_prior_per_t, log_obs_to_x, horizon
        )

        # Step 4: Backward pass + action marginals
        log_bwd_msgs, q_u = backward_pass_nuijten(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior_per_t,
            log_obs_to_x, horizon
        )

        # Step 5: dyn→θ messages using original factors
        new_log_dyn_to_theta = compute_dyn_to_theta_msgs_nuijten(
            log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_action_prior_per_t, horizon
        )

        # Step 6: Region beliefs using original factors
        log_dyn_regions = compute_dyn_region_beliefs_nuijten(
            log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_dyn, log_action_prior_per_t
        )
        obs_regions = compute_obs_region_beliefs_original(
            log_B_flat, log_fwd_msgs, log_bwd_msgs, log_cavity_obs
        )

        # Step 7: EFE-based action prior from region beliefs
        new_action_prior = compute_efe_action_prior(log_dyn_regions, action_mask)

        return new_log_dyn_to_theta, q_u, new_action_prior, log_dyn_regions, obs_regions

    log_dyn_to_theta, q_u, action_prior_per_t, log_dyn_region_beliefs, obs_region_beliefs = lax.fori_loop(
        0, n_iterations, body_fn,
        (log_dyn_to_theta, q_u_init, action_prior_init, log_dyn_regions_init, obs_regions_init)
    )

    return q_u[0], log_dyn_region_beliefs, obs_region_beliefs


# =============================================================================
# θ-fixed variant
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def reduced_nuijten_mp_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta (used as fixed cavity)
    transition_tensor,    # (n_states, n_states, n_static, n_actions) probability
    observation_tensor,   # (fov_w, fov_h, N_CELL_TYPES, n_states, n_static) probability
    goal,                 # (n_states,)
    horizon,              # int (static)
    n_iterations,         # int (static)
):
    """
    Plan actions via reduced Nuijten MP with fixed θ.

    Accepts full probability tensors, logs once at the top.

    Returns:
        action_dist: (n_actions,) distribution over first action
        log_dyn_region_beliefs: (T, x_old, x_new, θ, u)
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]
    fov_w, fov_h = observation_tensor.shape[0], observation_tensor.shape[1]
    n_fov = fov_w * fov_h
    action_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    # Log once at top
    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_T_kernel_tiled = jnp.broadcast_to(
        log_T_kernel[None], (horizon, n_states, n_states, n_static, n_actions)
    )
    log_B_flat = safe_log(observation_tensor.reshape(n_fov, N_CELL_TYPES, n_states, n_static))
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    # Fixed cavities (log-normalized)
    log_q_static = safe_log(q_static_state)
    log_q_static_norm = log_q_static - logsumexp(log_q_static)
    log_cavity_dyn = jnp.tile(log_q_static_norm, (horizon, 1))
    log_cavity_obs = jnp.tile(log_q_static_norm, (horizon + 1, 1))

    # Compute reduced tensor once (constant across iterations)
    log_reduced_per_t = compute_log_reduced(log_T_kernel_tiled, log_cavity_dyn)

    # Initialize carry
    q_u_init = jnp.zeros((horizon, n_actions))
    action_prior_init = jnp.tile(
        jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0]), (horizon, 1)
    )
    log_dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, N_CELL_TYPES, n_states, n_static))

    def body_fn(_, carry):
        _, action_prior_per_t, _, obs_regions = carry

        log_action_prior_per_t = safe_log(action_prior_per_t)

        # Obs EFE message from carried region beliefs
        log_obs_to_x = compute_obs_efe_to_x(obs_regions)

        # Forward pass with per-timestep action prior
        log_fwd_msgs = forward_pass_nuijten(
            log_reduced_per_t, log_q0, log_action_prior_per_t, log_obs_to_x, horizon
        )

        # Backward pass + action marginals
        log_bwd_msgs, q_u = backward_pass_nuijten(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior_per_t,
            log_obs_to_x, horizon
        )

        # Region beliefs using original factors
        log_dyn_regions = compute_dyn_region_beliefs_nuijten(
            log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_dyn, log_action_prior_per_t
        )
        obs_regions = compute_obs_region_beliefs_original(
            log_B_flat, log_fwd_msgs, log_bwd_msgs, log_cavity_obs
        )

        # EFE-based action prior from region beliefs
        new_action_prior = compute_efe_action_prior(log_dyn_regions, action_mask)

        return q_u, new_action_prior, log_dyn_regions, obs_regions

    q_u, action_prior_per_t, log_dyn_region_beliefs, obs_region_beliefs = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, action_prior_init, log_dyn_regions_init, obs_regions_init)
    )

    return q_u[0], log_dyn_region_beliefs, obs_region_beliefs
