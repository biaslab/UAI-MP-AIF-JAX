"""Nuijten message passing: region beliefs with EFE-based action priors.

Computes region beliefs using original transition/observation tensors — no
channels, no kernels. Region beliefs produce EFE-based per-timestep action
priors that feed back into message passing, forming a fixed-point iteration.

θ-inferred variant: loopy BP iterations refine θ cavities via dyn→θ messages
  and action priors via EFE from region beliefs.
θ-fixed variant: iterative — action prior evolves via EFE even with fixed θ.

With deterministic original factors, H(x_new | x_old, θ, u) = 0 for all u,
so the EFE prior is uniform over valid actions (same as fixed prior).
The infrastructure becomes meaningful when kernels replace original factors.

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
from functools import partial

from .messages import EPSILON
from .region_extended_loopy_bp import (
    compute_reduced_per_t_from_kernels,
    compute_theta_cavities_extended,
)
from environments.minigrid import N_CELL_TYPES


# =============================================================================
# Observation region beliefs using original (deterministic) factors
# =============================================================================


def compute_obs_region_beliefs_original(obs_idx, fwd_msgs, bwd_msgs, cavity_obs):
    """
    Compute region beliefs for observation factors using original deterministic
    p(y|x,θ) = δ(y = obs_idx[k,x,θ]).

    Args:
        obs_idx: (fov_w, fov_h, n_states, n_static) -> cell_type index
        fwd_msgs: (T+1, n_states) forward messages
        bwd_msgs: (T+1, n_states) backward messages
        cavity_obs: (T+1, n_static) cavity beliefs for obs factors

    Returns:
        region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    """
    fov_w, fov_h = obs_idx.shape[0], obs_idx.shape[1]
    n_fov = fov_w * fov_h
    n_states = obs_idx.shape[2]
    n_static = obs_idx.shape[3]
    obs_flat = obs_idx.reshape(n_fov, n_states, n_static)

    # Deterministic p(y|x,θ) as one-hot over cell types
    # (n_fov, n_states, n_static, N_CELL_TYPES) → (n_fov, N_CELL_TYPES, n_states, n_static)
    p_y = jax.nn.one_hot(obs_flat, N_CELL_TYPES)
    p_y = jnp.transpose(p_y, (0, 3, 1, 2))

    def compute_single_t(t):
        x_belief = fwd_msgs[t] * bwd_msgs[t]
        x_belief = x_belief / (x_belief.sum() + EPSILON)

        # q(y, x, θ) ∝ p(y|x,θ) · q(x) · cavity(θ)
        belief = (p_y
                  * x_belief[None, None, :, None]
                  * cavity_obs[t][None, None, None, :])

        Z = belief.sum() + EPSILON
        return belief / Z

    T_plus_1 = fwd_msgs.shape[0]
    return jax.vmap(compute_single_t)(jnp.arange(T_plus_1))


# =============================================================================
# EFE-based action prior from region beliefs
# =============================================================================


def compute_efe_action_prior(dyn_region_beliefs, action_mask):
    """
    Compute EFE-based per-timestep action prior from dynamics region beliefs.

    π_t(u) = softmax( H(q(x_new, x_old, θ | u)) − H(q(x_old, θ | u)) )

    This is the conditional entropy H(x_new | x_old, θ, u) — measures
    transition ambiguity per action. Lower ambiguity → higher prior.

    Args:
        dyn_region_beliefs: (T, n_states, n_states, n_static, n_actions)
        action_mask: (n_actions,) binary — 1 for valid actions

    Returns:
        action_prior_per_t: (T, n_actions)
    """
    def compute_single_t(region_t):
        # region_t: (n_states, n_states, n_static, n_actions)
        def compute_single_u(u):
            # q(x_old, x_new, θ | u) = region[:,:,:,u] / Z
            joint = region_t[:, :, :, u]
            Z = joint.sum() + EPSILON
            q_joint = joint / Z

            # H_joint = -Σ q · log q
            log_q = jnp.log(q_joint + EPSILON)
            H_joint = -(q_joint * log_q).sum()

            # q(x_old, θ | u) = Σ_{x_new} q(x_old, x_new, θ | u)
            q_marg = q_joint.sum(axis=1)  # (n_states, n_static)
            log_q_marg = jnp.log(q_marg + EPSILON)
            H_marginal = -(q_marg * log_q_marg).sum()

            # EFE = H_joint - H_marginal = H(x_new | x_old, θ, u)
            return H_joint - H_marginal

        n_actions = region_t.shape[3]
        efe = jax.vmap(compute_single_u)(jnp.arange(n_actions))

        # Mask invalid actions to -inf, then softmax
        efe = jnp.where(action_mask > 0, efe, -jnp.inf)
        return jax.nn.softmax(efe)

    return jax.vmap(compute_single_t)(dyn_region_beliefs)


def compute_obs_efe_to_x(obs_region_beliefs):
    """
    Message to x_t from observation factors via EFE.

    obs_to_x[t](x) = softmax(-sum_k H_k(y | theta, x))

    Per FOV position k, computes H(y | theta, x) as the conditional entropy
    of y given theta and x, then sums over k and applies softmax(-total).

    Args:
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)

    Returns:
        obs_to_x: (T+1, n_states)
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

        return H_joint - H_marg  # H(y | theta, x)

    # vmap over x: (n_states, N_CELL_TYPES, n_static) -> (n_states,)
    compute_per_x = jax.vmap(compute_H_cond)

    def compute_per_k(belief_k):
        # belief_k: (N_CELL_TYPES, n_states, n_static)
        return compute_per_x(jnp.transpose(belief_k, (1, 0, 2)))  # (n_states,)

    def compute_per_t(beliefs_t):
        # beliefs_t: (n_fov, N_CELL_TYPES, n_states, n_static)
        H_per_k = jax.vmap(compute_per_k)(beliefs_t)  # (n_fov, n_states)
        total = H_per_k.sum(axis=0)  # (n_states,)
        return jax.nn.softmax(-total)

    return jax.vmap(compute_per_t)(obs_region_beliefs)  # (T+1, n_states)


def compute_obs_efe_to_theta(obs_region_beliefs):
    """
    Message to theta from observation factors via EFE (log-space).

    log_obs_to_theta[t](theta) = log softmax(-sum_k H_k(y | x, theta))

    Per FOV position k, computes H(y | x, theta) as the conditional entropy
    of y given x and theta, then sums over k and returns log(softmax(-total)).

    Args:
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)

    Returns:
        log_obs_to_theta: (T+1, n_static)
    """
    def compute_H_cond(belief_slice):
        # belief_slice: (N_CELL_TYPES, n_states) — q(y, x) for fixed theta, k
        Z = belief_slice.sum() + EPSILON
        q = belief_slice / Z
        log_q = jnp.log(q + EPSILON)
        H_joint = -(q * log_q).sum()

        q_marg_x = q.sum(axis=0)  # (n_states,)
        log_q_marg = jnp.log(q_marg_x + EPSILON)
        H_marg = -(q_marg_x * log_q_marg).sum()

        return H_joint - H_marg  # H(y | x, theta)

    # vmap over theta: (n_static, N_CELL_TYPES, n_states) -> (n_static,)
    compute_per_theta = jax.vmap(compute_H_cond)

    def compute_per_k(belief_k):
        # belief_k: (N_CELL_TYPES, n_states, n_static)
        return compute_per_theta(jnp.transpose(belief_k, (2, 0, 1)))  # (n_static,)

    def compute_per_t(beliefs_t):
        # beliefs_t: (n_fov, N_CELL_TYPES, n_states, n_static)
        H_per_k = jax.vmap(compute_per_k)(beliefs_t)  # (n_fov, n_static)
        total = H_per_k.sum(axis=0)  # (n_static,)
        return jnp.log(jax.nn.softmax(-total) + EPSILON)

    return jax.vmap(compute_per_t)(obs_region_beliefs)  # (T+1, n_static)


# =============================================================================
# Local forward/backward functions with per-timestep action prior
# =============================================================================


def forward_pass_nuijten(reduced_per_t, q_x0, action_prior_per_t, obs_to_x, horizon):
    """
    Forward pass with per-timestep action prior.

    Args:
        reduced_per_t: (T, n_states, n_states, n_actions)
        q_x0: (n_states,) initial state belief
        action_prior_per_t: (T, n_actions) per-timestep action prior
        obs_to_x: (T+1, n_states) messages from obs factors to x variables
        horizon: T

    Returns:
        fwd_msgs: (T+1, n_states) forward messages
    """
    n_states = q_x0.shape[0]
    fwd_msgs = jnp.zeros((horizon + 1, n_states))
    fwd_msgs = fwd_msgs.at[0].set(q_x0)

    def body_fn(t, fwd_msgs):
        x_msg = fwd_msgs[t] * obs_to_x[t]
        q_next = jnp.einsum("ijk,j,k->i", reduced_per_t[t], x_msg, action_prior_per_t[t])
        q_next = q_next / (q_next.sum() + EPSILON)
        return fwd_msgs.at[t + 1].set(q_next)

    return lax.fori_loop(0, horizon, body_fn, fwd_msgs)


def backward_pass_nuijten(reduced_per_t, fwd_msgs, goal, action_prior_per_t, obs_to_x, horizon):
    """
    Backward pass with per-timestep action prior, compute action marginals.

    Args:
        reduced_per_t: (T, n_states, n_states, n_actions)
        fwd_msgs: (T+1, n_states) forward messages
        goal: (n_states,) goal distribution
        action_prior_per_t: (T, n_actions) per-timestep action prior
        obs_to_x: (T+1, n_states) messages from obs factors to x variables
        horizon: T

    Returns:
        bwd_msgs: (T+1, n_states) backward messages
        q_u: (T, n_actions) action marginals
    """
    n_states = goal.shape[0]
    n_actions = action_prior_per_t.shape[1]
    bwd_msgs = jnp.zeros((horizon + 1, n_states))
    bwd_msgs = bwd_msgs.at[horizon].set(goal)
    q_u = jnp.zeros((horizon, n_actions))

    def body_fn(carry, t):
        bwd_msgs, q_u = carry
        bwd_t1 = bwd_msgs[t + 1] * obs_to_x[t + 1]

        fwd_t = fwd_msgs[t] * obs_to_x[t]
        msg_to_u = jnp.einsum("ijk,i,j->k", reduced_per_t[t], bwd_t1, fwd_t)
        q_u_t = msg_to_u * action_prior_per_t[t]
        q_u_t = q_u_t / (q_u_t.sum() + EPSILON)
        q_u = q_u.at[t].set(q_u_t)

        bwd_t = jnp.einsum("ijk,i,k->j", reduced_per_t[t], bwd_t1, action_prior_per_t[t])
        bwd_t = bwd_t / (bwd_t.sum() + EPSILON)
        bwd_msgs = bwd_msgs.at[t].set(bwd_t)

        return (bwd_msgs, q_u), None

    (bwd_msgs, q_u), _ = lax.scan(
        body_fn, (bwd_msgs, q_u), jnp.arange(horizon - 1, -1, -1)
    )

    return bwd_msgs, q_u


def compute_dyn_to_theta_msgs_nuijten(dyn_kernels, fwd_msgs, bwd_msgs, obs_to_x, action_prior_per_t, horizon):
    """
    Compute dyn→θ messages with per-timestep action prior.

    Args:
        dyn_kernels: (T, n_states, n_states, n_static, n_actions)
        fwd_msgs: (T+1, n_states) forward messages
        bwd_msgs: (T+1, n_states) backward messages
        obs_to_x: (T+1, n_states) obs factor messages to x
        action_prior_per_t: (T, n_actions) per-timestep action prior
        horizon: T

    Returns:
        log_dyn_to_theta: (T, n_static) log-space messages
    """
    def compute_msg_t(t):
        fwd_t = fwd_msgs[t] * obs_to_x[t]
        bwd_t1 = bwd_msgs[t + 1] * obs_to_x[t + 1]

        msg = jnp.einsum("ijkl,i,j,l->k", dyn_kernels[t], fwd_t, bwd_t1, action_prior_per_t[t])

        return jnp.log(msg + EPSILON)

    return jax.vmap(compute_msg_t)(jnp.arange(horizon))


def compute_dyn_region_beliefs_nuijten(dyn_kernels, fwd_msgs, bwd_msgs, obs_to_x,
                                       cavity_dyn, action_prior_per_t):
    """
    Compute dynamics region beliefs with per-timestep action prior.

    Args:
        dyn_kernels: (T, n_states, n_states, n_static, n_actions)
        fwd_msgs: (T+1, n_states) forward messages
        bwd_msgs: (T+1, n_states) backward messages
        obs_to_x: (T+1, n_states) obs messages to x
        cavity_dyn: (T, n_static) cavity beliefs for dyn factors
        action_prior_per_t: (T, n_actions) per-timestep action prior

    Returns:
        region_beliefs: (T, n_states, n_states, n_static, n_actions)
    """
    T = cavity_dyn.shape[0]

    def compute_single_t(t):
        fwd_t = fwd_msgs[t] * obs_to_x[t]
        bwd_t1 = bwd_msgs[t + 1] * obs_to_x[t + 1]

        belief = (dyn_kernels[t]
                  * fwd_t[:, None, None, None]
                  * bwd_t1[None, :, None, None]
                  * cavity_dyn[t][None, None, :, None]
                  * action_prior_per_t[t][None, None, None, :])

        Z = belief.sum() + EPSILON
        return belief / Z

    return jax.vmap(compute_single_t)(jnp.arange(T))


# =============================================================================
# θ-inferred variant
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def nuijten_mp_planning_indexed(
    q_current_state,    # (n_states,)
    q_static_state,     # (n_static,) prior on theta
    transition_idx,     # (n_states, n_static, n_actions)
    obs_idx,            # (fov_w, fov_h, n_states, n_static) -> cell_type
    goal,               # (n_states,)
    horizon,            # int (static)
    n_iterations,       # int (static)
):
    """
    Plan actions via Nuijten MP with θ as a variable node.

    Iterations refine both θ cavities (via dyn→θ messages) and action priors
    (via EFE from region beliefs). Region beliefs → EFE action prior → messages
    → region beliefs forms the iterative loop.

    Returns:
        action_dist: (n_actions,) distribution over first action
        dyn_region_beliefs: (T, n_states, n_states, n_static, n_actions)
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_idx.shape[2]
    fov_w, fov_h = obs_idx.shape[0], obs_idx.shape[1]
    n_fov = fov_w * fov_h
    action_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    log_prior_theta = jnp.log(q_static_state + EPSILON)

    # Original factors (static, never change)
    p_xnew = jax.nn.one_hot(transition_idx, n_states)   # (x_old, θ, u, x_new)
    p_xnew = jnp.transpose(p_xnew, (0, 3, 1, 2))       # (x_old, x_new, θ, u)
    p_xnew_tiled = jnp.broadcast_to(
        p_xnew[None], (horizon, n_states, n_states, n_static, n_actions)
    )

    # Initialize carry
    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    action_prior_init = jnp.tile(
        jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0]), (horizon, 1)
    )
    dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, N_CELL_TYPES, n_states, n_static))

    def body_fn(_, carry):
        log_dyn_to_theta, _, action_prior_per_t, _, obs_regions = carry

        # Obs EFE messages from carried region beliefs
        obs_to_x = compute_obs_efe_to_x(obs_regions)
        log_obs_to_theta = compute_obs_efe_to_theta(obs_regions)

        # Step 1: theta cavities (total-minus-self)
        cavity_dyn, cavity_obs = compute_theta_cavities_extended(
            log_prior_theta, log_dyn_to_theta, log_obs_to_theta
        )

        # Step 2: Per-timestep reduced tensors from original p_xnew
        reduced_per_t = compute_reduced_per_t_from_kernels(
            p_xnew_tiled, cavity_dyn
        )

        # Step 3: Forward pass with per-timestep action prior
        fwd_msgs = forward_pass_nuijten(
            reduced_per_t, q_current_state, action_prior_per_t, obs_to_x, horizon
        )

        # Step 4: Backward pass + action marginals
        bwd_msgs, q_u = backward_pass_nuijten(
            reduced_per_t, fwd_msgs, goal, action_prior_per_t, obs_to_x, horizon
        )

        # Step 5: dyn→θ messages using original factors
        new_log_dyn_to_theta = compute_dyn_to_theta_msgs_nuijten(
            p_xnew_tiled, fwd_msgs, bwd_msgs, obs_to_x, action_prior_per_t, horizon
        )

        # Step 6: Region beliefs using original factors
        dyn_regions = compute_dyn_region_beliefs_nuijten(
            p_xnew_tiled, fwd_msgs, bwd_msgs, obs_to_x,
            cavity_dyn, action_prior_per_t
        )
        obs_regions = compute_obs_region_beliefs_original(
            obs_idx, fwd_msgs, bwd_msgs, cavity_obs
        )

        # Step 7: EFE-based action prior from region beliefs
        new_action_prior = compute_efe_action_prior(dyn_regions, action_mask)

        return new_log_dyn_to_theta, q_u, new_action_prior, dyn_regions, obs_regions

    log_dyn_to_theta, q_u, action_prior_per_t, dyn_region_beliefs, obs_region_beliefs = lax.fori_loop(
        0, n_iterations, body_fn,
        (log_dyn_to_theta, q_u_init, action_prior_init, dyn_regions_init, obs_regions_init)
    )

    return q_u[0], dyn_region_beliefs, obs_region_beliefs


# =============================================================================
# θ-fixed variant
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def reduced_nuijten_mp_planning_indexed(
    q_current_state,    # (n_states,)
    q_static_state,     # (n_static,) prior on theta (used as fixed cavity)
    transition_idx,     # (n_states, n_static, n_actions)
    obs_idx,            # (fov_w, fov_h, n_states, n_static) -> cell_type
    goal,               # (n_states,)
    horizon,            # int (static)
    n_iterations,       # int (static)
):
    """
    Plan actions via reduced Nuijten MP with fixed θ.

    Now iterative: action prior evolves via EFE from region beliefs even
    with fixed θ. Pre-computes reduced tensors and cavities outside loop.

    Returns:
        action_dist: (n_actions,) distribution over first action
        dyn_region_beliefs: (T, n_states, n_states, n_static, n_actions)
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_idx.shape[2]
    fov_w, fov_h = obs_idx.shape[0], obs_idx.shape[1]
    n_fov = fov_w * fov_h
    action_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    # Original factors (static)
    p_xnew = jax.nn.one_hot(transition_idx, n_states)   # (x_old, θ, u, x_new)
    p_xnew = jnp.transpose(p_xnew, (0, 3, 1, 2))       # (x_old, x_new, θ, u)
    p_xnew_tiled = jnp.broadcast_to(
        p_xnew[None], (horizon, n_states, n_states, n_static, n_actions)
    )

    # Fixed cavities (constant across iterations)
    cavity_dyn = jnp.tile(q_static_state, (horizon, 1))       # (T, n_static)
    cavity_obs = jnp.tile(q_static_state, (horizon + 1, 1))   # (T+1, n_static)

    # Compute reduced tensor once (constant across iterations)
    reduced_per_t = compute_reduced_per_t_from_kernels(p_xnew_tiled, cavity_dyn)

    # Initialize carry
    q_u_init = jnp.zeros((horizon, n_actions))
    action_prior_init = jnp.tile(
        jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0]), (horizon, 1)
    )
    dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, N_CELL_TYPES, n_states, n_static))

    def body_fn(_, carry):
        _, action_prior_per_t, _, obs_regions = carry

        # Obs EFE message from carried region beliefs
        obs_to_x = compute_obs_efe_to_x(obs_regions)

        # Forward pass with per-timestep action prior
        fwd_msgs = forward_pass_nuijten(
            reduced_per_t, q_current_state, action_prior_per_t, obs_to_x, horizon
        )

        # Backward pass + action marginals
        bwd_msgs, q_u = backward_pass_nuijten(
            reduced_per_t, fwd_msgs, goal, action_prior_per_t, obs_to_x, horizon
        )

        # Region beliefs using original factors
        dyn_regions = compute_dyn_region_beliefs_nuijten(
            p_xnew_tiled, fwd_msgs, bwd_msgs, obs_to_x,
            cavity_dyn, action_prior_per_t
        )
        obs_regions = compute_obs_region_beliefs_original(
            obs_idx, fwd_msgs, bwd_msgs, cavity_obs
        )

        # EFE-based action prior from region beliefs
        new_action_prior = compute_efe_action_prior(dyn_regions, action_mask)

        return q_u, new_action_prior, dyn_regions, obs_regions

    q_u, action_prior_per_t, dyn_region_beliefs, obs_region_beliefs = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, action_prior_init, dyn_regions_init, obs_regions_init)
    )

    return q_u[0], dyn_region_beliefs, obs_region_beliefs
