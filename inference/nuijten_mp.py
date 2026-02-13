"""Nuijten message passing: region beliefs without channel reparameterization.

Computes region beliefs using original transition/observation tensors — no
channels, no kernels. Region beliefs are returned as output for later stages.

θ-inferred variant: loopy BP iterations refine θ cavities via dyn→θ messages.
θ-fixed variant: single forward-backward pass with fixed θ.

Mathematically equivalent to standard loopy BP / standard BP respectively,
since obs factors with uniform y and original factors produce trivially
uniform messages (obs→x = 1, obs→θ = 1).

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
    forward_pass,
    backward_pass,
    compute_dyn_to_theta_msgs,
    compute_theta_cavities_extended,
    compute_dyn_region_beliefs,
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

    Like region_extended_loopy_bp but without channel reparameterization:
    original factors are used throughout, obs→x = 1, obs→θ = 1.
    Iterations refine θ cavities via dyn→θ messages only.
    Region beliefs are computed every iteration (carried through the loop).

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
    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])

    log_prior_theta = jnp.log(q_static_state + EPSILON)

    # Original factors (static, never change)
    p_xnew = jax.nn.one_hot(transition_idx, n_states)   # (x_old, θ, u, x_new)
    p_xnew = jnp.transpose(p_xnew, (0, 3, 1, 2))       # (x_old, x_new, θ, u)
    p_xnew_tiled = jnp.broadcast_to(
        p_xnew[None], (horizon, n_states, n_states, n_static, n_actions)
    )

    # Uninformative obs messages (no kernel reparameterization)
    obs_to_x = jnp.ones((horizon + 1, n_states))
    log_obs_to_theta = jnp.zeros((horizon + 1, n_static))

    # Initialize carry
    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, N_CELL_TYPES, n_states, n_static))

    def body_fn(_, carry):
        log_dyn_to_theta, _, _, _ = carry

        # Step 1: theta cavities (total-minus-self, obs_to_theta = zeros)
        cavity_dyn, cavity_obs = compute_theta_cavities_extended(
            log_prior_theta, log_dyn_to_theta, log_obs_to_theta
        )

        # Step 2: Per-timestep reduced tensors from original p_xnew
        reduced_per_t = compute_reduced_per_t_from_kernels(
            p_xnew_tiled, cavity_dyn
        )

        # Step 3: Forward pass (obs_to_x = ones)
        fwd_msgs = forward_pass(
            reduced_per_t, q_current_state, action_prior, obs_to_x, horizon
        )

        # Step 4: Backward pass + action marginals
        bwd_msgs, q_u = backward_pass(
            reduced_per_t, fwd_msgs, goal, action_prior, obs_to_x, horizon
        )

        # Step 5: dyn→θ messages using original factors
        new_log_dyn_to_theta = compute_dyn_to_theta_msgs(
            p_xnew_tiled, fwd_msgs, bwd_msgs, obs_to_x, action_prior, horizon
        )

        # Step 6: Region beliefs using original factors
        dyn_regions = compute_dyn_region_beliefs(
            p_xnew_tiled, fwd_msgs, bwd_msgs, obs_to_x,
            cavity_dyn, action_prior
        )
        obs_regions = compute_obs_region_beliefs_original(
            obs_idx, fwd_msgs, bwd_msgs, cavity_obs
        )

        return new_log_dyn_to_theta, q_u, dyn_regions, obs_regions

    log_dyn_to_theta, q_u, dyn_region_beliefs, obs_region_beliefs = lax.fori_loop(
        0, n_iterations, body_fn,
        (log_dyn_to_theta, q_u_init, dyn_regions_init, obs_regions_init)
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
    n_iterations,       # int (static, unused — single pass)
):
    """
    Plan actions via reduced Nuijten MP with fixed θ.

    No iteration needed: original factors + fixed θ → single forward-backward
    pass. Region beliefs are computed from the messages.

    Returns:
        action_dist: (n_actions,) distribution over first action
        dyn_region_beliefs: (T, n_states, n_states, n_static, n_actions)
        obs_region_beliefs: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_idx.shape[2]
    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])

    # Original factors (static)
    p_xnew = jax.nn.one_hot(transition_idx, n_states)   # (x_old, θ, u, x_new)
    p_xnew = jnp.transpose(p_xnew, (0, 3, 1, 2))       # (x_old, x_new, θ, u)
    p_xnew_tiled = jnp.broadcast_to(
        p_xnew[None], (horizon, n_states, n_states, n_static, n_actions)
    )

    # Fixed cavities
    cavity_dyn = jnp.tile(q_static_state, (horizon, 1))       # (T, n_static)
    cavity_obs = jnp.tile(q_static_state, (horizon + 1, 1))   # (T+1, n_static)

    # Compute reduced tensor once
    reduced_per_t = compute_reduced_per_t_from_kernels(p_xnew_tiled, cavity_dyn)

    # Uninformative obs messages
    obs_to_x = jnp.ones((horizon + 1, n_states))

    # Single forward-backward pass
    fwd_msgs = forward_pass(
        reduced_per_t, q_current_state, action_prior, obs_to_x, horizon
    )
    bwd_msgs, q_u = backward_pass(
        reduced_per_t, fwd_msgs, goal, action_prior, obs_to_x, horizon
    )

    # Region beliefs using original factors
    dyn_region_beliefs = compute_dyn_region_beliefs(
        p_xnew_tiled, fwd_msgs, bwd_msgs, obs_to_x,
        cavity_dyn, action_prior
    )
    obs_region_beliefs = compute_obs_region_beliefs_original(
        obs_idx, fwd_msgs, bwd_msgs, cavity_obs
    )

    return q_u[0], dyn_region_beliefs, obs_region_beliefs
