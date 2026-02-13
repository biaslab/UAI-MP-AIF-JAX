"""Reduced region-extended planning with fixed θ.

Like region_extended_loopy_bp.py but treats θ as known (fixed at q_static_state).
Skips all θ backward messages and cavity computation, but keeps observation factors
and kernel reparametrization.

This is to region_extended_loopy_bp.py what planning.py is to loopy_bp.py.

Factor graph is the same, but θ is clamped rather than inferred:

    θ = q_static_state (fixed)
                |
    p(x0)--x0--[dyn0]--x1--[dyn1]--x2-- ... --[dyn_{T-1}]--x_T--goal
            |           |           |                        |
         [obs0]_k    [obs1]_k    [obs2]_k               [obsT]_k
            |           |           |                        |
          y0,k        y1,k        y2,k                    yT,k     (uniform)

Iterations are still meaningful because kernel reparametrization evolves:
region beliefs → channels → kernels change each iteration.
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
    compute_obs_to_x_msgs,
    compute_dyn_region_beliefs,
    compute_obs_region_beliefs,
    compute_dyn_channels,
    compute_obs_channels_from_beliefs,
    compute_dyn_kernels,
    compute_obs_kernels,
)
from environments.minigrid import N_CELL_TYPES


@partial(jax.jit, static_argnums=(5, 6))
def reduced_region_extended_planning_indexed(
    q_current_state,    # (n_states,)
    q_static_state,     # (n_static,) prior on theta (used as fixed cavity)
    transition_idx,     # (n_states, n_static, n_actions)
    obs_idx,            # (7, 7, n_states, n_static) -> cell_type
    goal,               # (n_states,)
    horizon,            # int (static)
    n_iterations,       # int (static)
) -> jnp.ndarray:
    """
    Plan actions via reduced region-extended BP with fixed θ.

    Same as region_extended_loopy_bp_planning_indexed but θ is fixed at
    q_static_state. No θ backward messages or cavity computation. Kernel
    reparametrization still evolves across iterations.

    Args:
        q_current_state: (n_states,) current belief over dynamic state
        q_static_state: (n_static,) belief over static configuration theta (fixed)
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        obs_idx: (7, 7, n_states, n_static) -> cell_type index
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: number of iterations (static for JIT)

    Returns:
        action_dist: (n_actions,) distribution over first action
        dyn_channels: (T, n_states, n_states, n_actions)
        obs_channels: (T+1, 49, n_cell_types, n_states, n_static)
        dyn_kernels: (T, n_states, n_states, n_static, n_actions)
        obs_kernels: (T+1, 7, 7, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_idx.shape[2]
    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])

    # Fixed cavities: θ = q_static_state tiled over time
    cavity_dyn = jnp.tile(q_static_state, (horizon, 1))         # (T, n_static)
    cavity_obs = jnp.tile(q_static_state, (horizon + 1, 1))     # (T+1, n_static)

    # Initialize carry
    q_u_init = jnp.zeros((horizon, n_actions))
    dyn_channels_init = jnp.zeros((horizon, n_states, n_states, n_actions))

    obs_flat = obs_idx.reshape(49, n_states, n_static)
    r_init = jax.nn.one_hot(obs_flat, N_CELL_TYPES)              # (49, n_states, n_static, N_CELL_TYPES)
    r_init = jnp.transpose(r_init, (0, 3, 1, 2))                 # (49, N_CELL_TYPES, n_states, n_static)
    obs_channels_init = jnp.broadcast_to(
        r_init[None], (horizon + 1, 49, N_CELL_TYPES, n_states, n_static)
    )

    p_xnew = jax.nn.one_hot(transition_idx, n_states)           # (x_old, θ, u, x_new)
    p_xnew = jnp.transpose(p_xnew, (0, 3, 1, 2))               # (x_old, x_new, θ, u)
    dyn_kernels_init = jnp.broadcast_to(
        p_xnew[None], (horizon, n_states, n_states, n_static, n_actions)
    )
    obs_kernels_init = jnp.ones((horizon + 1, 7, 7, n_states, n_static))

    def body_fn(_, carry):
        q_u, dyn_channels, obs_channels, dyn_kernels, obs_kernels = carry

        # Step 2: Per-timestep reduced tensors from dyn-kernels + FIXED cavity
        reduced_per_t = compute_reduced_per_t_from_kernels(
            dyn_kernels, cavity_dyn
        )

        # Step 3: obs->x messages from obs kernels + FIXED cavity
        obs_to_x = compute_obs_to_x_msgs(obs_kernels, cavity_obs, horizon)

        # Step 4: Forward pass (with obs->x)
        fwd_msgs = forward_pass(
            reduced_per_t, q_current_state, action_prior, obs_to_x, horizon
        )

        # Step 5: Backward pass (with obs->x) + action marginals
        bwd_msgs, q_u = backward_pass(
            reduced_per_t, fwd_msgs, goal, action_prior, obs_to_x, horizon
        )

        # Steps 6-7: REMOVED (no θ backward messages)

        # Step 8: Region beliefs (use FIXED cavities)
        dyn_regions = compute_dyn_region_beliefs(
            dyn_kernels, fwd_msgs, bwd_msgs, obs_to_x,
            cavity_dyn, action_prior
        )
        obs_regions = compute_obs_region_beliefs(
            obs_kernels, fwd_msgs, bwd_msgs, obs_to_x, cavity_obs
        )

        # Step 9: Channel distributions from region beliefs
        dyn_channels = compute_dyn_channels(dyn_regions)
        obs_channels = compute_obs_channels_from_beliefs(obs_regions)

        # Step 10: Dynamics kernels
        dyn_kernels = compute_dyn_kernels(transition_idx, dyn_channels, n_states)

        # Step 11: Observation kernels
        obs_kernels = compute_obs_kernels(obs_idx, obs_channels)

        return q_u, dyn_channels, obs_channels, dyn_kernels, obs_kernels

    q_u, dyn_channels, obs_channels, dyn_kernels, obs_kernels = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, dyn_channels_init, obs_channels_init, dyn_kernels_init, obs_kernels_init)
    )

    return q_u[0], dyn_channels, obs_channels, dyn_kernels, obs_kernels
