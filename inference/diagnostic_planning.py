"""Diagnostic wrappers for BP and AIF planning.

Unrolls the outer iteration loop in Python so that all intermediates
(q_state, q_u, q_theta, channels, messages) can be captured per iteration.
Calls the same JIT-compiled sub-functions used by the production planners.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp

from .planning import (
    forward_pass,
    backward_pass,
    marginalize_static_indexed,
)
import jax

from .aif_planning import (
    N_CELL_TYPES,
    N_FOV,
    compute_cavities,
    compute_modified_kernel,
    compute_all_obs_msgs_to_x,
    compute_obs_msgs_to_theta_per_t,
    aif_forward_pass,
    aif_backward_pass_with_messages,
    compute_theta_messages_from_dynamics,
    update_q_theta,
    channel_update_dynamics,
    channel_update_obs,
)
from .messages import EPSILON


@dataclass
class BPDiagnostics:
    """Diagnostics from BP planning."""
    q_state: jnp.ndarray          # (T+1, n_states) final state beliefs
    q_u: jnp.ndarray              # (T, n_actions) final action beliefs
    reduced_tensor: jnp.ndarray   # (n_states, n_states, n_actions)
    # Per-iteration history
    q_state_history: list = field(default_factory=list)
    q_u_history: list = field(default_factory=list)


@dataclass
class AIFDiagnostics:
    """Diagnostics from AIF planning."""
    q_state: jnp.ndarray                # (T+1, n_states) final state beliefs
    q_u: jnp.ndarray                    # (T, n_actions) final action beliefs
    q_theta: jnp.ndarray                # (n_static,) final theta belief
    reduced_per_t: jnp.ndarray          # (T, n_states, n_states, n_actions) from final iteration
    K_mod: jnp.ndarray                  # (T, n_states, n_states, n_actions) per-timestep modified kernel
    r_x: jnp.ndarray                    # (T, n_states, n_states, n_actions) per-timestep dynamics channel
    obs_msgs_to_x: jnp.ndarray         # (T+1, n_states) per-timestep log obs msgs to x
    log_obs_msgs_per_t: jnp.ndarray    # (T, n_static) per-timestep log obs msgs to theta
    log_dyn_msgs_per_t: jnp.ndarray    # (T, n_static) per-timestep log dyn msgs to theta
    backward_msgs: jnp.ndarray          # (T+1, n_states) final backward messages
    cavities_dyn: jnp.ndarray           # (T, n_static) final dyn cavities
    cavities_obs: jnp.ndarray           # (T, n_static) final obs cavities
    # Per-iteration history
    q_state_history: list = field(default_factory=list)
    q_u_history: list = field(default_factory=list)
    q_theta_history: list = field(default_factory=list)
    K_mod_history: list = field(default_factory=list)
    r_x_history: list = field(default_factory=list)
    obs_msgs_to_x_history: list = field(default_factory=list)
    log_obs_msgs_per_t_history: list = field(default_factory=list)
    log_dyn_msgs_per_t_history: list = field(default_factory=list)
    cavities_dyn_history: list = field(default_factory=list)
    cavities_obs_history: list = field(default_factory=list)


def diagnostic_planning_indexed(
    q_current_state: jnp.ndarray,
    q_static_state: jnp.ndarray,
    transition_idx: jnp.ndarray,
    goal: jnp.ndarray,
    horizon: int,
    n_iterations: int = 1,
) -> BPDiagnostics:
    """
    BP planning with full diagnostics. Mirrors planning_indexed() but unrolls
    iterations in Python, capturing intermediates.

    Args:
        q_current_state: (n_states,) current belief over state
        q_static_state: (n_static,) belief over static configuration
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T
        n_iterations: number of forward-backward passes

    Returns:
        BPDiagnostics with full iteration history
    """
    n_states = q_current_state.shape[0]

    reduced_tensor = marginalize_static_indexed(transition_idx, q_static_state, n_states)
    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])

    q_u = jnp.tile(action_prior, (horizon, 1))
    q_state = jnp.concatenate([
        q_current_state[None, :],
        jnp.ones((horizon, n_states)) / n_states
    ], axis=0)

    q_state_history = []
    q_u_history = []

    for _ in range(n_iterations):
        q_state = forward_pass(reduced_tensor, q_state, action_prior, horizon)
        q_u = backward_pass(reduced_tensor, q_state, goal, action_prior, horizon)
        q_state_history.append(q_state)
        q_u_history.append(q_u)

    return BPDiagnostics(
        q_state=q_state,
        q_u=q_u,
        reduced_tensor=reduced_tensor,
        q_state_history=q_state_history,
        q_u_history=q_u_history,
    )


def diagnostic_aif_planning_indexed(
    q_current_state: jnp.ndarray,
    q_static_state: jnp.ndarray,
    transition_idx: jnp.ndarray,
    observation_idx: jnp.ndarray,
    goal: jnp.ndarray,
    horizon: int,
    n_iterations: int = 1,
) -> AIFDiagnostics:
    """
    AIF planning with full diagnostics. Mirrors aif_planning_indexed() but
    unrolls iterations in Python, capturing intermediates.

    Args:
        q_current_state: (n_states,) current belief over state
        q_static_state: (n_static,) belief over static configuration
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        observation_idx: (7, 7, n_states, n_static) -> cell_type index
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T
        n_iterations: number of AIF iterations

    Returns:
        AIFDiagnostics with full iteration history
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_idx.shape[2]

    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])

    # Initialize beliefs (mirrors aif_planning_indexed)
    q_u = jnp.tile(action_prior, (horizon, 1))
    q_state = jnp.concatenate([
        q_current_state[None, :],
        jnp.ones((horizon, n_states)) / n_states
    ], axis=0)
    q_theta = q_static_state

    r_x = jnp.ones((horizon, n_states, n_states, n_actions)) / n_states
    r_y = jnp.ones((horizon + 1, N_FOV, N_CELL_TYPES, n_states, n_static)) / N_CELL_TYPES

    # Initialize per-timestep factor→θ messages to zero (cavities = prior)
    log_dyn_msgs_per_t = jnp.zeros((horizon, n_static))
    log_obs_msgs_per_t = jnp.zeros((horizon, n_static))

    obs_idx_flat = observation_idx.reshape(N_FOV, n_states, n_static)
    prior_theta = q_static_state
    log_prior = jnp.log(prior_theta + EPSILON)

    # History lists
    q_state_history = []
    q_u_history = []
    q_theta_history = []
    K_mod_history = []
    r_x_history = []
    obs_msgs_to_x_history = []
    log_obs_msgs_per_t_history = []
    log_dyn_msgs_per_t_history = []
    cavities_dyn_history = []
    cavities_obs_history = []

    # Intermediates that persist across iterations
    reduced_per_t = None
    K_mod = None
    log_obs_msgs_to_x = None
    backward_msgs = None
    cavities_dyn = None
    cavities_obs = None

    for _ in range(n_iterations):
        # Step 1: Compute per-timestep cavity messages for θ
        cavities_dyn, cavities_obs = compute_cavities(
            log_prior, log_dyn_msgs_per_t, log_obs_msgs_per_t
        )

        # Step 2: Per-timestep reduced tensors using cavity_dyn[t]
        reduced_per_t = jax.vmap(
            lambda q: marginalize_static_indexed(transition_idx, q, n_states)
        )(cavities_dyn)

        # Step 3: Per-timestep modified kernel K_mod[t] = T_t / r_x[t]
        K_mod = compute_modified_kernel(reduced_per_t, r_x)

        # Step 4: θ for obs: uniform at t=0 (no obs factor), cavities for t=1..T
        uniform_theta = jnp.ones(n_static) / n_static
        q_theta_for_obs = jnp.concatenate(
            [uniform_theta[None, :], cavities_obs], axis=0
        )

        # Step 5: Per-timestep obs messages to x using per-timestep θ cavities
        log_obs_msgs_to_x = compute_all_obs_msgs_to_x(obs_idx_flat, r_y, q_theta_for_obs)
        maxes = log_obs_msgs_to_x.max(axis=1, keepdims=True)
        log_obs_msgs_to_x = log_obs_msgs_to_x - maxes
        # No observation factor at x₀ (degree 2: only p(x₀) and dyn_1)
        log_obs_msgs_to_x = log_obs_msgs_to_x.at[0].set(0.0)

        # Step 6: Forward pass with per-timestep K_mod and obs messages
        q_state = aif_forward_pass(K_mod, q_state, action_prior, log_obs_msgs_to_x, horizon)

        # Step 7: Backward pass with per-timestep K_mod and obs messages
        q_u, backward_msgs = aif_backward_pass_with_messages(
            K_mod, q_state, goal, action_prior, log_obs_msgs_to_x, horizon
        )

        # Step 8: Per-timestep obs→θ messages (uses backward_msgs)
        log_obs_msgs_per_t = compute_obs_msgs_to_theta_per_t(
            obs_idx_flat, r_y, q_state, backward_msgs, horizon
        )

        # Step 9: Per-timestep dyn→θ messages
        log_dyn_msgs_per_t = compute_theta_messages_from_dynamics(
            transition_idx, q_state, backward_msgs,
            action_prior, r_x, log_obs_msgs_to_x, horizon
        )

        # Step 10: Update q_theta from per-timestep messages
        q_theta = update_q_theta(prior_theta, log_dyn_msgs_per_t, log_obs_msgs_per_t)

        # Step 11: Update per-timestep dynamics channel r_x
        r_x = channel_update_dynamics(
            K_mod, q_state, q_u, backward_msgs, log_obs_msgs_to_x, horizon
        )

        # Step 12: Update per-timestep observation channel r_y
        r_y = channel_update_obs(obs_idx_flat, n_states, n_static, horizon)

        # Store history
        q_state_history.append(q_state)
        q_u_history.append(q_u)
        q_theta_history.append(q_theta)
        K_mod_history.append(K_mod)
        r_x_history.append(r_x)
        obs_msgs_to_x_history.append(log_obs_msgs_to_x)
        log_obs_msgs_per_t_history.append(log_obs_msgs_per_t)
        log_dyn_msgs_per_t_history.append(log_dyn_msgs_per_t)
        cavities_dyn_history.append(cavities_dyn)
        cavities_obs_history.append(cavities_obs)

    return AIFDiagnostics(
        q_state=q_state,
        q_u=q_u,
        q_theta=q_theta,
        reduced_per_t=reduced_per_t,
        K_mod=K_mod,
        r_x=r_x,
        obs_msgs_to_x=log_obs_msgs_to_x,
        log_obs_msgs_per_t=log_obs_msgs_per_t,
        log_dyn_msgs_per_t=log_dyn_msgs_per_t,
        backward_msgs=backward_msgs,
        cavities_dyn=cavities_dyn,
        cavities_obs=cavities_obs,
        q_state_history=q_state_history,
        q_u_history=q_u_history,
        q_theta_history=q_theta_history,
        K_mod_history=K_mod_history,
        r_x_history=r_x_history,
        obs_msgs_to_x_history=obs_msgs_to_x_history,
        log_obs_msgs_per_t_history=log_obs_msgs_per_t_history,
        log_dyn_msgs_per_t_history=log_dyn_msgs_per_t_history,
        cavities_dyn_history=cavities_dyn_history,
        cavities_obs_history=cavities_obs_history,
    )
