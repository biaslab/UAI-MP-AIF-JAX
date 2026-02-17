"""Region-extended loopy BP planning with observation factors.

Extends loopy_bp.py to include observation factors at each timestep, computing
region beliefs for both dynamics and observation factors. Kernels are computed
inline as log_factor - log_channel.

All internal computation is in log-space. Accepts probability-space tensors.

Factor graph:

    p(theta)---theta (single variable, connected to all factors)
                |
    p(x0)--x0--[dyn0]--x1--[dyn1]--x2-- ... --[dyn_{T-1}]--x_T--goal
            |           |           |                        |
         [obs0]_k    [obs1]_k    [obs2]_k               [obsT]_k   (k=1..n_fov)
            |           |           |                        |
          y0,k        y1,k        y2,k                    yT,k     (uniform)

theta cavity: total-minus-self (not prefix sums on equality chain).
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .planning import LOG_ZERO, safe_log
from .messages import safe_log_div
from environments.minigrid import N_CELL_TYPES


# =============================================================================
# Channel damping
# =============================================================================


def damp_log_channel(log_old, log_new, damping, cond_axis):
    """Arithmetic damping: (1-d)*old + d*new in probability space.

    In log-space: logsumexp([log(1-d) + log_old, log(d) + log_new]).

    damping=1.0 -> new channels (no damping)
    damping=0.5 -> arithmetic mean
    """
    log_d = jnp.log(damping)
    log_1md = jnp.log(jnp.maximum(1.0 - damping, 1e-30))

    # Arithmetic mix via logsumexp over a 2-element stack
    stacked = jnp.stack([log_1md + log_old, log_d + log_new])
    damped = logsumexp(stacked, axis=0)

    # Structural zeros: only zero if BOTH old and new are LOG_ZERO
    valid = (log_old > LOG_ZERO / 2) | (log_new > LOG_ZERO / 2)
    damped = jnp.where(valid, damped, LOG_ZERO)

    # Renormalize conditional for numerical stability
    normalizer = logsumexp(damped, axis=cond_axis, keepdims=True)
    return jnp.where(valid, damped - normalizer, LOG_ZERO)


# =============================================================================
# Forward/backward passes (log-space with obs_to_x injection)
# =============================================================================


def compute_log_reduced(log_kernels, log_cavity):
    """
    Compute per-timestep reduced tensors from dynamics kernels (log-space).

    reduced[t](x_new, x_old, u) = Σ_θ κ_t(x_old, x_new, θ, u) · cavity[t](θ)

    Args:
        log_kernels: (T, x_old, x_new, θ, u) log dynamics kernels
        log_cavity: (T, n_static) log per-timestep cavity beliefs on theta

    Returns:
        (T, x_new, x_old, u) per-timestep log reduced tensors
    """
    terms = log_kernels + log_cavity[:, None, None, :, None]
    result = logsumexp(terms, axis=3)  # (T, x_old, x_new, u)
    return result.transpose(0, 2, 1, 3)  # (T, x_new, x_old, u)


def forward_pass(log_reduced_per_t, log_q_x0, log_action_prior, log_obs_to_x, horizon):
    """
    Forward pass with obs->x message injection (log-space).

    Args:
        log_reduced_per_t: (T, x_new, x_old, u)
        log_q_x0: (n_states,) log initial state belief
        log_action_prior: (n_actions,) log prior over actions
        log_obs_to_x: (T+1, n_states) log messages from obs factors to x
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
                 + log_action_prior[None, None, :])
        log_q_next = logsumexp(terms, axis=(1, 2))
        log_q_next = log_q_next - logsumexp(log_q_next)
        return log_fwd.at[t + 1].set(log_q_next)

    return lax.fori_loop(0, horizon, body_fn, log_fwd)


def backward_pass(log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                  log_obs_to_x, horizon):
    """
    Backward pass with obs->x message injection (log-space).

    Returns:
        log_bwd_msgs: (T+1, n_states) log backward messages
        q_u: (T, n_actions) action marginals (probability space)
    """
    n_states = log_goal.shape[0]
    n_actions = log_action_prior.shape[0]
    log_bwd = jnp.zeros((horizon + 1, n_states))
    log_bwd = log_bwd.at[horizon].set(log_goal)
    q_u = jnp.zeros((horizon, n_actions))

    def body_fn(carry, t):
        log_bwd, q_u = carry
        log_bwd_t1 = log_bwd[t + 1] + log_obs_to_x[t + 1]

        # Action marginal
        log_fwd_t = log_fwd_msgs[t] + log_obs_to_x[t]
        terms = (log_reduced_per_t[t]
                 + log_bwd_t1[:, None, None]
                 + log_fwd_t[None, :, None])
        log_msg_to_u = logsumexp(terms, axis=(0, 1))
        q_u_t = jax.nn.softmax(log_msg_to_u + log_action_prior)
        q_u = q_u.at[t].set(q_u_t)

        # Backward message to x_t (no obs_to_x[t])
        terms_bwd = (log_reduced_per_t[t]
                     + log_bwd_t1[:, None, None]
                     + log_action_prior[None, None, :])
        log_bwd_t = logsumexp(terms_bwd, axis=(0, 2))
        log_bwd_t = log_bwd_t - logsumexp(log_bwd_t)
        log_bwd = log_bwd.at[t].set(log_bwd_t)

        return (log_bwd, q_u), None

    (log_bwd, q_u), _ = lax.scan(
        body_fn, (log_bwd, q_u), jnp.arange(horizon - 1, -1, -1)
    )

    return log_bwd, q_u


# =============================================================================
# Messages to/from theta
# =============================================================================


def compute_dyn_to_theta_msgs(log_dyn_kernels, log_fwd_msgs, log_bwd_msgs,
                               log_obs_to_x, log_action_prior, horizon):
    """
    Compute messages from each dynamics factor to theta (log-space).

    Args:
        log_dyn_kernels: (T, x_old, x_new, θ, u) log dynamics kernels
        log_fwd_msgs: (T+1, n_states)
        log_bwd_msgs: (T+1, n_states)
        log_obs_to_x: (T+1, n_states)
        log_action_prior: (n_actions,)
        horizon: T

    Returns:
        log_dyn_to_theta: (T, n_static) log-space messages
    """
    def compute_msg_t(t):
        log_fwd_t = log_fwd_msgs[t] + log_obs_to_x[t]
        log_bwd_t1 = log_bwd_msgs[t + 1] + log_obs_to_x[t + 1]

        terms = (log_dyn_kernels[t]
                 + log_fwd_t[:, None, None, None]
                 + log_bwd_t1[None, :, None, None]
                 + log_action_prior[None, None, None, :])
        return logsumexp(terms, axis=(0, 1, 3))

    return jax.vmap(compute_msg_t)(jnp.arange(horizon))


def compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs):
    """
    Compute messages from observation factors to x variables (log-space).

    Args:
        log_obs_kernels: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
        log_cavity_obs: (T+1, n_static) log cavity beliefs for obs factors

    Returns:
        log_obs_to_x: (T+1, n_states) log-normalized messages
    """
    terms = log_obs_kernels + log_cavity_obs[:, None, None, None, :]
    log_per_k = logsumexp(terms, axis=(2, 4))  # (T+1, n_fov, n_states)
    log_obs_to_x = log_per_k.sum(axis=1)  # product over FOV
    log_obs_to_x = log_obs_to_x - logsumexp(log_obs_to_x, axis=1, keepdims=True)
    return log_obs_to_x


def compute_obs_to_theta_msgs(log_obs_kernels, log_fwd_msgs, log_bwd_msgs,
                               log_obs_to_x):
    """
    Compute aggregated messages from observation factors to theta (log-space).

    Returns:
        log_obs_to_theta: (T+1, n_static) log-space messages
    """
    T_plus_1 = log_fwd_msgs.shape[0]

    def compute_msg_t(t):
        log_x_msg = log_fwd_msgs[t] + log_bwd_msgs[t]
        log_x_msg = log_x_msg - logsumexp(log_x_msg)

        terms = log_obs_kernels[t] + log_x_msg[None, None, :, None]
        log_per_k = logsumexp(terms, axis=(1, 2))  # (n_fov, n_static)
        return log_per_k.sum(axis=0)

    return jax.vmap(compute_msg_t)(jnp.arange(T_plus_1))


def compute_theta_cavities_extended(log_prior, log_dyn_to_theta, log_obs_to_theta):
    """
    Compute cavity messages for theta via total-minus-self (log-space).

    Returns:
        log_cavity_dyn: (T, n_static) log-normalized cavity beliefs for dynamics
        log_cavity_obs: (T+1, n_static) log-normalized cavity beliefs for obs
    """
    total = log_prior + log_dyn_to_theta.sum(axis=0) + log_obs_to_theta.sum(axis=0)

    log_cavity_dyn = total[None, :] - log_dyn_to_theta
    log_cavity_dyn = log_cavity_dyn - logsumexp(log_cavity_dyn, axis=1, keepdims=True)

    log_cavity_obs = total[None, :] - log_obs_to_theta
    log_cavity_obs = log_cavity_obs - logsumexp(log_cavity_obs, axis=1, keepdims=True)

    return log_cavity_dyn, log_cavity_obs


# =============================================================================
# Region beliefs and channels
# =============================================================================


def compute_dyn_region_beliefs(log_dyn_kernels, log_fwd_msgs, log_bwd_msgs,
                                log_obs_to_x, log_cavity_dyn, log_action_prior):
    """
    Compute region beliefs for dynamics factors (unnormalized log-space).

    Returns:
        (T, x_old, x_new, θ, u) unnormalized log beliefs
    """
    T = log_cavity_dyn.shape[0]

    def compute_single_t(t):
        log_fwd_t = log_fwd_msgs[t] + log_obs_to_x[t]
        log_bwd_t1 = log_bwd_msgs[t + 1] + log_obs_to_x[t + 1]

        return (log_dyn_kernels[t]
                + log_fwd_t[:, None, None, None]
                + log_bwd_t1[None, :, None, None]
                + log_cavity_dyn[t][None, None, :, None]
                + log_action_prior[None, None, None, :])

    return jax.vmap(compute_single_t)(jnp.arange(T))


def compute_dyn_channels(log_region_beliefs):
    """
    Compute dynamic channel r(x_new | x_old, u) from log region beliefs.

    Returns:
        (T, x_old, x_new, u) log-conditional r(x_new | x_old, u)
    """
    log_joint = logsumexp(log_region_beliefs, axis=3)  # marginalize θ
    return log_joint - logsumexp(log_joint, axis=2, keepdims=True)


def compute_obs_region_beliefs(log_obs_kernels, log_fwd_msgs, log_bwd_msgs,
                                log_obs_to_x, log_cavity_obs):
    """
    Compute region beliefs for observation factors (unnormalized log-space).

    Returns:
        (T+1, n_fov, N_CELL_TYPES, n_states, n_static) unnormalized log beliefs
    """
    T_plus_1 = log_cavity_obs.shape[0]

    def compute_single_t(t):
        log_x_belief = log_fwd_msgs[t] + log_bwd_msgs[t]

        return (log_obs_kernels[t]
                + log_x_belief[None, None, :, None]
                + log_cavity_obs[t][None, None, None, :])

    return jax.vmap(compute_single_t)(jnp.arange(T_plus_1))


def compute_obs_channels(log_region_beliefs):
    """
    Compute obs channel r(y|x,θ) from log region beliefs.

    Returns:
        (T+1, n_fov, N_CELL_TYPES, n_states, n_static) log r(y|x,θ)
    """
    return log_region_beliefs - logsumexp(log_region_beliefs, axis=2, keepdims=True)


# =============================================================================
# Main planning function
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def region_extended_loopy_bp_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta
    transition_tensor,    # (n_states, n_states, n_static, n_actions)
    observation_tensor,   # (fov_w, fov_h, N_CELL_TYPES, n_states, n_static)
    goal,                 # (n_states,)
    horizon,              # int (static)
    n_iterations,         # int (static)
    damping=1.0,          # float - channel update damping (1.0 = no damping)
) -> jnp.ndarray:
    """
    Plan actions via region-extended loopy BP with observation factors.

    Accepts full probability tensors, logs once at the top, all internal
    computation in log-space.

    Returns:
        action_dist: (n_actions,)
        log_dyn_channels: (T, n_states, n_states, n_actions)
        log_obs_channels: (T+1, n_fov, N_CELL_TYPES, n_states, n_static)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]
    fov_w, fov_h = observation_tensor.shape[0], observation_tensor.shape[1]
    n_fov = fov_w * fov_h

    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
    log_action_prior = safe_log(action_prior)

    # Log once at top
    log_T = safe_log(transition_tensor)                   # (x_new, x_old, θ, u)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)           # (x_old, x_new, θ, u)
    log_B_flat = safe_log(observation_tensor.reshape(n_fov, N_CELL_TYPES, n_states, n_static))
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)
    log_goal = safe_log(goal)

    # Initialize messages
    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    log_obs_to_theta = jnp.zeros((horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))

    # Initial channels: zeros → kernel = original factor
    log_dyn_channels_init = jnp.zeros((horizon, n_states, n_states, n_actions))
    log_obs_channels_init = jnp.zeros((horizon + 1, n_fov, N_CELL_TYPES, n_states, n_static))

    def body_fn(i, carry):
        log_dyn_to_theta, log_obs_to_theta, _, log_dyn_channels, log_obs_channels = carry

        # Step 1: theta cavities
        log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
            log_prior_theta, log_dyn_to_theta, log_obs_to_theta
        )

        # Step 2: Inline kernels (factor / channel in log-space)
        log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
        log_obs_kernels = safe_log_div(log_B_flat[None], log_obs_channels)

        # Step 3: Reduced tensors
        log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_cavity_dyn)

        # Step 4: obs->x messages
        log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs)

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
        new_log_dyn_to_theta = compute_dyn_to_theta_msgs(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_action_prior, horizon
        )

        # Step 8: obs->theta messages
        new_log_obs_to_theta = compute_obs_to_theta_msgs(
            log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
        )

        # Step 9: Region beliefs
        log_dyn_regions = compute_dyn_region_beliefs(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_dyn, log_action_prior
        )
        log_obs_regions = compute_obs_region_beliefs(
            log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_obs
        )

        # Step 10: Channels from region beliefs (with damping)
        raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
        raw_log_obs_channels = compute_obs_channels(log_obs_regions)

        new_log_dyn_channels = damp_log_channel(
            log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)
        new_log_obs_channels = damp_log_channel(
            log_obs_channels, raw_log_obs_channels, damping, cond_axis=2)

        return (new_log_dyn_to_theta, new_log_obs_to_theta, q_u,
                new_log_dyn_channels, new_log_obs_channels)

    result = lax.fori_loop(
        0, n_iterations, body_fn,
        (log_dyn_to_theta, log_obs_to_theta, q_u_init,
         log_dyn_channels_init, log_obs_channels_init)
    )
    _, _, q_u, log_dyn_channels, log_obs_channels = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_dyn_channels, log_obs_channels
