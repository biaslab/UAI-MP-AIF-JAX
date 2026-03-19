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

theta: uses prior p(θ) directly (no cavity messages).
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .messages import LOG_ZERO, safe_log
from .messages import safe_log_div


# =============================================================================
# Channel damping
# =============================================================================


def damp_log_channel(log_old, log_new, damping, cond_axis,
                     log_prev=None, momentum=0.0):
    """Geometric damping with optional inertial (momentum) term.

    Geometric:  (1-α)*log_old + α*log_new
    Heavy-ball: (1-α+β)*log_old + α*log_new - β*log_prev

    damping=1.0 -> new channels (no damping)
    damping=0.5 -> geometric mean in probability space
    """
    alpha = damping
    beta = momentum
    if log_prev is not None:
        safe_prev = jnp.where(log_prev > LOG_ZERO / 2, log_prev, log_old)
        damped = (1.0 - alpha + beta) * log_old + alpha * log_new - beta * safe_prev
    else:
        damped = (1.0 - alpha) * log_old + alpha * log_new

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


def forward_pass(log_reduced_per_t, log_q_x0, log_action_prior, log_obs_to_x, horizon,
                 log_prev_fwd=None, msg_damping=1.0):
    """
    Forward pass with obs->x message injection (log-space).

    Args:
        log_reduced_per_t: (T, x_new, x_old, u)
        log_q_x0: (n_states,) log initial state belief
        log_action_prior: (n_actions,) log prior over actions
        log_obs_to_x: (T+1, n_states) log messages from obs factors to x
        horizon: T
        log_prev_fwd: (T+1, n_states) previous iteration's fwd messages (optional)
        msg_damping: float, inertial damping weight (1.0 = no damping)

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
        # Inertial damping against previous iteration's message
        if log_prev_fwd is not None:
            log_q_next = (1.0 - msg_damping) * log_prev_fwd[t + 1] + msg_damping * log_q_next
            log_q_next = log_q_next - logsumexp(log_q_next)
        return log_fwd.at[t + 1].set(log_q_next)

    return lax.fori_loop(0, horizon, body_fn, log_fwd)


def backward_pass(log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                  log_obs_to_x, horizon, log_prev_bwd=None, msg_damping=1.0):
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
        # Inertial damping against previous iteration's message
        if log_prev_bwd is not None:
            log_bwd_t = (1.0 - msg_damping) * log_prev_bwd[t] + msg_damping * log_bwd_t
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
    log_fwd_t = (log_fwd_msgs[:-1] + log_obs_to_x[:-1])[:, :, None, None, None]
    log_bwd_t1 = (log_bwd_msgs[1:] + log_obs_to_x[1:])[:, None, :, None, None]

    terms = (log_dyn_kernels
             + log_fwd_t
             + log_bwd_t1
             + log_action_prior[None, None, None, None, :])
    return logsumexp(terms, axis=(1, 2, 4))


def compute_pref_to_x_msgs(log_C, log_cavity_pref):
    """
    Compute messages from preference factors to x variables (log-space).

    Args:
        log_C: (n_states, n_static) log preference factor C(x, θ)
        log_cavity_pref: (T+1, n_static) log cavity beliefs for pref factors

    Returns:
        log_pref_to_x: (T+1, n_states) log-normalized messages
    """
    terms = log_C[None, :, :] + log_cavity_pref[:, None, :]  # (T+1, n_states, n_static)
    log_pref_to_x = logsumexp(terms, axis=2)  # (T+1, n_states)
    return log_pref_to_x - logsumexp(log_pref_to_x, axis=1, keepdims=True)


def compute_pref_to_theta_msgs(log_C, log_fwd_msgs, log_bwd_msgs, log_obs_to_x):
    """
    Compute messages from preference factors to theta (log-space).

    The x belief for pref→θ at time t is fwd[t]·bwd[t]·obs_to_x[t],
    which excludes pref_to_x[t] (correct cavity).

    Args:
        log_C: (n_states, n_static) log preference factor C(x, θ)
        log_fwd_msgs: (T+1, n_states)
        log_bwd_msgs: (T+1, n_states)
        log_obs_to_x: (T+1, n_states) obs-only messages to x

    Returns:
        log_pref_to_theta: (T+1, n_static) log-space messages
    """
    log_x_belief = log_fwd_msgs + log_bwd_msgs + log_obs_to_x  # (T+1, n_states)
    terms = log_C[None, :, :] + log_x_belief[:, :, None]  # (T+1, n_states, n_static)
    return logsumexp(terms, axis=1)  # (T+1, n_static)


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
                               log_obs_to_x, log_extra_to_x=None):
    """
    Compute aggregated messages from observation factors to theta (log-space).

    When log_extra_to_x is provided (e.g. pref_to_x), it is included in the
    x belief so that the cavity correctly includes all non-obs messages.

    Returns:
        log_obs_to_theta: (T+1, n_static) log-space messages
    """
    log_x_belief = log_fwd_msgs + log_bwd_msgs
    if log_extra_to_x is not None:
        log_x_belief = log_x_belief + log_extra_to_x
    log_x_msg = log_x_belief[:, None, None, :, None]

    terms = log_obs_kernels + log_x_msg
    log_per_k = logsumexp(terms, axis=(2, 3))  # (T+1, n_fov, n_static)
    return log_per_k.sum(axis=1)


def compute_theta_cavities_extended(log_prior, log_dyn_to_theta, log_obs_to_theta,
                                     log_pref_to_theta=None):
    """
    Compute cavity messages for theta via forward-backward prefix sums (log-space).

    More numerically stable than total-minus-self: avoids subtracting large
    accumulated sums. There are T dyn messages and T+1 obs messages.
    Optionally, T+1 pref messages for per-step preference factors.

    Returns:
        log_cavity_dyn: (T, n_static) log-normalized cavity beliefs for dynamics
        log_cavity_obs: (T+1, n_static) log-normalized cavity beliefs for obs
        log_cavity_pref: (T+1, n_static) only when log_pref_to_theta is provided
    """
    n_static = log_dyn_to_theta.shape[1]
    zeros = jnp.zeros((1, n_static))

    total_dyn = log_dyn_to_theta.sum(axis=0)
    total_obs = log_obs_to_theta.sum(axis=0)
    total_pref = log_pref_to_theta.sum(axis=0) if log_pref_to_theta is not None else jnp.zeros(n_static)

    # Dyn cavities: exclude dyn[t], include all obs + all pref
    base_dyn = log_prior + total_obs + total_pref
    dyn_fwd = jnp.concatenate([zeros, jnp.cumsum(log_dyn_to_theta, axis=0)[:-1]]) + base_dyn
    dyn_bwd = jnp.concatenate([jnp.cumsum(log_dyn_to_theta[::-1], axis=0)[::-1][1:], zeros])
    log_cavity_dyn = dyn_fwd + dyn_bwd
    log_cavity_dyn = log_cavity_dyn - logsumexp(log_cavity_dyn, axis=1, keepdims=True)

    # Obs cavities: exclude obs[t], include all dyn + all pref
    base_obs = log_prior + total_dyn + total_pref
    obs_fwd = jnp.concatenate([zeros, jnp.cumsum(log_obs_to_theta, axis=0)[:-1]]) + base_obs
    obs_bwd = jnp.concatenate([jnp.cumsum(log_obs_to_theta[::-1], axis=0)[::-1][1:], zeros])
    log_cavity_obs = obs_fwd + obs_bwd
    log_cavity_obs = log_cavity_obs - logsumexp(log_cavity_obs, axis=1, keepdims=True)

    if log_pref_to_theta is None:
        return log_cavity_dyn, log_cavity_obs

    # Pref cavities: exclude pref[t], include all dyn + all obs
    base_pref = log_prior + total_dyn + total_obs
    pref_fwd = jnp.concatenate([zeros, jnp.cumsum(log_pref_to_theta, axis=0)[:-1]]) + base_pref
    pref_bwd = jnp.concatenate([jnp.cumsum(log_pref_to_theta[::-1], axis=0)[::-1][1:], zeros])
    log_cavity_pref = pref_fwd + pref_bwd
    log_cavity_pref = log_cavity_pref - logsumexp(log_cavity_pref, axis=1, keepdims=True)

    return log_cavity_dyn, log_cavity_obs, log_cavity_pref


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
    log_fwd_t = (log_fwd_msgs[:-1] + log_obs_to_x[:-1])[:, :, None, None, None]
    log_bwd_t1 = (log_bwd_msgs[1:] + log_obs_to_x[1:])[:, None, :, None, None]

    return (log_dyn_kernels
            + log_fwd_t
            + log_bwd_t1
            + log_cavity_dyn[:, None, None, :, None]
            + log_action_prior[None, None, None, None, :])


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
    log_x_belief = (log_fwd_msgs + log_bwd_msgs)[:, None, None, :, None]

    return (log_obs_kernels
            + log_x_belief
            + log_cavity_obs[:, None, None, None, :])


def compute_obs_channels(log_region_beliefs):
    """
    Compute obs channel r(y|x,θ) from log region beliefs.

    Returns:
        (T+1, n_fov, N_CELL_TYPES, n_states, n_static) log r(y|x,θ)
    """
    return log_region_beliefs - logsumexp(log_region_beliefs, axis=2, keepdims=True)


def compute_marginal_obs_channels(log_region_beliefs):
    """Compute marginal obs channel r(y|x) from obs region beliefs.

    Marginalizes θ, then normalizes over obs_types.

    Returns:
        (T+1, n_fov, N_CELL_TYPES, n_states) log r(y|x)
    """
    log_marginal = logsumexp(log_region_beliefs, axis=4)  # Σ_θ
    return log_marginal - logsumexp(log_marginal, axis=2, keepdims=True)


# =============================================================================
# Main planning function
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def region_extended_loopy_bp_planning(
    q_current_state,      # (n_states,)
    q_static_state,       # (n_static,) prior on theta
    transition_tensor,    # (n_states, n_states, n_static, n_actions)
    observation_tensor,   # (n_channels, n_obs_types, n_states, n_static)
    goal,                 # (n_states,) or (n_states, n_static) preference
    horizon,              # int (static)
    n_iterations,         # int (static)
    damping=1.0,          # float - channel update damping (1.0 = no damping)
    action_prior=None,    # (n_actions,) prior over actions. If None, uniform.
    momentum=0.0,         # float - inertial (heavy-ball) momentum coefficient
) -> jnp.ndarray:
    """
    Plan actions via region-extended loopy BP with observation factors.

    Accepts full probability tensors, logs once at the top, all internal
    computation in log-space.

    When goal is 2D (n_states, n_static), it is treated as a per-config
    preference factor C(x, θ) applied at every timestep.

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

    # Precompute prior broadcasts (replacing theta cavity messages)
    log_prior_dyn = jnp.broadcast_to(log_prior_theta[None, :], (horizon, n_static))
    log_prior_obs = jnp.broadcast_to(log_prior_theta[None, :], (horizon + 1, n_static))

    q_u_init = jnp.zeros((horizon, n_actions))
    log_fwd_prev_init = jnp.zeros((horizon + 1, n_states))
    log_bwd_prev_init = jnp.zeros((horizon + 1, n_states))

    # Initial dyn channels: r(x_new | x_old, u) from θ-marginalized transition
    # This gives meaningful conditionals so damping doesn't corrupt with uniform.
    log_dyn_ch0 = logsumexp(log_T + log_prior_theta[None, None, :, None], axis=2)
    log_dyn_ch0 = log_dyn_ch0 - logsumexp(log_dyn_ch0, axis=0, keepdims=True)  # normalize over x_new
    log_dyn_ch0 = log_dyn_ch0.transpose(1, 0, 2)  # (x_old, x_new, u)
    log_dyn_channels_init = jnp.broadcast_to(log_dyn_ch0[None], (horizon, n_states, n_states, n_actions))

    # Initial obs channels: r(y | x, θ) = B(y | x, θ) (already a proper conditional)
    log_obs_ch0 = log_B_flat - logsumexp(log_B_flat, axis=1, keepdims=True)  # normalize over obs_type
    log_obs_channels_init = jnp.broadcast_to(log_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states, n_static))

    # Initial marginal obs channels: r(y | x) by marginalizing θ with prior weighting
    log_marginal_obs_ch0 = logsumexp(
        log_B_flat + log_prior_theta[None, None, None, :], axis=3)
    log_marginal_obs_ch0 = log_marginal_obs_ch0 - logsumexp(log_marginal_obs_ch0, axis=1, keepdims=True)
    log_marginal_obs_channels_init = jnp.broadcast_to(
        log_marginal_obs_ch0[None], (horizon + 1, n_fov, n_obs_types, n_states))

    if has_pref:
        def body_fn(i, carry):
            (_, log_dyn_channels, log_obs_channels, log_marginal_obs_channels,
             log_dyn_ch_prev, log_obs_ch_prev, log_marg_obs_ch_prev,
             log_fwd_prev, log_bwd_prev) = carry

            # Step 1: Inline kernels
            log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            # Step 2: Reduced tensors (use prior instead of cavity)
            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_prior_dyn)

            # Step 3: obs->x and pref->x messages (use prior instead of cavity)
            log_obs_to_x = compute_obs_to_x_msgs(log_obs_kernels, log_prior_obs)
            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_prior_obs)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            # Step 4: Forward pass (with combined local messages + inertial damping)
            log_fwd_msgs = forward_pass(
                log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping
            )

            # Step 5: Backward pass (uniform terminal, pref enters via local_to_x)
            log_bwd_msgs, q_u = backward_pass(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                log_local_to_x, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping
            )

            # Step 6: Region beliefs (using combined local_to_x, prior instead of cavity)
            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_prior_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_prior_obs
            )

            # Step 7: Channels from region beliefs (with geometric + momentum damping)
            raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)

            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2,
                log_prev=log_dyn_ch_prev, momentum=momentum)
            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2,
                log_prev=log_obs_ch_prev, momentum=momentum)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2,
                log_prev=log_marg_obs_ch_prev, momentum=momentum)

            return (q_u, new_log_dyn_channels, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_dyn_channels, log_obs_channels, log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init)
        )
        q_u, log_dyn_channels, log_obs_channels, _, _, _, _, _, _ = result
    else:
        def body_fn(i, carry):
            (_, log_dyn_channels, log_obs_channels, log_marginal_obs_channels,
             log_dyn_ch_prev, log_obs_ch_prev, log_marg_obs_ch_prev,
             log_fwd_prev, log_bwd_prev) = carry

            # Step 1: Inline kernels
            log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
            log_obs_kernels = (log_B_flat[None] + log_obs_channels
                               + safe_log_div(log_obs_channels,
                                              log_marginal_obs_channels[:, :, :, :, None]))

            # Step 2: Reduced tensors (use prior instead of cavity)
            log_reduced_per_t = compute_log_reduced(log_dyn_kernels, log_prior_dyn)

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

            # Step 6: Region beliefs (use prior instead of cavity)
            log_dyn_regions = compute_dyn_region_beliefs(
                log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_prior_dyn, log_action_prior
            )
            log_obs_regions = compute_obs_region_beliefs(
                log_obs_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_prior_obs
            )

            # Step 7: Channels from region beliefs (with geometric + momentum damping)
            raw_log_dyn_channels = compute_dyn_channels(log_dyn_regions)
            raw_log_obs_channels = compute_obs_channels(log_obs_regions)
            raw_log_marginal_obs_channels = compute_marginal_obs_channels(log_obs_regions)

            new_log_dyn_channels = damp_log_channel(
                log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2,
                log_prev=log_dyn_ch_prev, momentum=momentum)
            new_log_obs_channels = damp_log_channel(
                log_obs_channels, raw_log_obs_channels, damping, cond_axis=2,
                log_prev=log_obs_ch_prev, momentum=momentum)
            new_log_marginal_obs_channels = damp_log_channel(
                log_marginal_obs_channels, raw_log_marginal_obs_channels, damping, cond_axis=2,
                log_prev=log_marg_obs_ch_prev, momentum=momentum)

            return (q_u, new_log_dyn_channels, new_log_obs_channels,
                    new_log_marginal_obs_channels,
                    log_dyn_channels, log_obs_channels, log_marginal_obs_channels,
                    log_fwd_msgs, log_bwd_msgs)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (q_u_init, log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_dyn_channels_init, log_obs_channels_init,
             log_marginal_obs_channels_init,
             log_fwd_prev_init, log_bwd_prev_init)
        )
        q_u, log_dyn_channels, log_obs_channels, _, _, _, _, _, _ = result

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist, log_dyn_channels, log_obs_channels
