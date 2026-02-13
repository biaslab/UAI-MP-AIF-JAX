"""Region-extended loopy BP planning with observation factors.

Extends loopy_bp.py to include observation factors at each timestep, computing
region beliefs for both dynamics and observation factors. With uniform y messages,
obs factors are uninformative and inference results match the original loopy BP.

Factor graph:

    p(theta)---theta (single variable, connected to all factors)
                |
    p(x0)--x0--[dyn0]--x1--[dyn1]--x2-- ... --[dyn_{T-1}]--x_T--goal
            |           |           |                        |
         [obs0]_k    [obs1]_k    [obs2]_k               [obsT]_k   (k=1..49)
            |           |           |                        |
          y0,k        y1,k        y2,k                    yT,k     (uniform)

theta cavity: total-minus-self (not prefix sums on equality chain).
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from .messages import EPSILON
from environments.minigrid import N_CELL_TYPES


# =============================================================================
# Reused from loopy_bp.py (self-contained copy with obs_to_x extensions)
# =============================================================================


def compute_reduced_per_t_from_kernels(dyn_kernels, cavity_theta):
    """
    Compute per-timestep reduced tensors from dynamics kernels.

    reduced[t](x_new, x_old, u) = Σ_θ κ_t(x_old, x_new, θ, u) · cavity[t](θ)

    Args:
        dyn_kernels: (T, n_states, n_states, n_static, n_actions) — κ_t(x_old, x_new, θ, u)
        cavity_theta: (T, n_static) per-timestep cavity beliefs on theta

    Returns:
        (T, n_states, n_states, n_actions) per-timestep reduced tensors
    """
    # tijkl = (T, x_old, x_new, θ, u), tk = (T, θ) → tjil = (T, x_new, x_old, u)
    return jnp.einsum("tijkl,tk->tjil", dyn_kernels, cavity_theta)


def forward_pass(reduced_per_t, q_x0, action_prior, obs_to_x, horizon):
    """
    Forward pass with obs->x message injection.

    mu_{x_t -> dyn_t} = fwd[t] * obs_to_x[t]

    Args:
        reduced_per_t: (T, n_states, n_states, n_actions)
        q_x0: (n_states,) initial state belief
        action_prior: (n_actions,) prior over actions
        obs_to_x: (T+1, n_states) messages from obs factors to x variables
        horizon: T

    Returns:
        fwd_msgs: (T+1, n_states) forward messages
    """
    n_states = q_x0.shape[0]
    fwd_msgs = jnp.zeros((horizon + 1, n_states))
    fwd_msgs = fwd_msgs.at[0].set(q_x0)

    def body_fn(t, fwd_msgs):
        # Incoming to dyn_t from x_t: fwd[t] * obs_to_x[t]
        x_msg = fwd_msgs[t] * obs_to_x[t]
        q_next = jnp.einsum("ijk,j,k->i", reduced_per_t[t], x_msg, action_prior)
        q_next = q_next / (q_next.sum() + EPSILON)
        return fwd_msgs.at[t + 1].set(q_next)

    return lax.fori_loop(0, horizon, body_fn, fwd_msgs)


def backward_pass(reduced_per_t, fwd_msgs, goal, action_prior, obs_to_x, horizon):
    """
    Backward pass with obs->x message injection, compute action marginals.

    mu_{x_{t+1} -> dyn_t} = bwd[t+1] * obs_to_x[t+1]

    Args:
        reduced_per_t: (T, n_states, n_states, n_actions)
        fwd_msgs: (T+1, n_states) forward messages
        goal: (n_states,) goal distribution
        action_prior: (n_actions,) prior over actions
        obs_to_x: (T+1, n_states) messages from obs factors to x variables
        horizon: T

    Returns:
        bwd_msgs: (T+1, n_states) backward messages
        q_u: (T, n_actions) action marginals
    """
    n_states = goal.shape[0]
    n_actions = action_prior.shape[0]
    bwd_msgs = jnp.zeros((horizon + 1, n_states))
    bwd_msgs = bwd_msgs.at[horizon].set(goal)
    q_u = jnp.zeros((horizon, n_actions))

    def body_fn(carry, t):
        bwd_msgs, q_u = carry
        # Incoming to dyn_t from x_{t+1}: bwd[t+1] * obs_to_x[t+1]
        bwd_t1 = bwd_msgs[t + 1] * obs_to_x[t + 1]

        # Action marginal: mu_{dyn_t -> u_t}
        # Incoming from x_t: fwd[t] * obs_to_x[t]
        fwd_t = fwd_msgs[t] * obs_to_x[t]
        msg_to_u = jnp.einsum("ijk,i,j->k", reduced_per_t[t], bwd_t1, fwd_t)
        q_u_t = msg_to_u * action_prior
        q_u_t = q_u_t / (q_u_t.sum() + EPSILON)
        q_u = q_u.at[t].set(q_u_t)

        # Backward message to x_t (from dyn_t, excluding x_t's own forward)
        bwd_t = jnp.einsum("ijk,i,k->j", reduced_per_t[t], bwd_t1, action_prior)
        bwd_t = bwd_t / (bwd_t.sum() + EPSILON)
        bwd_msgs = bwd_msgs.at[t].set(bwd_t)

        return (bwd_msgs, q_u), None

    (bwd_msgs, q_u), _ = lax.scan(
        body_fn, (bwd_msgs, q_u), jnp.arange(horizon - 1, -1, -1)
    )

    return bwd_msgs, q_u


def compute_dyn_to_theta_msgs(dyn_kernels, fwd_msgs, bwd_msgs, obs_to_x, action_prior, horizon):
    """
    Compute messages from each dynamics factor to theta, using obs-augmented x messages.

    msg(θ) = Σ_{x,x',u} κ_t(x, x', θ, u) · fwd(x) · bwd(x') · p(u)

    Args:
        dyn_kernels: (T, n_states, n_states, n_static, n_actions) — κ_t(x_old, x_new, θ, u)
        fwd_msgs: (T+1, n_states) forward messages
        bwd_msgs: (T+1, n_states) backward messages
        obs_to_x: (T+1, n_states) obs factor messages to x
        action_prior: (n_actions,) prior over actions
        horizon: T

    Returns:
        log_dyn_to_theta: (T, n_static) log-space messages
    """
    def compute_msg_t(t):
        fwd_t = fwd_msgs[t] * obs_to_x[t]
        bwd_t1 = bwd_msgs[t + 1] * obs_to_x[t + 1]

        # κ: (x_old, x_new, θ, u), fwd: (x_old,), bwd: (x_new,), prior: (u,)
        msg = jnp.einsum("ijkl,i,j,l->k", dyn_kernels[t], fwd_t, bwd_t1, action_prior)

        return jnp.log(msg + EPSILON)

    return jax.vmap(compute_msg_t)(jnp.arange(horizon))


# =============================================================================
# New functions for observation factors
# =============================================================================


def compute_obs_to_x_msgs(obs_kernels, cavity_theta_obs, horizon):
    """
    Compute messages from observation factors to x variables.

    Per FOV position k:
        μ_{obs_k→x}(x) = Σ_θ κ_{obs,t,k}(x,θ) · cavity_obs_t(θ)
    Aggregated (product over k, normalized):
        obs_to_x[t](x) ∝ Π_k μ_{obs_k→x}(x)

    Args:
        obs_kernels: (T+1, 7, 7, n_states, n_static) — compact obs kernels
        cavity_theta_obs: (T+1, n_static) cavity beliefs for obs factors
        horizon: T

    Returns:
        obs_to_x: (T+1, n_states) normalized messages
    """
    T_plus_1 = obs_kernels.shape[0]
    n_states = obs_kernels.shape[3]
    n_static = obs_kernels.shape[4]
    n_fov = obs_kernels.shape[1] * obs_kernels.shape[2]
    kernels_flat = obs_kernels.reshape(T_plus_1, n_fov, n_states, n_static)

    # Per-k message: Σ_θ κ(x,θ) · cavity(θ) → (T+1, n_fov, n_states)
    per_k_msg = jnp.einsum("tkis,ts->tki", kernels_flat, cavity_theta_obs)

    # Product over n_fov FOV positions in log-space → (T+1, n_states)
    log_per_k = jnp.log(per_k_msg + EPSILON)
    log_obs_to_x = log_per_k.sum(axis=1)

    # Normalize via softmax
    return jax.nn.softmax(log_obs_to_x, axis=1)


def compute_obs_to_theta_msgs(obs_kernels, fwd_msgs, bwd_msgs, obs_to_x, horizon):
    """
    Compute aggregated messages from observation factors to theta.

    Per FOV position k:
        μ_{obs_k→θ}(θ) = Σ_x κ_{obs,t,k}(x,θ) · μ_{x→obs_k}(x)
    Aggregated in log-space:
        log_obs_to_theta[t](θ) = Σ_k log μ_{obs_k→θ}(θ)

    Args:
        obs_kernels: (T+1, fov_w, fov_h, n_states, n_static) — compact obs kernels
        fwd_msgs: (T+1, n_states) forward messages
        bwd_msgs: (T+1, n_states) backward messages
        obs_to_x: (T+1, n_states) aggregated obs messages to x (unused in cavity)
        horizon: T

    Returns:
        log_obs_to_theta: (T+1, n_static) log-space messages
    """
    T_plus_1 = obs_kernels.shape[0]
    n_states = obs_kernels.shape[3]
    n_static = obs_kernels.shape[4]
    n_fov = obs_kernels.shape[1] * obs_kernels.shape[2]
    kernels_flat = obs_kernels.reshape(T_plus_1, n_fov, n_states, n_static)

    def compute_msg_t(t):
        # x→obs cavity: fwd * bwd (excludes obs_to_x)
        x_msg = fwd_msgs[t] * bwd_msgs[t]
        x_msg = x_msg / (x_msg.sum() + EPSILON)

        # Per-k: Σ_x κ(x,θ) · x_msg(x) → (n_fov, n_static)
        per_k_msg = jnp.einsum("kis,i->ks", kernels_flat[t], x_msg)

        # Sum of logs over k → (n_static,)
        return jnp.log(per_k_msg + EPSILON).sum(axis=0)

    return jax.vmap(compute_msg_t)(jnp.arange(horizon + 1))


def compute_theta_cavities_extended(log_prior, log_dyn_to_theta, log_obs_to_theta):
    """
    Compute cavity messages for theta via total-minus-self.

    total_log_theta = log p(theta) + sum_t log_dyn->theta[t] + sum_t log_obs->theta[t]
    cavity_dyn[t] = softmax(total_log_theta - log_dyn->theta[t])
    cavity_obs[t] = softmax(total_log_theta - log_obs->theta[t])

    Args:
        log_prior: (n_static,) log prior on theta
        log_dyn_to_theta: (T, n_static) per-timestep dynamics messages
        log_obs_to_theta: (T+1, n_static) per-timestep observation messages

    Returns:
        cavity_dyn: (T, n_static) normalized cavity beliefs for dynamics factors
        cavity_obs: (T+1, n_static) normalized cavity beliefs for obs factors
    """
    # Total log-belief on theta
    total = log_prior + log_dyn_to_theta.sum(axis=0) + log_obs_to_theta.sum(axis=0)

    # Cavity for each dynamics factor: exclude its own message
    log_cavity_dyn = total[None, :] - log_dyn_to_theta  # (T, n_static)
    cavity_dyn = jax.nn.softmax(log_cavity_dyn, axis=1)

    # Cavity for each obs factor: exclude its own message
    log_cavity_obs = total[None, :] - log_obs_to_theta  # (T+1, n_static)
    cavity_obs = jax.nn.softmax(log_cavity_obs, axis=1)

    return cavity_dyn, cavity_obs


# =============================================================================
# Region belief computation
# =============================================================================


def compute_dyn_region_beliefs(dyn_kernels, fwd_msgs, bwd_msgs, obs_to_x,
                               cavity_dyn, action_prior):
    """
    Compute region beliefs for dynamics factors.

    q_{t,dyn}(x_old, x_new, θ, u) ∝ κ_t(x_old, x_new, θ, u) · fwd(x_old) · bwd(x_new)
                                       · cavity(θ) · p(u)

    Args:
        dyn_kernels: (T, n_states, n_states, n_static, n_actions) — κ_t(x_old, x_new, θ, u)
        fwd_msgs: (T+1, n_states) forward messages
        bwd_msgs: (T+1, n_states) backward messages
        obs_to_x: (T+1, n_states) obs messages to x
        cavity_dyn: (T, n_static) cavity beliefs for dyn factors
        action_prior: (n_actions,) prior over actions

    Returns:
        region_beliefs: (T, n_states, n_states, n_static, n_actions) normalized region beliefs
    """
    T = cavity_dyn.shape[0]

    def compute_single_t(t):
        fwd_t = fwd_msgs[t] * obs_to_x[t]
        bwd_t1 = bwd_msgs[t + 1] * obs_to_x[t + 1]

        belief = (dyn_kernels[t]
                  * fwd_t[:, None, None, None]
                  * bwd_t1[None, :, None, None]
                  * cavity_dyn[t][None, None, :, None]
                  * action_prior[None, None, None, :])

        Z = belief.sum() + EPSILON
        return belief / Z

    return jax.vmap(compute_single_t)(jnp.arange(T))


def compute_dyn_channels(dyn_region_beliefs):
    """
    Compute dynamic channel distributions r_t(x_new | x_old, u) by marginalizing
    theta from the dynamics region beliefs.

    Args:
        dyn_region_beliefs: (T, n_states, n_states, n_static, n_actions) normalized region beliefs

    Returns:
        (T, n_states, n_states, n_actions) — r_t[x_old, x_new, u]
    """
    joint = dyn_region_beliefs.sum(axis=3)              # marginalize θ → (T, x_old, x_new, u)
    marginal = joint.sum(axis=2, keepdims=True)         # (T, x_old, 1, u)
    return joint / (marginal + EPSILON)


def compute_dyn_kernels(transition_idx, dyn_channels, n_states):
    """
    Compute per-timestep dynamics kernels:
        kernel_t = p(x_new | x_old, θ, u) / r_t(x_new | x_old, u)

    Args:
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        dyn_channels: (T, n_states, n_states, n_actions) — r_t[x_old, x_new, u]
        n_states: int

    Returns:
        (T, n_states, n_states, n_static, n_actions) — kernel per dynamics factor
    """
    # p(x_new | x_old, θ, u) as one-hot: (x_old, x_new, θ, u)
    p_xnew = jax.nn.one_hot(transition_idx, n_states)   # (x_old, θ, u, x_new)
    p_xnew = jnp.transpose(p_xnew, (0, 3, 1, 2))       # (x_old, x_new, θ, u)

    # Broadcast channel over θ: (T, x_old, x_new, 1, u)
    r = dyn_channels[:, :, :, None, :]

    # kernel = p / r
    return p_xnew[None] / (r + EPSILON)  # (T, x_old, x_new, θ, u)


def compute_obs_channels_from_beliefs(obs_region_beliefs):
    """Compute r(y|x,θ) from obs region beliefs q(y,x,θ).

    r(y|x,θ) = q(y,x,θ) / Σ_y' q(y',x,θ)

    Args:
        obs_region_beliefs: (T+1, n_fov, n_cell_types, n_states, n_static)

    Returns:
        (T+1, n_fov, n_cell_types, n_states, n_static) — r_t(y|x,θ) per FOV
    """
    marginal = obs_region_beliefs.sum(axis=2, keepdims=True)  # (T+1, n_fov, 1, n_states, n_static)
    return obs_region_beliefs / (marginal + EPSILON)


def compute_obs_kernels(obs_idx, obs_channels):
    """
    Compute per-timestep observation kernels (compact form):
        obs_kernel_{t,k}(x, θ) = r_{t,k}(y* | x, θ)  where y* = obs_idx[k, x, θ]

    Args:
        obs_idx: (fov_w, fov_h, n_states, n_static) -> cell_type index
        obs_channels: (T+1, n_fov, n_cell_types, n_states, n_static) — r_{t,k}(y | x, θ)

    Returns:
        (T+1, fov_w, fov_h, n_states, n_static) — kernel value at deterministic y
    """
    fov_w, fov_h = obs_idx.shape[0], obs_idx.shape[1]
    n_fov = fov_w * fov_h
    obs_flat = obs_idx.reshape(n_fov, obs_idx.shape[2], obs_idx.shape[3])
    # Index into cell_type axis: (1, n_fov, 1, n_states, n_static)
    y_idx = obs_flat[None, :, None, :, :]
    # Gather r at deterministic y → (T+1, n_fov, 1, n_states, n_static)
    kernels = jnp.take_along_axis(obs_channels, y_idx, axis=2)
    kernels = kernels.squeeze(axis=2)          # (T+1, n_fov, n_states, n_static)
    T_plus_1, _, n_states, n_static = kernels.shape
    return kernels.reshape(T_plus_1, fov_w, fov_h, n_states, n_static)


def compute_obs_region_beliefs(obs_kernels, fwd_msgs, bwd_msgs, obs_to_x, cavity_obs):
    """
    Compute region beliefs for observation factors (per FOV position).

    q_{t,obs,k}(y, x, θ) ∝ κ_{t,k}(x, θ) · μ_y(y) · μ_{x→obs_k}(x) · cavity_obs_t(θ)

    With μ_y(y) = 1 (uniform, no observation), the belief is uniform over y.

    Args:
        obs_kernels: (T+1, fov_w, fov_h, n_states, n_static) — compact obs kernels
        fwd_msgs: (T+1, n_states) forward messages
        bwd_msgs: (T+1, n_states) backward messages
        obs_to_x: (T+1, n_states) aggregated obs messages to x
        cavity_obs: (T+1, n_static) cavity beliefs for obs factors

    Returns:
        region_beliefs: (T+1, n_fov, n_cell_types, n_states, n_static) normalized region beliefs
    """
    T_plus_1 = cavity_obs.shape[0]
    n_states = obs_kernels.shape[3]
    n_static = obs_kernels.shape[4]
    n_fov = obs_kernels.shape[1] * obs_kernels.shape[2]
    kernels_flat = obs_kernels.reshape(T_plus_1, n_fov, n_states, n_static)

    def compute_single_t(t):
        x_belief = fwd_msgs[t] * bwd_msgs[t]
        x_belief = x_belief / (x_belief.sum() + EPSILON)

        # (n_fov, n_states, n_static) — belief over (x, θ) per FOV position
        belief_xtheta = (kernels_flat[t]
                         * x_belief[None, :, None]
                         * cavity_obs[t][None, None, :])

        # Broadcast uniform μ_y over y → (n_fov, n_cell_types, n_states, n_static)
        belief = jnp.broadcast_to(
            belief_xtheta[:, None, :, :],
            (n_fov, N_CELL_TYPES, n_states, n_static)
        )

        Z = belief.sum() + EPSILON
        return belief / Z

    return jax.vmap(compute_single_t)(jnp.arange(T_plus_1))


# =============================================================================
# Main planning function
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def region_extended_loopy_bp_planning_indexed(
    q_current_state,    # (n_states,)
    q_static_state,     # (n_static,) prior on theta
    transition_idx,     # (n_states, n_static, n_actions)
    obs_idx,            # (7, 7, n_states, n_static) -> cell_type
    goal,               # (n_states,)
    horizon,            # int (static)
    n_iterations,       # int (static)
) -> jnp.ndarray:
    """
    Plan actions via region-extended loopy BP with observation factors.

    Each outer iteration:
      1. Compute theta cavities (dyn + obs) from previous iteration's messages
      2. compute_reduced_per_t using dyn-cavities
      3. Compute obs->x messages (uniform for now)
      4. Forward pass (with obs->x)
      5. Backward pass (with obs->x) + action marginals
      6. Compute dyn->theta messages (using obs-augmented x messages)
      7. Compute obs->theta messages (uniform for now)
      8. Compute region beliefs (dyn + obs)

    With uniform observation messages, results are identical to loopy_bp_planning_indexed.

    Args:
        q_current_state: (n_states,) current belief over dynamic state
        q_static_state: (n_static,) prior belief over static configuration theta
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        obs_idx: (7, 7, n_states, n_static) -> cell_type index
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: number of loopy BP iterations (static for JIT)

    Returns:
        action_dist: (n_actions,) distribution over first action
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_idx.shape[2]
    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])

    log_prior_theta = jnp.log(q_static_state + EPSILON)

    # Initialize messages: all zeros in log-space = uniform
    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    log_obs_to_theta = jnp.zeros((horizon + 1, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    dyn_channels_init = jnp.zeros((horizon, n_states, n_states, n_actions))

    # Initialize obs channels: one-hot from deterministic obs_idx
    fov_w, fov_h = obs_idx.shape[0], obs_idx.shape[1]
    n_fov = fov_w * fov_h
    obs_flat = obs_idx.reshape(n_fov, n_states, n_static)
    r_init = jax.nn.one_hot(obs_flat, N_CELL_TYPES)          # (n_fov, n_states, n_static, N_CELL_TYPES)
    r_init = jnp.transpose(r_init, (0, 3, 1, 2))             # (n_fov, N_CELL_TYPES, n_states, n_static)
    obs_channels_init = jnp.broadcast_to(
        r_init[None], (horizon + 1, n_fov, N_CELL_TYPES, n_states, n_static)
    )
    p_xnew = jax.nn.one_hot(transition_idx, n_states)       # (x_old, θ, u, x_new)
    p_xnew = jnp.transpose(p_xnew, (0, 3, 1, 2))           # (x_old, x_new, θ, u)
    dyn_kernels_init = jnp.broadcast_to(
        p_xnew[None], (horizon, n_states, n_states, n_static, n_actions)
    )
    obs_kernels_init = jnp.ones((horizon + 1, fov_w, fov_h, n_states, n_static))

    def body_fn(_, carry):
        log_dyn_to_theta, log_obs_to_theta, _, _, _, dyn_kernels, obs_kernels = carry

        # Step 1: theta cavities (total-minus-self)
        cavity_dyn, cavity_obs = compute_theta_cavities_extended(
            log_prior_theta, log_dyn_to_theta, log_obs_to_theta
        )

        # Step 2: Per-timestep reduced tensors from dyn-kernels
        reduced_per_t = compute_reduced_per_t_from_kernels(
            dyn_kernels, cavity_dyn
        )

        # Step 3: obs->x messages from obs kernels
        obs_to_x = compute_obs_to_x_msgs(obs_kernels, cavity_obs, horizon)

        # Step 4: Forward pass (with obs->x)
        fwd_msgs = forward_pass(
            reduced_per_t, q_current_state, action_prior, obs_to_x, horizon
        )

        # Step 5: Backward pass (with obs->x) + action marginals
        bwd_msgs, q_u = backward_pass(
            reduced_per_t, fwd_msgs, goal, action_prior, obs_to_x, horizon
        )

        # Step 6: dyn->theta messages
        new_log_dyn_to_theta = compute_dyn_to_theta_msgs(
            dyn_kernels, fwd_msgs, bwd_msgs, obs_to_x, action_prior, horizon
        )

        # Step 7: obs->theta messages from obs kernels
        new_log_obs_to_theta = compute_obs_to_theta_msgs(
            obs_kernels, fwd_msgs, bwd_msgs, obs_to_x, horizon
        )

        # Step 8: Region beliefs
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

        return new_log_dyn_to_theta, new_log_obs_to_theta, q_u, dyn_channels, obs_channels, dyn_kernels, obs_kernels

    log_dyn_to_theta, log_obs_to_theta, q_u, dyn_channels, obs_channels, dyn_kernels, obs_kernels = lax.fori_loop(
        0, n_iterations, body_fn,
        (log_dyn_to_theta, log_obs_to_theta, q_u_init, dyn_channels_init, obs_channels_init, dyn_kernels_init, obs_kernels_init)
    )

    return q_u[0], dyn_channels, obs_channels, dyn_kernels, obs_kernels
