"""Active Inference planning via message passing with dynamics and observation channels.

Implements the full AIF message-passing scheme:
- Per-timestep dynamics channel r_x[t]: captures certainty reward -H[x|x',u]
- Per-timestep observation channel r_y[t]: captures ambiguity penalty +H[y|x,θ]
- Explicit θ variable updated each iteration
- Modified factor kernels: dyn uses T/r_x[t], obs uses p·r_y[t]

With n_iterations=1 and uniform channels, reproduces standard BP exactly.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from .messages import EPSILON
from .planning import forward_pass, marginalize_static_indexed

N_CELL_TYPES = 11
N_FOV = 49  # 7 x 7


@partial(jax.jit, static_argnums=(5, 6))
def aif_planning_indexed(
    q_current_state: jnp.ndarray,
    q_static_state: jnp.ndarray,
    transition_idx: jnp.ndarray,
    observation_idx: jnp.ndarray,
    goal: jnp.ndarray,
    horizon: int,
    n_iterations: int = 1,
) -> jnp.ndarray:
    """
    AIF planning with dynamics and observation channels.

    Args:
        q_current_state: (n_states,) current belief over state
        q_static_state: (n_static,) belief over static configuration (prior p(θ))
        transition_idx: (n_states, n_static, n_actions) -> new_state index
        observation_idx: (7, 7, n_states, n_static) -> cell_type index
        goal: (n_states,) goal distribution over final state
        horizon: planning horizon T (static for JIT)
        n_iterations: number of AIF iterations (static for JIT)

    Returns:
        action_dist: (n_actions,) distribution over first action
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_idx.shape[2]

    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])

    # Initialize beliefs
    q_u = jnp.tile(action_prior, (horizon, 1))
    q_state = jnp.concatenate([
        q_current_state[None, :],
        jnp.ones((horizon, n_states)) / n_states
    ], axis=0)
    q_theta = q_static_state

    # Initialize per-timestep channels as uniform (standard BP on first iteration)
    r_x = jnp.ones((horizon, n_states, n_states, n_actions)) / n_states
    r_y = jnp.ones((horizon + 1, N_FOV, N_CELL_TYPES, n_states, n_static)) / N_CELL_TYPES

    # Initialize per-timestep factor→θ messages to zero (cavities = prior)
    log_dyn_msgs_per_t = jnp.zeros((horizon, n_static))
    log_obs_msgs_per_t = jnp.zeros((horizon, n_static))

    # Flatten obs_idx for efficient computation
    obs_idx_flat = observation_idx.reshape(N_FOV, n_states, n_static)

    prior_theta = q_static_state
    log_prior = jnp.log(prior_theta + EPSILON)

    def body_fn(_, carry):
        q_state, q_u, q_theta, r_x, r_y, log_dyn_msgs_per_t, log_obs_msgs_per_t = carry

        # Step 1: Compute per-timestep cavity messages for θ
        cavities_dyn, cavities_obs = compute_cavities(
            log_prior, log_dyn_msgs_per_t, log_obs_msgs_per_t
        )

        # Step 2: Per-timestep reduced tensors using cavity_dyn[t]
        reduced_per_t = jax.vmap(
            lambda q: marginalize_static_indexed(transition_idx, q, n_states)
        )(cavities_dyn)  # (T, n_states, n_states, n_actions)

        # Step 3: Per-timestep modified kernel K_mod[t] = T_t / r_x[t]
        K_mod = compute_modified_kernel(reduced_per_t, r_x)

        # Step 4: θ for obs: uniform at t=0 (no obs factor), cavities for t=1..T
        uniform_theta = jnp.ones(n_static) / n_static
        q_theta_for_obs = jnp.concatenate(
            [uniform_theta[None, :], cavities_obs], axis=0
        )  # (T+1, n_static)

        # Step 5: Per-timestep obs messages to x using per-timestep θ cavities
        log_obs_msgs_to_x = compute_all_obs_msgs_to_x(obs_idx_flat, r_y, q_theta_for_obs)
        maxes = log_obs_msgs_to_x.max(axis=1, keepdims=True)
        log_obs_msgs_to_x = log_obs_msgs_to_x - maxes
        # No observation factor at x₀ (degree 2: only p(x₀) and dyn_1)
        log_obs_msgs_to_x = log_obs_msgs_to_x.at[0].set(0.0)

        # Step 6: Forward pass with per-timestep K_mod and obs messages
        q_state_new = aif_forward_pass(K_mod, q_state, action_prior, log_obs_msgs_to_x, horizon)

        # Step 7: Backward pass with per-timestep K_mod and obs messages
        q_u_new, backward_msgs = aif_backward_pass_with_messages(
            K_mod, q_state_new, goal, action_prior, log_obs_msgs_to_x, horizon
        )

        # Step 8: Per-timestep obs→θ messages (uses backward_msgs)
        log_obs_msgs_per_t_new = compute_obs_msgs_to_theta_per_t(
            obs_idx_flat, r_y, q_state_new, backward_msgs, horizon
        )

        # Step 9: Per-timestep dyn→θ messages
        log_dyn_msgs_per_t_new = compute_theta_messages_from_dynamics(
            transition_idx, q_state_new, backward_msgs,
            action_prior, r_x, log_obs_msgs_to_x, horizon
        )

        # Step 10: Update q_theta from per-timestep messages
        q_theta_new = update_q_theta(
            prior_theta, log_dyn_msgs_per_t_new, log_obs_msgs_per_t_new
        )

        # Step 11: Update per-timestep dynamics channel r_x
        r_x_new = channel_update_dynamics(
            K_mod, q_state_new, q_u_new, backward_msgs, log_obs_msgs_to_x, horizon
        )

        # Step 12: Update per-timestep observation channel r_y
        r_y_new = channel_update_obs(obs_idx_flat, n_states, n_static, horizon)

        return (q_state_new, q_u_new, q_theta_new, r_x_new, r_y_new,
                log_dyn_msgs_per_t_new, log_obs_msgs_per_t_new)

    q_state, q_u, q_theta, r_x, r_y, _, _ = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_state, q_u, q_theta, r_x, r_y, log_dyn_msgs_per_t, log_obs_msgs_per_t)
    )

    return q_u[0]


def compute_cavities(
    log_prior: jnp.ndarray,
    log_dyn_msgs_per_t: jnp.ndarray,
    log_obs_msgs_per_t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute per-timestep cavity messages for θ.

    cavity_dyn[t] = q_theta excluding dyn_t's own message to θ
    cavity_obs[t] = q_theta excluding obs_t's own message to θ

    Args:
        log_prior: (n_static,) log p(θ)
        log_dyn_msgs_per_t: (T, n_static) per-timestep dyn→θ log messages
        log_obs_msgs_per_t: (T, n_static) per-timestep obs→θ log messages

    Returns:
        cavities_dyn: (T, n_static) cavity beliefs for dynamics factors
        cavities_obs: (T, n_static) cavity beliefs for observation factors
    """
    log_q_unnorm = (
        log_prior
        + log_dyn_msgs_per_t.sum(axis=0)
        + log_obs_msgs_per_t.sum(axis=0)
    )
    cavities_dyn = jax.vmap(
        lambda log_msg: jax.nn.softmax(log_q_unnorm - log_msg)
    )(log_dyn_msgs_per_t)
    cavities_obs = jax.vmap(
        lambda log_msg: jax.nn.softmax(log_q_unnorm - log_msg)
    )(log_obs_msgs_per_t)
    return cavities_dyn, cavities_obs


def compute_modified_kernel(
    reduced_per_t: jnp.ndarray,
    r_x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute per-timestep modified transition kernel K_mod[t] = T_t / r_x[t].

    Args:
        reduced_per_t: (T, n_states, n_states, n_actions) per-timestep marginalized transitions
        r_x: (T, n_states, n_states, n_actions) per-timestep dynamics channels

    Returns:
        K_mod: (T, n_states, n_states, n_actions) per-timestep modified kernels
    """
    return reduced_per_t / (r_x + EPSILON)


def compute_all_obs_msgs_to_x(
    obs_idx_flat: jnp.ndarray,
    r_y: jnp.ndarray,
    q_theta_per_t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute obs→x messages for all timesteps using per-timestep r_y and q_theta.

    Each timestep uses its own cavity θ belief (μ_{θ→obs_t}).

    Args:
        obs_idx_flat: (49, n_states, n_static) -> cell_type index
        r_y: (T+1, 49, 11, n_states, n_static) per-timestep obs channels
        q_theta_per_t: (T+1, n_static) per-timestep θ beliefs (cavities)

    Returns:
        log_obs_msgs_to_x: (T+1, n_states) per-timestep log obs messages to x
    """
    n_states = obs_idx_flat.shape[1]
    n_static = obs_idx_flat.shape[2]

    fov_idx = jnp.arange(N_FOV)[:, None, None]
    x_idx = jnp.arange(n_states)[None, :, None]
    theta_idx = jnp.arange(n_static)[None, None, :]

    fov_bc = jnp.broadcast_to(fov_idx, obs_idx_flat.shape)
    x_bc = jnp.broadcast_to(x_idx, obs_idx_flat.shape)
    theta_bc = jnp.broadcast_to(theta_idx, obs_idx_flat.shape)

    def single_msg(r_y_t, q_theta_t):
        r_y_at_obs = r_y_t[fov_bc, obs_idx_flat, x_bc, theta_bc]
        weighted = r_y_at_obs * q_theta_t[None, None, :]
        msg_per_fov = weighted.sum(axis=2)
        return jnp.log(msg_per_fov + EPSILON).sum(axis=0)

    return jax.vmap(single_msg)(r_y, q_theta_per_t)


def compute_obs_msgs_to_theta_per_t(
    obs_idx_flat: jnp.ndarray,
    r_y: jnp.ndarray,
    q_states: jnp.ndarray,
    backward_msgs: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    """
    Compute per-timestep obs→θ messages for t=1..T using per-timestep r_y.

    The correct variable-to-factor message μ_{x_t→obs_t} includes BOTH
    forward and backward messages:
      μ_{x_t→obs_t}(x) ∝ q_state[t](x) · backward_msgs[t](x)
    This ensures θ is updated with goal-directed information.

    Each obs factor sends:
      log_msg_t(θ) = Σ_fov log( Σ_x r_y_at_obs[t+1,fov,x,θ] · μ_{x→obs}(x) )

    Args:
        obs_idx_flat: (49, n_states, n_static) -> cell_type index
        r_y: (T+1, 49, 11, n_states, n_static) per-timestep obs channels
        q_states: (T+1, n_states) state beliefs (index 0 = initial, 1..T = planning)
        backward_msgs: (T+1, n_states) backward messages
        horizon: T

    Returns:
        log_obs_msgs_per_t: (T, n_static) per-timestep log obs messages to θ
    """
    n_states = obs_idx_flat.shape[1]
    n_static = obs_idx_flat.shape[2]

    # Precompute r_y_at_obs for all timesteps
    fov_idx = jnp.arange(N_FOV)[:, None, None]
    x_idx = jnp.arange(n_states)[None, :, None]
    theta_idx = jnp.arange(n_static)[None, None, :]

    fov_bc = jnp.broadcast_to(fov_idx, obs_idx_flat.shape)
    x_bc = jnp.broadcast_to(x_idx, obs_idx_flat.shape)
    theta_bc = jnp.broadcast_to(theta_idx, obs_idx_flat.shape)

    def gather_single(r_y_t):
        return r_y_t[fov_bc, obs_idx_flat, x_bc, theta_bc]

    all_r_y_at_obs = jax.vmap(gather_single)(r_y)  # (T+1, 49, n_states, n_static)

    def body_fn(_, t):
        # μ_{x→obs_t} includes both forward and backward messages
        q_x_t = q_states[t + 1] * backward_msgs[t + 1]
        q_x_t = q_x_t / (q_x_t.sum() + EPSILON)
        r_y_at_obs = all_r_y_at_obs[t + 1]  # (49, n_states, n_static)
        weighted = r_y_at_obs * q_x_t[None, :, None]
        msg_per_fov = weighted.sum(axis=1)
        log_msg_t = jnp.log(msg_per_fov + EPSILON).sum(axis=0)
        return None, log_msg_t

    _, log_obs_msgs_per_t = lax.scan(body_fn, None, jnp.arange(horizon))
    return log_obs_msgs_per_t


def aif_forward_pass(
    K_mod: jnp.ndarray,
    q_state: jnp.ndarray,
    action_prior: jnp.ndarray,
    log_obs_msgs_to_x: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    """
    Forward pass with per-timestep modified kernels and observation messages.

    At each step t:
      q_state[t+1] = Σ K_mod[t] · (q_state[t] · exp(obs_msg[t])) · p(u)

    The BP message μ_{u→dyn_t} = p(u) since u_t only connects to p(u_t) and dyn_t.

    Args:
        K_mod: (T, n_states, n_states, n_actions) per-timestep modified kernels
        q_state: (T+1, n_states) state beliefs
        action_prior: (n_actions,) prior over actions
        log_obs_msgs_to_x: (T+1, n_states) per-timestep log obs messages
        horizon: T

    Returns:
        q_state: (T+1, n_states) updated state beliefs
    """
    def body_fn(t, q_state):
        obs_factor = jnp.exp(log_obs_msgs_to_x[t])
        q_aug = q_state[t] * obs_factor
        q_aug = q_aug / (q_aug.sum() + EPSILON)

        q_next = jnp.einsum("ijk,j,k->i", K_mod[t], q_aug, action_prior)
        q_next = q_next / (q_next.sum() + EPSILON)
        return q_state.at[t + 1].set(q_next)

    return lax.fori_loop(0, horizon, body_fn, q_state)


def aif_backward_pass_with_messages(
    K_mod: jnp.ndarray,
    q_state: jnp.ndarray,
    goal: jnp.ndarray,
    action_prior: jnp.ndarray,
    log_obs_msgs_to_x: jnp.ndarray,
    horizon: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Backward pass with per-timestep modified kernels and observation messages.

    At each step t (going backward):
      bwd_aug = backward_msg · exp(obs_msg[t+1])
      fwd_aug = q_state[t] · exp(obs_msg[t])
      q_u[t] ∝ Σ K_mod[t] · bwd_aug · fwd_aug · action_prior

    Args:
        K_mod: (T, n_states, n_states, n_actions) per-timestep modified kernels
        q_state: (T+1, n_states) beliefs from forward pass
        goal: (n_states,) goal distribution
        action_prior: (n_actions,) prior over actions
        log_obs_msgs_to_x: (T+1, n_states) per-timestep log obs messages
        horizon: T

    Returns:
        q_u: (T, n_actions) action beliefs
        backward_msgs: (T+1, n_states) backward messages (un-augmented)
    """
    n_states = q_state.shape[1]
    n_actions = action_prior.shape[0]

    def body_fn(carry, t):
        backward_msg, q_u, backward_msgs = carry

        # Store un-augmented backward message
        backward_msgs = backward_msgs.at[t + 1].set(backward_msg)

        # Augment backward message with obs at state t+1
        obs_factor_tp1 = jnp.exp(log_obs_msgs_to_x[t + 1])
        bwd_aug = backward_msg * obs_factor_tp1
        bwd_aug = bwd_aug / (bwd_aug.sum() + EPSILON)

        # Augment forward message with obs at state t
        obs_factor_t = jnp.exp(log_obs_msgs_to_x[t])
        fwd_aug = q_state[t] * obs_factor_t
        fwd_aug = fwd_aug / (fwd_aug.sum() + EPSILON)

        # Action belief using K_mod at dynamics factor t
        msg_to_u = jnp.einsum("ijk,i,j->k", K_mod[t], bwd_aug, fwd_aug)
        q_u_t = msg_to_u * action_prior
        q_u_t = q_u_t / (q_u_t.sum() + EPSILON)
        q_u = q_u.at[t].set(q_u_t)

        # Propagate backward using K_mod at dynamics factor t
        backward_msg = jnp.einsum("ijk,i,k->j", K_mod[t], bwd_aug, action_prior)
        backward_msg = backward_msg / (backward_msg.sum() + EPSILON)

        return (backward_msg, q_u, backward_msgs), None

    q_u_init = jnp.zeros((horizon, n_actions))
    backward_msgs_init = jnp.zeros((horizon + 1, n_states))
    backward_msgs_init = backward_msgs_init.at[horizon].set(goal)

    (final_backward_msg, q_u, backward_msgs), _ = lax.scan(
        body_fn,
        (goal, q_u_init, backward_msgs_init),
        jnp.arange(horizon - 1, -1, -1),
    )

    backward_msgs = backward_msgs.at[0].set(final_backward_msg)
    return q_u, backward_msgs


def compute_theta_messages_from_dynamics(
    transition_idx: jnp.ndarray,
    q_state: jnp.ndarray,
    backward_msgs: jnp.ndarray,
    action_prior: jnp.ndarray,
    r_x: jnp.ndarray,
    log_obs_msgs_to_x: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    """
    Compute per-timestep dynamics messages to θ.

    For each t and θ:
      μ_{dyn_t→θ}(θ) = Σ_{x,u} fwd_aug[t,x] · p(u) ·
          bwd_aug[transition_idx[x,θ,u]] / r_x[t, transition_idx[x,θ,u], x, u]

    Args:
        transition_idx: (n_states, n_static, n_actions) -> next_state
        q_state: (T+1, n_states) forward beliefs
        backward_msgs: (T+1, n_states) backward messages
        action_prior: (n_actions,) prior over actions
        r_x: (T, n_states, n_states, n_actions) per-timestep dynamics channels
        log_obs_msgs_to_x: (T+1, n_states) per-timestep log obs messages
        horizon: T

    Returns:
        log_dyn_msgs_per_t: (T, n_static) per-timestep log messages to θ
    """
    n_states, n_static, n_actions = transition_idx.shape

    # Precompute index arrays for gathering r_x
    old_idx = jnp.arange(n_states)[:, None, None]  # (n_states, 1, 1)
    act_idx = jnp.arange(n_actions)[None, None, :]  # (1, 1, n_actions)
    old_idx_bc = jnp.broadcast_to(old_idx, transition_idx.shape)
    act_idx_bc = jnp.broadcast_to(act_idx, transition_idx.shape)

    def body_fn(_, t):
        # Augmented forward: μ_{x_t→dyn_t} = q_state[t] · exp(obs_msg[t])
        obs_factor_t = jnp.exp(log_obs_msgs_to_x[t])
        fwd_aug = q_state[t] * obs_factor_t
        fwd_aug = fwd_aug / (fwd_aug.sum() + EPSILON)

        # Augmented backward: μ_{x_{t+1}→dyn_t} = backward_msgs[t+1] · exp(obs_msg[t+1])
        obs_factor_tp1 = jnp.exp(log_obs_msgs_to_x[t + 1])
        bwd_aug = backward_msgs[t + 1] * obs_factor_tp1
        bwd_aug = bwd_aug / (bwd_aug.sum() + EPSILON)

        # Gather backward at next states: bwd_aug[transition_idx[x,θ,u]]
        bwd_gathered = bwd_aug[transition_idx]  # (n_states, n_static, n_actions)

        # Gather r_x[t] at (next_state, old_state, action)
        r_x_t = r_x[t]
        r_x_gathered = r_x_t[transition_idx, old_idx_bc, act_idx_bc]

        # Compute: Σ_{x,u} [bwd(x*) / r_x(x*|x,u)] · fwd(x) · p(u)
        ratio = bwd_gathered / (r_x_gathered + EPSILON)
        weighted = ratio * fwd_aug[:, None, None] * action_prior[None, None, :]
        msg_t = weighted.sum(axis=(0, 2))  # (n_static,)

        return None, jnp.log(msg_t + EPSILON)

    _, log_dyn_msgs_per_t = lax.scan(body_fn, None, jnp.arange(horizon))
    return log_dyn_msgs_per_t


def update_q_theta(
    prior_theta: jnp.ndarray,
    log_dyn_msgs_per_t: jnp.ndarray,
    log_obs_msgs_per_t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Update θ belief from prior and per-timestep messages.

    q_theta(θ) ∝ p(θ) · Π_t exp(log_dyn_msgs[t](θ)) · Π_t exp(log_obs_msgs[t](θ))

    Args:
        prior_theta: (n_static,) prior p(θ)
        log_dyn_msgs_per_t: (T, n_static) per-timestep log dynamics messages
        log_obs_msgs_per_t: (T, n_static) per-timestep log obs messages

    Returns:
        q_theta: (n_static,) updated θ belief
    """
    log_q = (
        jnp.log(prior_theta + EPSILON)
        + log_dyn_msgs_per_t.sum(axis=0)
        + log_obs_msgs_per_t.sum(axis=0)
    )
    return jax.nn.softmax(log_q)


def channel_update_dynamics(
    K_mod: jnp.ndarray,
    q_state: jnp.ndarray,
    q_u: jnp.ndarray,
    backward_msgs: jnp.ndarray,
    log_obs_msgs_to_x: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    """
    Update per-timestep dynamics channel r_x[t] from factor beliefs.

    For each time step t independently, computes the dynamics factor belief:
      q_trip_t(x_t, x_{t-1}, u) ∝ K_mod[t] · fwd_aug[t] · bwd_aug[t+1] · q_u[t]
    Then:
      r_x[t] = q_trip_t / q_pair_t

    Args:
        K_mod: (T, n_states, n_states, n_actions) per-timestep modified kernels
        q_state: (T+1, n_states) forward beliefs
        q_u: (T, n_actions) action beliefs
        backward_msgs: (T+1, n_states) backward messages
        log_obs_msgs_to_x: (T+1, n_states) per-timestep log obs messages
        horizon: T

    Returns:
        r_x: (T, n_states, n_states, n_actions) per-timestep dynamics channels,
             each normalized over x_t (axis 1) for each (x_{t-1}, u)
    """
    n_states = K_mod.shape[1]

    def compute_r_x_at_t(carry, t):
        # Augmented forward with obs at state t
        obs_factor_t = jnp.exp(log_obs_msgs_to_x[t])
        fwd_aug = q_state[t] * obs_factor_t
        fwd_aug = fwd_aug / (fwd_aug.sum() + EPSILON)

        # Augmented backward with obs at state t+1
        obs_factor_tp1 = jnp.exp(log_obs_msgs_to_x[t + 1])
        bwd_aug = backward_msgs[t + 1] * obs_factor_tp1
        bwd_aug = bwd_aug / (bwd_aug.sum() + EPSILON)

        # q_trip(x_t, x_{t-1}, u) ∝ K_mod[t] · fwd_aug · bwd_aug · q_u[t]
        q_trip = (
            K_mod[t]
            * fwd_aug[None, :, None]
            * bwd_aug[:, None, None]
            * q_u[t][None, None, :]
        )

        # Normalize per-timestep: r_x[t] = q_trip_t / q_pair_t
        pair_sum = q_trip.sum(axis=0, keepdims=True)
        uniform = jnp.ones_like(q_trip) / n_states
        has_mass = pair_sum > 1e-8
        r_x_t = jnp.where(has_mass, q_trip / pair_sum, uniform)

        return None, r_x_t

    _, r_x = lax.scan(compute_r_x_at_t, None, jnp.arange(horizon))
    return r_x


def channel_update_obs(
    obs_idx_flat: jnp.ndarray,
    n_states: int,
    n_static: int,
    horizon: int,
) -> jnp.ndarray:
    """
    Update per-timestep observation channel r_y[t] from factor beliefs.

    For deterministic observations p(y|x,θ) = δ(y = obs_idx[x,θ]),
    each per-timestep channel converges to the same delta:
      r_y[t](y | x, θ) = δ(y = obs_idx[fov, x, θ])

    Args:
        obs_idx_flat: (49, n_states, n_static) -> cell_type index
        n_states: number of dynamic states
        n_static: number of static configurations
        horizon: T (to determine number of obs factors = T+1)

    Returns:
        r_y: (T+1, 49, 11, n_states, n_static) per-timestep obs channels
    """
    r_y_single = jnp.zeros((N_FOV, N_CELL_TYPES, n_states, n_static))

    fov_idx = jnp.arange(N_FOV)[:, None, None]
    x_idx = jnp.arange(n_states)[None, :, None]
    theta_idx = jnp.arange(n_static)[None, None, :]

    r_y_single = r_y_single.at[
        jnp.broadcast_to(fov_idx, obs_idx_flat.shape),
        obs_idx_flat,
        jnp.broadcast_to(x_idx, obs_idx_flat.shape),
        jnp.broadcast_to(theta_idx, obs_idx_flat.shape),
    ].set(1.0)

    # Tile to per-timestep (all identical for deterministic obs)
    return jnp.tile(r_y_single[None, :, :, :, :], (horizon + 1, 1, 1, 1, 1))
