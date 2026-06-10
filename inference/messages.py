"""Message passing primitives for discrete factor graphs."""

import jax
import jax.numpy as jnp
from jax import lax, nn
from jax.scipy.special import logsumexp

EPSILON = 1e-10
LOG_ZERO = -1e12


def safe_log(x):
    """Log that maps 0 → LOG_ZERO instead of -inf."""
    return jnp.where(x > 0, jnp.log(jnp.maximum(x, 1e-30)), LOG_ZERO)


def safe_log_div(log_num, log_den):
    """log(num/den) where 0/0 and nonzero/0 → LOG_ZERO (0 in probability space)."""
    valid = (log_num > LOG_ZERO / 2) & (log_den > LOG_ZERO / 2)
    return jnp.where(valid, log_num - log_den, LOG_ZERO)


# =============================================================================
# Tensor-based message passing
# =============================================================================


def forward_message_2d(tensor: jnp.ndarray, q_in: jnp.ndarray) -> jnp.ndarray:
    """
    Forward message through a 2D transition tensor.
    
    tensor: (n_out, n_in)
    q_in: (n_in,)
    returns: (n_out,) normalized
    """
    msg = jnp.einsum("ij,j->i", tensor, q_in)
    return msg / (msg.sum() + EPSILON)


def forward_message_3d(
    tensor: jnp.ndarray, q_in1: jnp.ndarray, q_in2: jnp.ndarray
) -> jnp.ndarray:
    """
    Forward message through a 3D transition tensor.
    
    tensor: (n_out, n_in1, n_in2)
    q_in1: (n_in1,), q_in2: (n_in2,)
    returns: (n_out,) normalized
    """
    msg = jnp.einsum("ijk,j,k->i", tensor, q_in1, q_in2)
    return msg / (msg.sum() + EPSILON)


def forward_message_4d(
    tensor: jnp.ndarray,
    q_in1: jnp.ndarray,
    q_in2: jnp.ndarray,
    q_in3: jnp.ndarray,
) -> jnp.ndarray:
    """
    Forward message through a 4D transition tensor.
    
    tensor: (n_out, n_in1, n_in2, n_in3)
    q_in1: (n_in1,), q_in2: (n_in2,), q_in3: (n_in3,)
    returns: (n_out,) normalized
    """
    msg = jnp.einsum("ijkl,j,k,l->i", tensor, q_in1, q_in2, q_in3)
    return msg / (msg.sum() + EPSILON)


def backward_message_2d(
    tensor: jnp.ndarray, obs_onehot: jnp.ndarray
) -> jnp.ndarray:
    """
    Backward message from clamped observation through 2D tensor.
    
    tensor: (n_obs, n_state)
    obs_onehot: (n_obs,)
    returns: (n_state,) unnormalized (will be combined with other messages)
    """
    return jnp.einsum("ij,i->j", tensor, obs_onehot)


def backward_message_3d(
    tensor: jnp.ndarray, obs_onehot: jnp.ndarray, q_other: jnp.ndarray
) -> jnp.ndarray:
    """
    Backward message from clamped observation through 3D tensor.
    
    tensor: (n_obs, n_state, n_other)
    obs_onehot: (n_obs,)
    q_other: (n_other,)
    returns: (n_state,) unnormalized
    """
    return jnp.einsum("ijk,i,k->j", tensor, obs_onehot, q_other)


def backward_message_to_other_3d(
    tensor: jnp.ndarray, obs_onehot: jnp.ndarray, q_state: jnp.ndarray
) -> jnp.ndarray:
    """
    Backward message to the 'other' variable through 3D tensor.
    
    tensor: (n_obs, n_state, n_other)
    obs_onehot: (n_obs,)
    q_state: (n_state,)
    returns: (n_other,) unnormalized
    """
    return jnp.einsum("ijk,i,j->k", tensor, obs_onehot, q_state)


def combine_messages(messages: list[jnp.ndarray]) -> jnp.ndarray:
    """
    Combine multiple messages into a posterior via normalized product.
    
    messages: list of (n,) arrays (same shape)
    returns: (n,) normalized posterior
    """
    log_msgs = [jnp.log(msg + EPSILON) for msg in messages]
    log_q = sum(log_msgs)
    return nn.softmax(log_q)


def marginalize_static(log_T, log_q_static):
    """Marginalize out static_state from log transition tensor.

    Args:
        log_T: (n_states, n_states, n_static, n_actions) log-space
        log_q_static: (n_static,) log-space

    Returns:
        (n_states, n_states, n_actions) log-space reduced tensor
    """
    return logsumexp(log_T + log_q_static[None, None, :, None], axis=2)


def combine_messages_log(log_messages: list[jnp.ndarray]) -> jnp.ndarray:
    """
    Combine log-space messages into a posterior.

    log_messages: list of (n,) log-probability arrays
    returns: (n,) normalized posterior
    """
    log_q = sum(log_messages)
    return nn.softmax(log_q)


# =============================================================================
# Sparse transition operations (using T_idx instead of dense tensor)
# =============================================================================


def compute_log_base_sparse(T_idx, log_weights, n_states):
    """Compute θ-marginalized transition base from sparse index representation.

    Equivalent to:
        log_T_kernel = log(transition_tensor).transpose(1, 0, 2, 3)
        log_base = logsumexp(log_T_kernel + log_weights[None, None, :, None], axis=2)

    but without materializing the dense (S, S, θ, A) tensor.

    Args:
        T_idx: (S, A, θ) int32 — deterministic next-state indices
               T_idx[x_old, action, theta] = x_new
        log_weights: (θ,) log weights per static config (e.g. log prior)
        n_states: S

    Returns:
        log_base: (S, S, A) float32
    """
    S = n_states
    A = T_idx.shape[1]
    n_static = T_idx.shape[2]

    log_base = jnp.full((S, S, A), LOG_ZERO)

    x_old_idx = jnp.arange(S)[:, None]  # (S, 1) — broadcasts over A
    action_idx = jnp.arange(A)[None, :]  # (1, A) — broadcasts over S

    def body(th, log_base):
        x_new = T_idx[:, :, th]  # (S, A)
        contrib = log_weights[th]
        current = log_base[x_old_idx, x_new, action_idx]  # (S, A)
        log_base = log_base.at[x_old_idx, x_new, action_idx].set(
            jnp.logaddexp(current, contrib)
        )
        return log_base

    return lax.fori_loop(0, n_static, body, log_base)


def sparse_reduced(T_idx, log_cavity, n_states):
    """Per-timestep θ-marginalization using sparse transition index.

    Equivalent to compute_log_reduced(log_T_kernel[None], log_cavity) where
    log_T_kernel is the deterministic transition tensor, but without
    materializing the dense (T, S, S, θ, A) kernel tensor.

    Args:
        T_idx: (S, A, θ) int32 — T_idx[x_old, action, theta] = x_new
        log_cavity: (T, θ) per-timestep cavity beliefs on theta
        n_states: S

    Returns:
        (T, S_new, S_old, A) log-space per-timestep reduced tensors
        (same convention as compute_log_reduced)
    """
    horizon = log_cavity.shape[0]
    S = n_states
    A = T_idx.shape[1]
    n_static = T_idx.shape[2]

    result = jnp.full((horizon, S, S, A), LOG_ZERO)

    x_old_idx = jnp.arange(S)[:, None]  # (S, 1)
    a_idx = jnp.arange(A)[None, :]      # (1, A)

    def body(th, result):
        x_new = T_idx[:, :, th]  # (S, A)
        contrib = log_cavity[:, th][:, None, None]  # (T, 1, 1)
        current = result[:, x_old_idx, x_new, a_idx]  # (T, S, A)
        result = result.at[:, x_old_idx, x_new, a_idx].set(
            jnp.logaddexp(current, contrib)
        )
        return result

    result = lax.fori_loop(0, n_static, body, result)
    return result.transpose(0, 2, 1, 3)  # (T, S_new, S_old, A)


def sparse_reduced_weighted(T_idx, log_cavity, log_weight, n_states):
    """Per-timestep θ-marginalization with additive kernel weight.

    Equivalent to compute_log_reduced(log_T_kernel[None] + log_weight[:,:,None,None,:],
                                       log_cavity)
    where log_weight is a θ-independent (T, S, A) tensor (e.g. action channels).

    Args:
        T_idx: (S, A, θ) int32
        log_cavity: (T, θ) per-timestep cavity beliefs
        log_weight: (T, S, A) additive kernel weight
        n_states: S

    Returns:
        (T, S_new, S_old, A) log-space per-timestep reduced tensors
    """
    horizon = log_cavity.shape[0]
    S = n_states
    A = T_idx.shape[1]
    n_static = T_idx.shape[2]

    result = jnp.full((horizon, S, S, A), LOG_ZERO)

    x_old_idx = jnp.arange(S)[:, None]  # (S, 1)
    a_idx = jnp.arange(A)[None, :]      # (1, A)

    def body(th, result):
        x_new = T_idx[:, :, th]  # (S, A)
        contrib = log_cavity[:, th][:, None, None] + log_weight  # (T, S, A)
        current = result[:, x_old_idx, x_new, a_idx]  # (T, S, A)
        result = result.at[:, x_old_idx, x_new, a_idx].set(
            jnp.logaddexp(current, contrib)
        )
        return result

    result = lax.fori_loop(0, n_static, body, result)
    return result.transpose(0, 2, 1, 3)  # (T, S_new, S_old, A)


def sparse_dyn_to_theta(T_idx, log_fwd, log_bwd, log_local_to_x, log_action,
                         n_states):
    """Compute dyn→θ messages using sparse transition index.

    Equivalent to compute_dyn_to_theta_msgs with deterministic kernels, but
    without materializing (T, S, S, θ, A).

    For each θ:
        msg[t, θ] = logsumexp_{x_old, u}(
            fwd_part[t, x_old] + bwd_part[t, T_idx[x_old, u, θ]] + action[u])

    Args:
        T_idx: (S, A, θ) int32
        log_fwd: (T+1, S) forward messages
        log_bwd: (T+1, S) backward messages
        log_local_to_x: (T+1, S) local messages to x (obs+pref), or None
        log_action: (A,) or (T, A) action prior
        n_states: S

    Returns:
        (T, θ) log-space dyn→θ messages
    """
    n_static = T_idx.shape[2]

    fwd_part = log_fwd[:-1]  # (T, S)
    bwd_part = log_bwd[1:]   # (T, S)
    if log_local_to_x is not None:
        fwd_part = fwd_part + log_local_to_x[:-1]
        bwd_part = bwd_part + log_local_to_x[1:]

    horizon = fwd_part.shape[0]
    result = jnp.full((horizon, n_static), LOG_ZERO)

    # Expand action prior for broadcasting
    if log_action.ndim == 1:
        log_action_bc = log_action[None, None, :]  # (1, 1, A)
    else:
        log_action_bc = log_action[:, None, :]  # (T, 1, A)

    def body(th, result):
        x_new = T_idx[:, :, th]  # (S, A)
        gathered_bwd = bwd_part[:, x_new]  # (T, S, A)
        terms = fwd_part[:, :, None] + gathered_bwd + log_action_bc  # (T, S, A)
        msg_th = logsumexp(terms, axis=(1, 2))  # (T,)
        return result.at[:, th].set(msg_th)

    return lax.fori_loop(0, n_static, body, result)


def sparse_dyn_to_theta_weighted(T_idx, log_fwd, log_bwd, log_local_to_x,
                                  log_action, log_kernel_weight, n_states):
    """Compute dyn→θ messages with per-entry kernel weights.

    Same as sparse_dyn_to_theta but adds log_kernel_weight[t, x_old, u] to
    each contribution. For VBP-channel, this is the action channel r(u|x).

    Args:
        T_idx: (S, A, θ) int32
        log_fwd: (T+1, S) forward messages
        log_bwd: (T+1, S) backward messages
        log_local_to_x: (T+1, S) or None
        log_action: (A,) or (T, A) action prior
        log_kernel_weight: (T, S, A) additive weight on kernel entries
        n_states: S

    Returns:
        (T, θ) log-space dyn→θ messages
    """
    n_static = T_idx.shape[2]

    fwd_part = log_fwd[:-1]  # (T, S)
    bwd_part = log_bwd[1:]   # (T, S)
    if log_local_to_x is not None:
        fwd_part = fwd_part + log_local_to_x[:-1]
        bwd_part = bwd_part + log_local_to_x[1:]

    horizon = fwd_part.shape[0]
    result = jnp.full((horizon, n_static), LOG_ZERO)

    if log_action.ndim == 1:
        log_action_bc = log_action[None, None, :]
    else:
        log_action_bc = log_action[:, None, :]

    def body(th, result):
        x_new = T_idx[:, :, th]  # (S, A)
        gathered_bwd = bwd_part[:, x_new]  # (T, S, A)
        terms = (fwd_part[:, :, None] + gathered_bwd + log_action_bc
                 + log_kernel_weight)  # (T, S, A)
        msg_th = logsumexp(terms, axis=(1, 2))  # (T,)
        return result.at[:, th].set(msg_th)

    return lax.fori_loop(0, n_static, body, result)


def sparse_pair_marginal(T_idx, log_fwd, log_bwd, log_local_to_x,
                          log_cavity_dyn, log_action, log_kernel_weight,
                          n_states):
    """Compute pair marginal q(x_old, u) from sparse region beliefs.

    No scatter needed — just logsumexp over θ. Used by VBP-channel where
    only the pair marginal (not dyn channels) is needed from region beliefs.

    Args:
        T_idx: (S, A, θ) int32
        log_fwd: (T+1, S) forward messages
        log_bwd: (T+1, S) backward messages
        log_local_to_x: (T+1, S) local messages to x
        log_cavity_dyn: (T, θ) cavity beliefs on theta for dyn factors
        log_action: (A,) action prior
        log_kernel_weight: (T, S, A) kernel weight, or None
        n_states: S

    Returns:
        log_pair_marginal: (T, S, A)
    """
    n_static = T_idx.shape[2]
    horizon = log_cavity_dyn.shape[0]
    S = n_states
    A = T_idx.shape[1]

    fwd_part = (log_fwd[:-1] + log_local_to_x[:-1])[:, :, None]  # (T, S, 1)
    bwd_part = log_bwd[1:] + log_local_to_x[1:]                   # (T, S)
    log_action_bc = log_action[None, None, :]                      # (1, 1, A)

    result = jnp.full((horizon, S, A), LOG_ZERO)

    def body(th, result):
        x_new = T_idx[:, :, th]  # (S, A)
        gathered_bwd = bwd_part[:, x_new]  # (T, S, A)

        contrib = (fwd_part + gathered_bwd + log_action_bc
                   + log_cavity_dyn[:, th][:, None, None])  # (T, S, A)
        if log_kernel_weight is not None:
            contrib = contrib + log_kernel_weight

        return jnp.logaddexp(result, contrib)

    return lax.fori_loop(0, n_static, body, result)


def _dyn_channel_kernel_weight(log_dyn_channels, log_r_ux, x_new, x_old_idx, a_idx):
    """Compute dyn-channel kernel weight for a given θ.

    weight = safe_log_div(0, dyn_ch[t, x_old, x_new, u]) + r_ux[t, x_old, u]

    This corresponds to the kernel T(x'|x,θ,u) * r(u|x) / r(x'|x,u)
    evaluated at the sparse transition x_new = T_idx[x_old, u, θ].
    """
    dch = log_dyn_channels[:, x_old_idx, x_new, a_idx]  # (T, S, A)
    return safe_log_div(jnp.zeros_like(dch), dch) + log_r_ux


def sparse_reduced_dyn_channel(T_idx, log_cavity, log_dyn_channels, log_r_ux,
                                n_states):
    """Sparse reduced for dyn-channel kernel: T * r(u|x) / r(x'|x,u).

    The kernel weight depends on θ (via T_idx into dyn_channels), so it
    cannot use sparse_reduced_weighted which takes a static weight.

    Args:
        T_idx: (S, A, θ) int32
        log_cavity: (T, θ)
        log_dyn_channels: (T, S, S, A) log conditional r(x'|x,u)
        log_r_ux: (T, S, A) log action channel r(u|x)
        n_states: S

    Returns:
        (T, S_new, S_old, A)
    """
    horizon = log_cavity.shape[0]
    S = n_states
    A = T_idx.shape[1]
    n_static = T_idx.shape[2]

    result = jnp.full((horizon, S, S, A), LOG_ZERO)

    x_old_idx = jnp.arange(S)[:, None]
    a_idx = jnp.arange(A)[None, :]

    def body(th, result):
        x_new = T_idx[:, :, th]
        weight = _dyn_channel_kernel_weight(
            log_dyn_channels, log_r_ux, x_new, x_old_idx, a_idx)
        contrib = log_cavity[:, th][:, None, None] + weight
        current = result[:, x_old_idx, x_new, a_idx]
        result = result.at[:, x_old_idx, x_new, a_idx].set(
            jnp.logaddexp(current, contrib))
        return result

    result = lax.fori_loop(0, n_static, body, result)
    return result.transpose(0, 2, 1, 3)


def sparse_dyn_to_theta_dyn_channel(T_idx, log_fwd, log_bwd, log_local_to_x,
                                     log_action, log_dyn_channels, log_r_ux,
                                     n_states):
    """Dyn→θ messages for dyn-channel kernel (θ-dependent weight).

    Args:
        T_idx: (S, A, θ) int32
        log_fwd, log_bwd: (T+1, S)
        log_local_to_x: (T+1, S) combined obs+pref messages
        log_action: (A,) action prior
        log_dyn_channels: (T, S, S, A)
        log_r_ux: (T, S, A)
        n_states: S

    Returns:
        (T, θ)
    """
    n_static = T_idx.shape[2]
    S = n_states
    A = T_idx.shape[1]

    fwd_part = log_fwd[:-1] + log_local_to_x[:-1]
    bwd_part = log_bwd[1:] + log_local_to_x[1:]
    horizon = fwd_part.shape[0]
    result = jnp.full((horizon, n_static), LOG_ZERO)

    x_old_idx = jnp.arange(S)[:, None]
    a_idx = jnp.arange(A)[None, :]
    log_action_bc = log_action[None, None, :]

    def body(th, result):
        x_new = T_idx[:, :, th]
        gathered_bwd = bwd_part[:, x_new]
        weight = _dyn_channel_kernel_weight(
            log_dyn_channels, log_r_ux, x_new, x_old_idx, a_idx)
        terms = (fwd_part[:, :, None] + gathered_bwd + log_action_bc
                 + weight)
        msg_th = logsumexp(terms, axis=(1, 2))
        return result.at[:, th].set(msg_th)

    return lax.fori_loop(0, n_static, body, result)


def sparse_dyn_channels_and_pair_dyn_channel(T_idx, log_fwd, log_bwd,
                                              log_local_to_x, log_cavity_dyn,
                                              log_action, log_dyn_channels,
                                              log_r_ux, n_states):
    """Dyn channels + pair marginal for dyn-channel kernel (θ-dependent weight).

    Args:
        T_idx: (S, A, θ) int32
        log_fwd, log_bwd: (T+1, S)
        log_local_to_x: (T+1, S)
        log_cavity_dyn: (T, θ)
        log_action: (A,)
        log_dyn_channels: (T, S, S, A) current channels
        log_r_ux: (T, S, A) current action channels
        n_states: S

    Returns:
        new_log_dyn_channels: (T, S, S, A)
        log_pair_marginal: (T, S, A)
    """
    n_static = T_idx.shape[2]
    horizon = log_cavity_dyn.shape[0]
    S = n_states
    A = T_idx.shape[1]

    fwd_part = (log_fwd[:-1] + log_local_to_x[:-1])[:, :, None]
    bwd_part = log_bwd[1:] + log_local_to_x[1:]
    log_action_bc = log_action[None, None, :]

    log_theta_marg = jnp.full((horizon, S, S, A), LOG_ZERO)

    x_old_idx = jnp.arange(S)[:, None]
    a_idx = jnp.arange(A)[None, :]

    def body(th, log_theta_marg):
        x_new = T_idx[:, :, th]
        gathered_bwd = bwd_part[:, x_new]
        weight = _dyn_channel_kernel_weight(
            log_dyn_channels, log_r_ux, x_new, x_old_idx, a_idx)

        sparse_region = (fwd_part + gathered_bwd + log_action_bc
                         + log_cavity_dyn[:, th][:, None, None] + weight)

        current = log_theta_marg[:, x_old_idx, x_new, a_idx]
        log_theta_marg = log_theta_marg.at[:, x_old_idx, x_new, a_idx].set(
            jnp.logaddexp(current, sparse_region))
        return log_theta_marg

    log_theta_marg = lax.fori_loop(0, n_static, body, log_theta_marg)

    new_log_dyn_channels = log_theta_marg - logsumexp(log_theta_marg, axis=2, keepdims=True)
    log_pair_marginal = logsumexp(log_theta_marg, axis=2)

    return new_log_dyn_channels, log_pair_marginal


def sparse_efe_action_prior(T_idx, log_fwd, log_bwd, log_local_to_x,
                             log_cavity_dyn, log_action_per_t, n_states,
                             action_mask):
    """Compute EFE-based action prior from sparse region beliefs.

    Equivalent to compute_dyn_region_beliefs_nuijten + compute_efe_action_prior
    but without materializing (T, S, S, θ, A). Operates on the sparse
    representation (T, S, A, θ) where x_new = T_idx[x_old, u, θ].

    The EFE is H(x_new | x_old, θ, u, t) computed from region beliefs
    normalized per (t, u). Uses a two-pass approach (normalizer then EFE)
    to keep peak memory at (T, S, A) per iteration.

    Args:
        T_idx: (S, A, θ) int32
        log_fwd: (T+1, S) forward messages
        log_bwd: (T+1, S) backward messages
        log_local_to_x: (T+1, S) combined obs+pref messages
        log_cavity_dyn: (T, θ) cavity beliefs on theta
        log_action_per_t: (T, A) per-timestep action prior
        n_states: S
        action_mask: (A,) binary mask

    Returns:
        action_prior_per_t: (T, A) probability-space action priors
    """
    import jax

    n_static = T_idx.shape[2]
    S = n_states
    A = T_idx.shape[1]

    fwd_part = log_fwd[:-1] + log_local_to_x[:-1]  # (T, S)
    bwd_part = log_bwd[1:] + log_local_to_x[1:]     # (T, S)
    horizon = fwd_part.shape[0]

    # Pass 1: compute log normalizer per (t, u)
    #   log_Z[t, u] = logsumexp_{x_old, θ} logit[t, x_old, u, θ]
    log_Z = jnp.full((horizon, A), LOG_ZERO)

    def pass1_body(th, log_Z):
        x_new = T_idx[:, :, th]           # (S, A)
        gathered_bwd = bwd_part[:, x_new]  # (T, S, A)
        logit = (fwd_part[:, :, None] + gathered_bwd
                 + log_cavity_dyn[:, th][:, None, None]
                 + log_action_per_t[:, None, :])  # (T, S, A)
        per_theta = logsumexp(logit, axis=1)      # (T, A) sum over x_old
        return jnp.logaddexp(log_Z, per_theta)

    log_Z = lax.fori_loop(0, n_static, pass1_body, log_Z)

    # Pass 2: compute EFE from normalized q
    #   For deterministic transitions: q_marg(x_old,θ) = q(x_old,θ) since
    #   there's exactly one x_new per (x_old,u,θ). So q_cond = q/(q+ε).
    EPSILON_LOCAL = 1e-10
    efe = jnp.zeros((horizon, A))

    def pass2_body(th, efe):
        x_new = T_idx[:, :, th]
        gathered_bwd = bwd_part[:, x_new]
        logit = (fwd_part[:, :, None] + gathered_bwd
                 + log_cavity_dyn[:, th][:, None, None]
                 + log_action_per_t[:, None, :])  # (T, S, A)

        q_th = jnp.exp(logit - log_Z[:, None, :])  # (T, S, A) normalized
        q_cond = q_th / (q_th + EPSILON_LOCAL)
        efe = efe - (q_th * jnp.log(q_cond + EPSILON_LOCAL)).sum(axis=1)  # sum over x_old
        return efe

    efe = lax.fori_loop(0, n_static, pass2_body, efe)

    efe = jnp.where(action_mask[None] > 0, efe, -jnp.inf)
    return jax.nn.softmax(efe, axis=1)


def sparse_dyn_channels_and_pair(T_idx, log_fwd, log_bwd, log_local_to_x,
                                  log_cavity_dyn, log_action, log_kernel_weight,
                                  n_states):
    """Compute dyn channels + pair marginal from sparse representation.

    Replaces compute_dyn_region_beliefs + compute_dyn_channels_and_pair_marginal
    without materializing (T, S, S, θ, A) region beliefs.

    Args:
        T_idx: (S, A, θ) int32
        log_fwd: (T+1, S) forward messages
        log_bwd: (T+1, S) backward messages
        log_local_to_x: (T+1, S) local messages to x (obs+pref)
        log_cavity_dyn: (T, θ) cavity beliefs on theta for dyn factors
        log_action: (A,) action prior
        log_kernel_weight: (T, S, A) kernel weight, or None for unweighted
        n_states: S

    Returns:
        log_dyn_channels: (T, S_old, S_new, A) log-conditional r(x_new|x_old,u)
        log_pair_marginal: (T, S, A) log pair marginal q(x_old, u)
    """
    n_static = T_idx.shape[2]
    horizon = log_cavity_dyn.shape[0]
    S = n_states
    A = T_idx.shape[1]

    fwd_part = (log_fwd[:-1] + log_local_to_x[:-1])[:, :, None]  # (T, S, 1)
    bwd_part = log_bwd[1:] + log_local_to_x[1:]                   # (T, S)
    log_action_bc = log_action[None, None, :]                      # (1, 1, A)

    # Accumulate θ-marginalized region into (T, S_old, S_new, A) for channels
    log_theta_marg = jnp.full((horizon, S, S, A), LOG_ZERO)

    x_old_idx = jnp.arange(S)[:, None]  # (S, 1)
    a_idx = jnp.arange(A)[None, :]      # (1, A)

    def body(th, log_theta_marg):
        x_new = T_idx[:, :, th]  # (S, A)
        gathered_bwd = bwd_part[:, x_new]  # (T, S, A)

        # Sparse region belief at (t, x_old, θ=th, u)
        sparse_region = (fwd_part + gathered_bwd + log_action_bc
                         + log_cavity_dyn[:, th][:, None, None])  # (T, S, A)
        if log_kernel_weight is not None:
            sparse_region = sparse_region + log_kernel_weight

        # Scatter into θ-marginalized result
        current = log_theta_marg[:, x_old_idx, x_new, a_idx]  # (T, S, A)
        log_theta_marg = log_theta_marg.at[:, x_old_idx, x_new, a_idx].set(
            jnp.logaddexp(current, sparse_region)
        )
        return log_theta_marg

    log_theta_marg = lax.fori_loop(0, n_static, body, log_theta_marg)

    # Channels: normalize over x_new
    log_dyn_channels = log_theta_marg - logsumexp(log_theta_marg, axis=2, keepdims=True)
    # Pair marginal: marginalize x_new
    log_pair_marginal = logsumexp(log_theta_marg, axis=2)

    return log_dyn_channels, log_pair_marginal
