---
name: discrete-factor-graph-implementer
description: Expert in implementing discrete message-passing schemes using JAX primitives. Translates LaTeX update rules into indexed or tensor-based contractions for state-space models and planning. Covers the full method family from standard BP to AIF-MP.
allowed-tools: Read, Write, Edit
---

# Discrete Message Passing Implementation Protocol

You implement inference loops for discrete factor graphs using JAX. All computation is in **log-space**. The codebase implements a family of planning algorithms that progressively add entropy corrections to standard Bethe BP.

Reference: paper Table 1 for the method taxonomy, Appendices D and E for derivations.

## 1. Method Family

| Method | Module | Channels | Obs kernel | Dyn kernel | theta |
|---|---|---|---|---|---|
| Loopy BP | `loopy_bp.py` | None | N/A (no obs factors) | $p(x\mid x',\theta,u)$ | Inferred (cavity) |
| Loopy VBP | `loopy_vbp.py` | None | N/A (no obs factors) | $p$ (max over u) | Inferred (cavity) |
| Region-extended | `region_extended_loopy_bp.py` | $r_{x\mid xu}$, $r_{y\mid x\theta}$, $r_{y\mid x}$ | $p \cdot r_{y\mid x\theta}^2 / r_{y\mid x}$ | $p / r_{x\mid xu}$ | Prior |
| Dyn-channel | `dyn_channel_loopy_bp.py` | $r_{x\mid xu}$, $r_{u\mid x}$ | Standard $p$ | $p \cdot r_{u\mid x} / r_{x\mid xu}$ | Inferred (cavity) |
| Nuijten-MP | `nuijten_mp.py` | None (EFE action prior) | Standard $p$ | Standard $p$ | Inferred (cavity) |
| VBP-channel | `vbp_channel.py` | $r_{u\mid x}$ | Standard $p$ | $p \cdot r_{u\mid x}$ | Inferred (cavity) |
| Precise-info-seeking | `precise_info_seeking.py` | $r_{u\mid x}$, $r_{y\mid x\theta}$, $r_{y\mid x}$ | $p \cdot r_{y\mid x\theta}^2 / r_{y\mid x}$ | $p \cdot r_{u\mid x}$ | Prior |
| AIF-MP | `active_inference.py` | $r_{u\mid x}$, $r_{x\mid xu}$, $r_{y\mid x\theta}$, $r_{y\mid x}$ | $p \cdot r_{y\mid x\theta}^2 / r_{y\mid x}$ | $p \cdot r_{u\mid x} / r_{x\mid xu}$ | Prior |

## 2. Core Primitives (`inference/messages.py`)

### Constants and utilities

| Function | Purpose |
|---|---|
| `LOG_ZERO = -1e12` | Sentinel for log(0) |
| `safe_log(x)` | Maps 0 -> LOG_ZERO, avoids -inf |
| `safe_log_div(log_num, log_den)` | log(num/den), handles 0/0 -> LOG_ZERO |
| `EPSILON = 1e-10` | Small constant for normalization |

### Dense tensor messages

| Function | Signature | Use |
|---|---|---|
| `forward_message_2d` | `(n_out, n_in), (n_in,) -> (n_out,)` | 2D forward |
| `forward_message_3d` | `(n_out, n_in1, n_in2), (n_in1,), (n_in2,) -> (n_out,)` | 3D forward |
| `forward_message_4d` | `(n_out, n_in1, n_in2, n_in3), ... -> (n_out,)` | 4D forward |
| `backward_message_2d` | `(n_obs, n_state), (n_obs,) -> (n_state,)` | Clamped obs backward |
| `backward_message_3d` | `(n_obs, n_state, n_other), ... -> (n_state,)` | 3D obs backward |
| `combine_messages(list)` | list of `(n,)` prob -> `(n,)` prob | Normalized product (prob space) |
| `combine_messages_log(list)` | list of `(n,)` log -> `(n,)` prob | Normalized product (log space) |
| `marginalize_static(log_T, log_q)` | `(S,S,theta,A), (theta,) -> (S,S,A)` | theta marginalization |

### Sparse transition operations (using `T_idx` instead of dense tensor)

For large state spaces, transitions are stored as `T_idx: (S, A, theta) int32` where `T_idx[x_old, action, theta] = x_new` (deterministic per config). These avoid materializing dense `(S, S, theta, A)` tensors.

| Function | Purpose |
|---|---|
| `compute_log_base_sparse(T_idx, log_weights, n_states)` | theta-marginalized base `(S, S, A)` |
| `sparse_reduced(T_idx, log_cavity, n_states)` | Per-timestep reduced `(T, S_new, S_old, A)` |
| `sparse_reduced_weighted(T_idx, log_cavity, log_weight, n_states)` | Same + additive kernel weight |
| `sparse_dyn_to_theta(T_idx, ...)` | dyn->theta messages |
| `sparse_dyn_to_theta_weighted(T_idx, ..., log_kernel_weight, ...)` | dyn->theta with kernel weight (VBP) |
| `sparse_pair_marginal(T_idx, ...)` | Pair marginal `q(x_old, u)` for channel update |
| `sparse_reduced_dyn_channel(T_idx, ..., log_dyn_channels, log_r_ux, ...)` | Reduced for dyn-channel kernel (theta-dependent weight) |
| `sparse_dyn_to_theta_dyn_channel(...)` | dyn->theta for dyn-channel kernel |
| `sparse_dyn_channels_and_pair(T_idx, ...)` | Dyn channels + pair marginal from sparse beliefs |
| `sparse_dyn_channels_and_pair_dyn_channel(...)` | Same for dyn-channel kernel |
| `sparse_efe_action_prior(T_idx, ...)` | EFE-based action prior (Nuijten-MP) |

### Shared functions from `region_extended_loopy_bp.py`

Most channel-based methods import these:

| Function | Purpose |
|---|---|
| `compute_log_reduced(log_kernels, log_cavity)` | `(T, x_old, x_new, theta, u) + (T, theta) -> (T, x_new, x_old, u)` |
| `forward_pass(log_reduced, log_q0, log_action, log_obs_to_x, T)` | Forward with obs injection + optional inertial damping |
| `backward_pass(log_reduced, log_fwd, log_goal, log_action, log_obs_to_x, T)` | Backward + action marginals + optional inertial damping |
| `compute_theta_cavities_extended(prior, dyn_to_theta, obs_to_theta, [pref_to_theta])` | Forward-backward prefix sums for per-timestep theta cavities |
| `compute_dyn_to_theta_msgs(log_kernels, fwd, bwd, obs_to_x, action, T)` | Messages from dyn factors to theta |
| `compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs)` | Obs factor -> x messages |
| `compute_obs_to_theta_msgs(log_obs_kernels, fwd, bwd, obs_to_x)` | Obs factor -> theta messages |
| `compute_pref_to_x_msgs(log_C, log_cavity_pref)` | Preference factor -> x messages |
| `compute_pref_to_theta_msgs(log_C, fwd, bwd, obs_to_x)` | Preference factor -> theta messages |
| `compute_dyn_region_beliefs(log_kernels, fwd, bwd, obs_to_x, cavity, action)` | `(T, x_old, x_new, theta, u)` unnormalized |
| `compute_dyn_channels(log_region)` | `r(x_new | x_old, u)` from region beliefs |
| `compute_dyn_channels_and_pair_marginal(log_region)` | Fused dyn channels + pair marginal |
| `compute_obs_channels(log_region)` | `r(y|x,theta)` from obs region beliefs |
| `compute_marginal_obs_channels(log_region)` | `r(y|x)` from obs region beliefs |
| `damp_log_channel(old, new, damping, cond_axis)` | Geometric damping in log-space |

## 3. Kernel Modification Rules

When implementing a channel-based method, replace the original factor with its modified kernel:

### In log-space

**Observation kernel** (AIF-MP, region-extended, precise-info-seeking):
```python
log_obs_kernels = log_B + log_obs_ch + safe_log_div(log_obs_ch, log_marg_obs_ch[..., None])
# = log(B) + log(r_{y|xθ}) + log(r_{y|xθ}/r_{y|x})
# = log(B * r²_{y|xθ} / r_{y|x})
```

**Dynamics kernel -- VBP** (vbp-channel, precise-info-seeking):
```python
log_dyn_kernels = log_T_kernel[None] + log_r_ux[:, :, None, None, :]
# = log(T * r_{u|x})  -- channel in NUMERATOR
```

**Dynamics kernel -- Dyn-channel** (dyn-channel, active-inference):
```python
log_dyn_kernels = safe_log_div(log_T_kernel[None], log_dyn_channels[:, :, :, None, :])
# = log(T / r_{x|xu})  -- channel in DENOMINATOR
```

**Dynamics kernel -- Combined AIF** (active-inference):
```python
# Uses precomputed theta-marginalized base:
base_4d = log_base[None] + neg_dyn_ch + log_r_ux[:, :, None, :]
# = log(T_marg * r_{u|x} / r_{x|xu})
```

## 4. Implementation Pipeline

### Phase 1: Setup
- Log all inputs once at top (`safe_log`)
- Transpose T to `(x_old, x_new, theta, u)` convention
- Initialize channels to uniform, messages to zeros

### Phase 2: Iteration loop (`lax.fori_loop`)
1. **Theta cavities** (if theta-inferred): `compute_theta_cavities_extended`
2. **Compute kernels**: Apply channel modifications to original factors
3. **Reduce**: Marginalize theta from dyn kernels -> `(T, x_new, x_old, u)`
4. **Local messages**: obs->x, pref->x (if applicable)
5. **Forward pass**: Propagate state beliefs forward
6. **Backward pass**: Propagate goal backward + compute action marginals
7. **Theta messages** (if theta-inferred): dyn->theta, obs->theta, pref->theta
8. **Region beliefs**: Factor beliefs from kernels * variable-to-factor messages
9. **Channel updates**: Conditionals from region beliefs, with damping

### Phase 3: Output
- Action distribution from `q_u[0]` (first timestep)
- Channels for warm-starting or diagnostics

## 5. Numerical Guardrails

- **Log-space throughout**: All messages, beliefs, channels stored as log-probabilities. Convert to prob only at final output via softmax.
- **LOG_ZERO = -1e12**: Sentinel for structural zeros. Check `> LOG_ZERO / 2` before division.
- **safe_log_div**: Handles 0/0 gracefully (returns LOG_ZERO).
- **Channel normalization**: After geometric damping, renormalize to maintain valid conditionals. `damp_log_channel` does this automatically.
- **Sparse ops**: Use when `T_idx` is provided. Avoid materializing `(T, S, S, theta, A)` tensors for large state spaces.
- **JIT compilation**: All planning functions use `@jax.jit` with `static_argnums` for horizon and n_iterations. Use `lax.fori_loop` inside JIT, not Python loops.
