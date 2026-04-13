---
name: vbp-message-passing
description: Converts a standard Bethe BP implementation to the VBP (cross-entropy planning) message-passing scheme. Single action channel in the dynamics numerator, unmodified observation kernel, theta inferred via cavities. Reference paper Appendix E.
---

# VBP Message Passing (Cross-Entropy Planning)

VBP is standard Bethe BP with a single entropy correction -- the policy entropy -- reparameterized into the dynamics kernel via one action channel per time step. Substantially simpler than AIF-MP (1 channel vs 4).

Reference: paper Section 4.2, Appendix E (full derivation).

## 1. Objective (paper eq E.4)

$$\Delta F_{\mathrm{VBP}} = +\sum_{t=1}^{T} \mathbb{H}[q(u_t|x_{t-1})]$$

This penalizes action entropy, pushing the agent toward a committed policy (cross-entropy planning = expected reward maximization with policy commitment).

## 2. Single Channel

| Channel | Symbol | Normalizes over | Enters kernel |
|---|---|---|---|
| Policy | $r_{u\mid x,t}(u_t\mid x_{t-1})$ | $\sum_{u_t} r = 1\;\forall x_{t-1}$ | Dyn numerator |

Setting $r_{u|x} \equiv$ uniform recovers standard BP. Since `+H[q(u|x)]` has a positive sign, the channel enters the **numerator** (multiplicative), unlike the AIF dynamics channel $r_{x|xu}$ which enters the denominator (paper Remark 17).

## 3. Modified Kernels (paper eq E.8)

| Factor | VBP kernel |
|---|---|
| $f_{\mathrm{obs}_t}$ | $p(y_t\mid x_t,\theta)$ -- **unmodified** |
| $f_{\mathrm{dyn}_t}$ | $p(x_t\mid x_{t-1},\theta,u_t) \cdot r_{u\mid x,t}(u_t\mid x_{t-1})$ |

## 4. Channel Update (paper Proposition 19, eq E.11)

$$r_{u|x,t}^*(u_t|x_{t-1}) = \frac{q_{\mathrm{pair},t}(x_{t-1}, u_t)}{q_{x_{t-1}}(x_{t-1})} = q_t(u_t|x_{t-1})$$

where $q_{\mathrm{pair}} = \sum_{x_t,\theta} q_{\mathrm{dyn},t}$ is the pair marginal from the dynamics factor belief.

## 5. Implementation Details (`inference/vbp_channel.py`)

- **theta handling**: Inferred via cavity messages. Uses `compute_theta_cavities_extended` for per-timestep dyn/obs/pref cavity beliefs. This is different from AIF-MP which uses theta prior directly.
- **Observation kernel**: Raw B tensor, no channels. Tiled over time as `log_B_tiled`.
- **Damping**: Geometric damping in log-space (same as AIF-MP): `damped = (1-alpha)*log_old + alpha*log_new`. Applied to action channels only.
- **Sparse transitions**: Supports `T_idx` (S, A, theta) int32 sparse index via `sparse_reduced_weighted`, `sparse_dyn_to_theta_weighted`, `sparse_pair_marginal` from `messages.py`.
- **Preference factors**: Supports 2D goal `C(x, theta)` with pref->x and pref->theta messages via 3-way theta cavities.
- **Return**: `(action_dist, log_action_channels)` where `log_action_channels` is `(T, n_states, n_actions)`.

### Iteration structure

```
Initialize: r_{u|x} = uniform, theta messages = zeros

For each iteration:
  1. Theta cavities (exclude dyn[t] from theta belief for dyn, exclude obs[t] for obs)
  2. Dyn kernels: T * r_{u|x} (channel in numerator)
  3. Reduce: marginalize theta from dyn kernels using cavity beliefs
  4. obs->x messages (raw B with obs cavity), pref->x if applicable
  5. Forward pass
  6. Backward pass + action marginals
  7. dyn->theta messages (with kernel weight = action channel)
  8. obs->theta messages (raw B)
  9. pref->theta messages (if applicable)
  10. Dyn region beliefs -> pair marginal -> action channel -> damp
```

### Key functions used

- `compute_dyn_kernels_vbp(log_T_kernel, log_r_ux)` -- T + r_ux (broadcast)
- `compute_pair_marginal(log_dyn_regions)` -- logsumexp over (x_new, theta)
- `compute_action_channel(log_pair)` -- normalize over actions
- `damp_log_channel(old, new, damping, cond_axis)` -- geometric damping

## 6. Properties (paper Section E.6)

- **No min-max structure**: Single channel in numerator only. Joint optimization over $(q, r_{u|x})$ is pure minimization, avoiding AIF's saddle-point convergence difficulties.
- **Reduction to standard BP**: Setting $r_{u|x} \equiv$ uniform removes the correction.
- **Fixed-point interpretation**: At convergence, dyn kernel becomes $p(x_t|x_{t-1},\theta,u_t) \cdot q(u_t|x_{t-1})$, reweighting the transition by the learned policy.

## 7. Verification Checklist

- [ ] Observation messages are standard sum-product (no channel)?
- [ ] Dynamics kernel: $p \cdot r_{u|x}$ (channel in **numerator**)?
- [ ] Setting $r_{u|x} \equiv$ uniform recovers standard BP?
- [ ] Channel update: $r_{u|x} \leftarrow q_{\mathrm{pair}} / q_x$?
- [ ] Geometric damping (log-space interpolation)?
- [ ] Theta inferred via cavity messages (not prior-only)?
