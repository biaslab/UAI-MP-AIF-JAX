---
name: aif-message-passing
description: Converts a standard Bethe BP implementation to the Active Inference (AIF-MP) message-passing scheme. Specifies four channel reparameterizations, modified factor kernels, and correct fixed-point equations. Reference paper Appendix D (T=1 derivation) and D.6 (generic T).
---

# Active Inference Message Passing (AIF-MP)

AIF-MP is standard Bethe BP with entropy corrections reparameterized into modified factor kernels via **four** channel variables per time step. It combines the cross-entropy planning correction (VBP) with three epistemic corrections.

Reference: paper Section 5, Appendix D (full derivation for T=1), Section D.6 (generic T).

## 1. Objective

The combined EFE-based planning objective adds to the Bethe free energy (paper eq 13, Corollary 9 eq B.5):

$$\Delta F_{\mathrm{comb}} = \underbrace{\sum_t 2\mathbb{H}[q(y_t|x_t,\theta)] - \mathbb{H}[q(x_t|x_{t-1},u_t)] - \mathbb{H}[q(y_t|x_t)]}_{\Delta^{\mathrm{AIF}}} + \underbrace{\sum_t \mathbb{H}[q(u_t|x_{t-1})]}_{\Delta^{\mathrm{planning}}}$$

## 2. Four Channels per Time Step

Each time step $t$ has four channel variables (paper eq D.37):

| Channel | Symbol | Normalizes over | Enters kernel |
|---|---|---|---|
| Observation | $r_{y\mid x\theta,t}(y_t\mid x_t,\theta)$ | $\sum_{y_t} r = 1\;\forall(x_t,\theta)$ | Obs numerator (squared) |
| Marginal observation | $r_{y\mid x,t}(y_t\mid x_t)$ | $\sum_{y_t} r = 1\;\forall x_t$ | Obs denominator |
| Predictive dynamics | $r_{x\mid xu,t}(x_t\mid x_{t-1},u_t)$ | $\sum_{x_t} r = 1\;\forall(x_{t-1},u_t)$ | Dyn denominator |
| Policy | $r_{u\mid x,t}(u_t\mid x_{t-1})$ | $\sum_{u_t} r = 1\;\forall x_{t-1}$ | Dyn numerator |

Setting all four channels to uniform recovers standard Bethe BP.

## 3. Modified Kernels (paper eq D.38)

| Factor | AIF kernel |
|---|---|
| $f_{\mathrm{obs}_t}$ | $\displaystyle\frac{p(y_t\mid x_t,\theta)\cdot r_{y\mid x\theta,t}^2(y_t\mid x_t,\theta)}{r_{y\mid x,t}(y_t\mid x_t)}$ |
| $f_{\mathrm{dyn}_t}$ | $\displaystyle\frac{p(x_t\mid x_{t-1},\theta,u_t)\cdot r_{u\mid x,t}(u_t\mid x_{t-1})}{r_{x\mid xu,t}(x_t\mid x_{t-1},u_t)}$ |

The dynamics kernel combines the VBP action channel (numerator, sharpens policy) with the predictive dynamics channel (denominator, spreads mass over futures). See paper Remark 16.

## 4. Channel Updates (paper eq D.42)

At each iteration, channels recover conditionals from factor beliefs:

$$r_{y|x\theta,t}^* = q_t(y_t|x_t,\theta), \quad r_{y|x,t}^* = q_t(y_t|x_t), \quad r_{x|xu,t}^* = q_t(x_t|x_{t-1},u_t), \quad r_{u|x,t}^* = q_t(u_t|x_{t-1})$$

Computed from region beliefs:
- $r_{y|x\theta} \leftarrow q_{\mathrm{obs}} / q_{\mathrm{sep}}$ where $q_{\mathrm{sep}} = \sum_y q_{\mathrm{obs}}$
- $r_{y|x} \leftarrow q_{yx} / q_x$ where $q_{yx} = \sum_\theta q_{\mathrm{obs}}$
- $r_{x|xu} \leftarrow q_{\mathrm{trip}} / q_{\mathrm{pair}}$ where $q_{\mathrm{trip}} = \sum_\theta q_{\mathrm{dyn}}$, $q_{\mathrm{pair}} = \sum_x q_{\mathrm{trip}}$
- $r_{u|x} \leftarrow q_{\mathrm{pair}} / q_{x_{t-1}}$

## 5. Implementation Details (`inference/active_inference.py`)

Key differences from the generic paper scheme:

- **theta handling**: Uses $p(\theta)$ prior directly -- no theta cavity messages, no theta inference. This simplifies the iteration (no dyn->theta or obs->theta messages).
- **Damping**: Geometric damping in log-space: `damped = (1-alpha)*log_old + alpha*log_new`, followed by renormalization. NOT arithmetic damping.
- **Precomputed obs path**: When `log_local_to_x` is provided, obs channels are precomputed outside the planning loop via `precompute_obs_channels()`, since obs channel updates depend only on B, prior_theta, and damping -- not on fwd/bwd messages.
- **theta-marginalized base**: Precomputes `log_base = logsumexp(log_T_kernel + log_prior_theta, axis=theta)` giving shape `(S, S, A)`. The per-iteration dynamics kernel is then `log_base - log_dyn_channels + log_r_ux` (4D, no theta dimension).
- **Return**: `(action_dist, log_dyn_channels, log_obs_channels)`. `log_obs_channels` is `None` when using the precomputed obs path.

### Iteration structure (precomputed obs path)

```
Initialize: r_{u|x} = uniform, r_{x|xu} = uniform
Precompute: log_base = marginalize_theta(T), log_local_to_x = obs_channels(B) + pref(C)

For each iteration:
  1. Build 4D kernel: base / r_{x|xu} * r_{u|x}
  2. Forward pass (inject log_local_to_x at each step)
  3. Backward pass + action marginals
  4. Compute theta-marginalized dyn region: kernel * fwd * bwd * action_prior
  5. Dyn channels: normalize region over x_new
  6. Action channels: marginalize x_new from region, normalize over u
  7. Damp both channels (geometric)
```

### Iteration structure (dense path, with obs channels in carry)

Same as above but additionally:
- Obs kernels recomputed each iteration: `B * r_{y|xθ} * r_{y|xθ}/r_{y|x}`
- Obs->x messages recomputed from obs kernels
- Obs channels updated: normalize obs kernels over y
- Marginal obs channels: marginalize theta from obs kernels, normalize over y

## 6. Verification Checklist

- [ ] Obs kernel: $p \cdot r_{y|x\theta}^2 / r_{y|x}$ (squared numerator, denominator)?
- [ ] Dyn kernel: $p \cdot r_{u|x} / r_{x|xu}$ (action channel in numerator, dyn channel in denominator)?
- [ ] Four channels per timestep, all initialized uniform?
- [ ] Setting all channels to uniform recovers standard BP?
- [ ] Geometric damping (log-space interpolation), not arithmetic?
- [ ] Channel normalization maintained after damping?
- [ ] theta uses prior directly (no cavity messages)?
