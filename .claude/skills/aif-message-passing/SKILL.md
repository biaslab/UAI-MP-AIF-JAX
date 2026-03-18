---
name: active-inference-bethe-passer
description: Converts a standard Bethe BP implementation to the Active Inference message-passing scheme. Specifies channel reparameterization, modified factor kernels, region beliefs, and correct fixed-point equations as derived in the paper.
---

# Active Inference: Converting Standard BP to the AIF Scheme

You are an expert in converting a **standard Bethe Belief Propagation** implementation into the **Active Inference message-passing scheme** as derived in the paper (Appendix, Section "Generic Message-Passing Scheme for Arbitrary Time Steps"). The AIF scheme is standard Bethe BP with two modifications: (1) three entropy corrections to the objective, and (2) channel reparameterization of those corrections into modified factor kernels using three channel variables per time step.

## 1. What AIF Changes vs Standard Bethe BP

The Active Inference objective is:
$$F_{\mathrm{AIF}}[q, r] = F_{\mathrm{Bethe}}[q] + \Delta F_{\mathrm{AIF}}$$

where the entropic correction is (Theorem 1, eq:aif_correction):
$$\Delta F_{\mathrm{AIF}} = \sum_{t=1}^{T} \Bigl[2\,H(q(y_t \mid x_t, \theta)) - H(q(x_t \mid x_{t-1}, u_t)) - H(q(y_t \mid x_t))\Bigr]$$

**Net effect**: AIF adds three corrections per time step: (1) observation ambiguity penalty ($+2H[y|x,\theta]$, doubled), (2) dynamics certainty reward ($-H[x|x_{t-1},u]$), and (3) marginal observation certainty reward ($-H[y|x]$). The factor of 2 on the observation term and the marginal observation term are what distinguish AIF from risk-minimizing planning.

## 2. The Conversion Rule: Channel Reparameterization

To make the entropy corrections tractable, introduce **time-local channel variables** as free variational parameters. Each time step $t$ has its own triple of channels:

| Channel | Symbol | Normalization | Recovers at fixed point |
| :--- | :--- | :--- | :--- |
| Observation (time $t$) | $r_{y \mid x\theta,t}(y_t \mid x_t, \theta)$ | $\sum_{y_t} r = 1\;\forall\,(x_t,\theta)$ | $q_t(y_t \mid x_t, \theta)$ |
| Marginal observation (time $t$) | $r_{y \mid x,t}(y_t \mid x_t)$ | $\sum_{y_t} r = 1\;\forall\,x_t$ | $q_t(y_t \mid x_t)$ |
| Dynamics (time $t$) | $r_{x \mid xu,t}(x_t \mid x_{t-1}, u_t)$ | $\sum_{x_t} r = 1\;\forall\,(x_{t-1},u_t)$ | $q_t(x_t \mid x_{t-1}, u_t)$ |

**Key identity**: $H[q(y|x,\theta)] = \min_r \mathbb{E}_q[-\log r(y|x,\theta)]$, achieved when $r = q(y|x,\theta)$. The channel variables make this a well-posed variational problem.

**Why time-local?** The entropy corrections in $\Delta F_{\mathrm{AIF}}$ decompose as a sum over time steps. Each term involves the conditional under the factor belief *at that specific time step*. The channel at time $t$ re-localizes the entropy correction by recovering the conditional $q_t(y_t|x_t,\theta)$, $q_t(y_t|x_t)$, or $q_t(x_t|x_{t-1},u_t)$ from the factor belief at time $t$. This per-factor locality is what makes the scheme amenable to message passing.

### The Central Modification

Replace the factor kernels used in standard BP **at each time step $t$**:

| Factor | Standard BP kernel | AIF kernel |
| :--- | :--- | :--- |
| $f_{\mathrm{obs}_t}$ | $p(y_t \mid x_t, \theta)$ | $\displaystyle\frac{p(y_t \mid x_t, \theta) \cdot r_{y \mid x\theta,t}^2(y_t \mid x_t, \theta)}{r_{y \mid x,t}(y_t \mid x_t)}$ |
| $f_{\mathrm{dyn}_t}$ | $p(x_t \mid x_{t-1}, \theta, u_t)$ | $\displaystyle\frac{p(x_t \mid x_{t-1}, \theta, u_t)}{r_{x \mid xu,t}(x_t \mid x_{t-1}, u_t)}$ |

**Setting $r_{y|x\theta,t} \equiv 1$, $r_{y|x,t} \equiv 1$, and $r_{x|xu,t} \equiv 1$ for all $t$ recovers standard Bethe BP exactly.** All other BP mechanics (variable-to-factor messages, belief updates, convergence) remain unchanged.

## 3. Generic Message Equations (Time Step $t$)

Using $\mu_{a \to i}$ for factor-to-variable and $\mu_{i \to a}$ for variable-to-factor messages, where:
$$\mu_{i \to a}(s_i) \propto \prod_{b \in \partial i \setminus a} \mu_{b \to i}(s_i)$$

### Messages from Observation Factor $f_{\mathrm{obs}_t}$

$$\mu_{\mathrm{obs}_t \to \theta}(\theta) = \sum_{y_t} \sum_{x_t} \frac{p(y_t|x_t,\theta)\, r_{y|x\theta,t}^2(y_t|x_t,\theta)}{r_{y|x,t}(y_t|x_t)}\, \mu_{y_t \to \mathrm{obs}_t}(y_t)\, \mu_{x_t \to \mathrm{obs}_t}(x_t)$$
$$\mu_{\mathrm{obs}_t \to x_t}(x_t) = \sum_{y_t} \sum_{\theta} \frac{p(y_t|x_t,\theta)\, r_{y|x\theta,t}^2(y_t|x_t,\theta)}{r_{y|x,t}(y_t|x_t)}\, \mu_{y_t \to \mathrm{obs}_t}(y_t)\, \mu_{\theta \to \mathrm{obs}_t}(\theta)$$
$$\mu_{\mathrm{obs}_t \to y_t}(y_t) = \sum_{x_t} \sum_{\theta} \frac{p(y_t|x_t,\theta)\, r_{y|x\theta,t}^2(y_t|x_t,\theta)}{r_{y|x,t}(y_t|x_t)}\, \mu_{\theta \to \mathrm{obs}_t}(\theta)\, \mu_{x_t \to \mathrm{obs}_t}(x_t)$$

### Messages from Dynamics Factor $f_{\mathrm{dyn}_t}$

$$\mu_{\mathrm{dyn}_t \to x_t}(x_t) = \sum_{x_{t-1}} \sum_{u_t} \sum_{\theta} \frac{p(x_t|x_{t-1},\theta,u_t)}{r_{x|xu,t}(x_t|x_{t-1},u_t)}\, \mu_{x_{t-1} \to \mathrm{dyn}_t}(x_{t-1})\, \mu_{u_t \to \mathrm{dyn}_t}(u_t)\, \mu_{\theta \to \mathrm{dyn}_t}(\theta)$$
$$\mu_{\mathrm{dyn}_t \to x_{t-1}}(x_{t-1}) = \sum_{x_t} \sum_{u_t} \sum_{\theta} \frac{p(x_t|x_{t-1},\theta,u_t)}{r_{x|xu,t}(x_t|x_{t-1},u_t)}\, \mu_{x_t \to \mathrm{dyn}_t}(x_t)\, \mu_{u_t \to \mathrm{dyn}_t}(u_t)\, \mu_{\theta \to \mathrm{dyn}_t}(\theta)$$
$$\mu_{\mathrm{dyn}_t \to u_t}(u_t) = \sum_{x_t} \sum_{x_{t-1}} \sum_{\theta} \frac{p(x_t|x_{t-1},\theta,u_t)}{r_{x|xu,t}(x_t|x_{t-1},u_t)}\, \mu_{x_{t-1} \to \mathrm{dyn}_t}(x_{t-1})\, \mu_{\theta \to \mathrm{dyn}_t}(\theta)\, \mu_{x_t \to \mathrm{dyn}_t}(x_t)$$
$$\mu_{\mathrm{dyn}_t \to \theta}(\theta) = \sum_{x_t} \sum_{x_{t-1}} \sum_{u_t} \frac{p(x_t|x_{t-1},\theta,u_t)}{r_{x|xu,t}(x_t|x_{t-1},u_t)}\, \mu_{x_{t-1} \to \mathrm{dyn}_t}(x_{t-1})\, \mu_{u_t \to \mathrm{dyn}_t}(u_t)\, \mu_{x_t \to \mathrm{dyn}_t}(x_t)$$

**Note**: The dynamics channel $r_{x|xu,t}$ does NOT depend on $\theta$. After marginalizing $\theta$ from the transition, the ratio $p/r$ can be precomputed on the reduced $(x_t, x_{t-1}, u_t)$ space.

## 4. Factor Belief Updates

The factor belief at each non-singleton factor is the **AIF-modified kernel times the product of all incoming variable-to-factor messages**:

$$q_{\mathrm{obs},t}(y_t, x_t, \theta) \propto \frac{p(y_t | x_t, \theta)\, r_{y|x\theta,t}^2(y_t | x_t, \theta)}{r_{y|x,t}(y_t | x_t)}\, \mu_{y_t \to \mathrm{obs}_t}(y_t)\, \mu_{x_t \to \mathrm{obs}_t}(x_t)\, \mu_{\theta \to \mathrm{obs}_t}(\theta)$$

$$q_{\mathrm{dyn},t}(x_t, x_{t-1}, \theta, u_t) \propto \frac{p(x_t | x_{t-1}, \theta, u_t)}{r_{x|xu,t}(x_t | x_{t-1}, u_t)}\, \mu_{x_t \to \mathrm{dyn}_t}(x_t)\, \mu_{x_{t-1} \to \mathrm{dyn}_t}(x_{t-1})\, \mu_{\theta \to \mathrm{dyn}_t}(\theta)\, \mu_{u_t \to \mathrm{dyn}_t}(u_t)$$

where $\mu_{i \to a}(s_i) \propto \prod_{b \in \partial i \setminus a} \mu_{b \to i}(s_i)$ is the variable-to-factor message (product of all incoming factor-to-variable messages *except* from factor $a$).

**These factor beliefs are needed to compute the channel updates** (Section 6) via the region beliefs (Section 5). They are the joint distributions over each factor's scope, from which region beliefs are obtained by marginalization.

## 5. Region Beliefs (Intermediate Marginals)

These are derived from factor beliefs by marginalization. Each is **time-indexed**:

| Symbol | Definition | Scope |
| :--- | :--- | :--- |
| $q_{\mathrm{obs},t}(y_t, x_t, \theta)$ | Observation factor belief at time $t$ (from Section 4) | $(y_t, x_t, \theta)$ |
| $q_{\mathrm{dyn},t}(x_t, x_{t-1}, \theta, u_t)$ | Dynamics factor belief at time $t$ (from Section 4) | $(x_t, x_{t-1}, \theta, u_t)$ |
| $q_{yx,t}(y_t, x_t)$ | $= \sum_{\theta} q_{\mathrm{obs},t}(y_t, x_t, \theta)$ | $(y_t, x_t)$ |
| $q_{\mathrm{sep},t}(x_t, \theta)$ | $= \sum_{y_t} q_{\mathrm{obs},t}(y_t, x_t, \theta)$ | $(x_t, \theta)$ |
| $q_{\mathrm{trip},t}(x_t, x_{t-1}, u_t)$ | $= \sum_{\theta} q_{\mathrm{dyn},t}(x_t, x_{t-1}, \theta, u_t)$ | $(x_t, x_{t-1}, u_t)$ |
| $q_{\mathrm{pair},t}(x_{t-1}, u_t)$ | $= \sum_{x_t} q_{\mathrm{trip},t}(x_t, x_{t-1}, u_t)$ | $(x_{t-1}, u_t)$ |

## 6. Channel Update Rules

At each iteration, after computing factor beliefs, update the channels **independently at each time step $t$**:

$$r_{y|x\theta,t}(y_t | x_t, \theta) \leftarrow \frac{q_{\mathrm{obs},t}(y_t, x_t, \theta)}{q_{\mathrm{sep},t}(x_t, \theta)} = q_t(y_t | x_t, \theta)$$
$$r_{y|x,t}(y_t | x_t) \leftarrow \frac{q_{yx,t}(y_t, x_t)}{q_{x_t}(x_t)} = q_t(y_t | x_t)$$
$$r_{x|xu,t}(x_t | x_{t-1}, u_t) \leftarrow \frac{q_{\mathrm{trip},t}(x_t, x_{t-1}, u_t)}{q_{\mathrm{pair},t}(x_{t-1}, u_t)} = q_t(x_t | x_{t-1}, u_t)$$

**Initialization**: Set all three channels to uniform for all $t$ (equivalent to starting from standard BP).

## 7. Singleton Belief Updates

The singleton beliefs follow the **standard sum-product form** — the product of the singleton factor (prior/goal prior) and all incoming factor-to-variable messages:

$$q_{x_t}^*(x_t) \propto \hat{p}_x(x_t)\, \mu_{\mathrm{obs}_t \to x_t}(x_t)\, \mu_{\mathrm{dyn}_t \to x_t}(x_t)\, \mu_{\mathrm{dyn}_{t+1} \to x_t}(x_t)$$
$$q_\theta^*(\theta) \propto p(\theta) \prod_{\tau=1}^{T} \mu_{\mathrm{obs}_\tau \to \theta}(\theta)\, \mu_{\mathrm{dyn}_\tau \to \theta}(\theta)$$
$$q_{u_t}^*(u_t) \propto p(u_t)\, \mu_{\mathrm{dyn}_t \to u_t}(u_t)$$
$$q_{y_t}^*(y_t) \propto \hat{p}_y(y_t)\, \mu_{\mathrm{obs}_t \to y_t}(y_t)$$

**IMPORTANT**: These are standard products. There is NO $1/d_i$ exponent on the beliefs. The Bethe entropy counting numbers $(d_i - 1)$ affect the *derivation* of the stationarity conditions (they determine how Lagrange multipliers combine), but the resulting fixed-point equations are simple products. This is proven in the paper's Appendix (Theorems for $\theta$ and $x_1$, extended to generic $t$ in the generic scheme).

## 8. Degree Counting

Variable degrees determine the Bethe entropy corrections $(d_i - 1) H[q_i]$:

| Variable | Degree | Adjacent factors |
| :--- | :--- | :--- |
| $x_t$ ($1 \leq t \leq T{-}1$) | $d_{x_t} = 4$ | $f_{\mathrm{obs}_t}$, $f_{\mathrm{dyn}_t}$, $f_{\mathrm{dyn}_{t+1}}$, $\hat{p}_x$ |
| $x_0$ | $d_{x_0} = 2$ | $p(x_0)$, $f_{\mathrm{dyn}_1}$ |
| $x_T$ | $d_{x_T} = 3$ | $f_{\mathrm{obs}_T}$, $f_{\mathrm{dyn}_T}$, $\hat{p}_x$ |
| $\theta$ | $d_\theta = 1 + 2T$ | $p(\theta)$, all $f_{\mathrm{obs}_\tau}$, all $f_{\mathrm{dyn}_\tau}$ |
| $u_t$ | $d_{u_t} = 2$ | $p(u_t)$, $f_{\mathrm{dyn}_t}$ |
| $y_t$ | $d_{y_t} = 2$ | $\hat{p}_y(y_t)$, $f_{\mathrm{obs}_t}$ |

### Boundary Conditions
- At $t = 1$: $\mu_{x_0 \to \mathrm{dyn}_1}(x_0) = p(x_0)$ (only prior connects to $x_0$ besides $f_{\mathrm{dyn}_1}$).
- At $t = T$: no factor $f_{\mathrm{dyn}_{T+1}}$ exists, so $\mu_{\mathrm{dyn}_{T+1} \to x_T}$ is absent (set to $1$).

## 9. Iteration Structure

```
Initialize: r_{y|xθ,t} = uniform, r_{y|x,t} = uniform, r_{x|xu,t} = uniform  for all t  (≡ standard BP)

For each iteration:
  1. Compute variable-to-factor messages (standard BP rule: product of
     all incoming factor-to-variable messages except from target factor)
  2. Compute factor-to-variable messages using AIF-modified kernels:
     - obs_t: p(y|x,θ) · r²_{y|xθ,t} / r_{y|x,t}   (time-local channels for time t)
     - dyn_t: p(x|x',θ,u) / r_{x|xu,t}               (time-local channel for time t)
  3. Update singleton beliefs (standard product form)
  4. Compute factor beliefs: AIF-modified kernel × all variable-to-factor messages
     - q_{obs,t} ∝ [p(y|x,θ) · r²_{y|xθ,t} / r_{y|x,t}] · μ_{y→obs} · μ_{x→obs} · μ_{θ→obs}
     - q_{dyn,t} ∝ [p(x|x',θ,u) / r_{x|xu,t}] · μ_{x→dyn} · μ_{x'→dyn} · μ_{θ→dyn} · μ_{u→dyn}
  5. For each time step t, compute region beliefs by marginalization:
     - q_{yx,t} = Σ_θ q_{obs,t}              (marginal observation region)
     - q_{sep,t} = Σ_y q_{obs,t}
     - q_{trip,t} = Σ_θ q_{dyn,t}
     - q_{pair,t} = Σ_x q_{trip,t}
  6. For each time step t, update channels (with damping, Section 10):
     - r_{y|xθ,t} ← q_{obs,t} / q_{sep,t}
     - r_{y|x,t}  ← q_{yx,t} / q_{x_t}
     - r_{x|xu,t} ← q_{trip,t} / q_{pair,t}
  7. Check convergence
```

## 10. Algorithmic Guardrails
- **The $d_\theta$ multiplier**: $d_\theta = 1 + 2T$ grows with $T$. Accumulate the product of $2T$ messages to $\theta$ in log-space to prevent numerical collapse.
- **Channel normalization**: After each channel update at time $t$, verify $\sum_{y_t} r_{y|x\theta,t} = 1$, $\sum_{y_t} r_{y|x,t} = 1$, and $\sum_{x_t} r_{x|xu,t} = 1$. Renormalize if needed.
- **Channel storage**: Maintain arrays of shape `[T, ...]` for each of the three channel types. Channel `t` is indexed and updated independently.
- **Dynamics ratio**: The ratio $p(x_t|x_{t-1},\theta,u_t) / r_{x|xu,t}(x_t|x_{t-1},u_t)$ can produce large values when $r$ is small. Compute in log-space: `log_ratio = log_transition - log_r_x[t]`.
- **Normalization axes**:
    - $q_{\mathrm{sep},t}(x_t, \theta)$: normalize over joint $(x_t, \theta)$ space.
    - $q_{\mathrm{pair},t}(x_{t-1}, u_t)$: normalize over joint $(x_{t-1}, u_t)$ space.
- **Damping**: Apply **arithmetic damping** to all three channels at each update $n$:
    - $r^{n} \propto (1 - \lambda)\, r^{n-1} + \lambda\, r^*$, where $r^*$ is the newly computed channel.
    - Use $\lambda = 0.25$ (recommended; see paper eq:damping).
    - The min-max structure of the objective (dynamics channel in denominator *maximizes* state entropy; observation channel in numerator *minimizes* it) makes the opposing forces on the same belief difficult to stabilize, so damping is critical.
    - Renormalize after damping to maintain valid conditional distributions.

## 11. Verification Checklist
- [ ] Observation kernel uses $p \cdot r_{y|x\theta}^2 / r_{y|x}$ (squared numerator, denominator)?
- [ ] Factor beliefs computed as AIF kernel × all variable-to-factor messages?
- [ ] Setting all three channels to $1$ for all $t$ recovers standard BP messages exactly?
- [ ] Singleton beliefs use simple product form (NO `1/d_i` exponent)?
- [ ] Each time step $t$ has its own channel triple $(r_{y|x\theta,t}, r_{y|x,t}, r_{x|xu,t})$?
- [ ] Channel $r_{x|xu,t}$ does not depend on $\theta$?
- [ ] Channel normalization holds: $\sum_{y_t} r_{y|x\theta,t} = 1$ for all $(x_t, \theta)$, at each $t$?
- [ ] Channel normalization holds: $\sum_{y_t} r_{y|x,t} = 1$ for all $x_t$, at each $t$?
- [ ] Channel normalization holds: $\sum_{x_t} r_{x|xu,t} = 1$ for all $(x_{t-1}, u_t)$, at each $t$?
- [ ] Region belief $q_{yx,t}(y_t, x_t) = \sum_\theta q_{\mathrm{obs},t}$ computed for marginal observation channel update?
- [ ] Region beliefs: $q_{\mathrm{sep},t} = \sum_y q_{\mathrm{obs},t}$, $q_{\mathrm{trip},t} = \sum_\theta q_{\mathrm{dyn},t}$, $q_{\mathrm{pair},t} = \sum_x q_{\mathrm{trip},t}$?
- [ ] Channel update: $r_{y|x,t} \leftarrow q_{yx,t} / q_{x_t}$?
- [ ] Arithmetic damping applied to all 3 channels with $\lambda = 0.25$?
- [ ] Degrees: $d_{x_t} = 4$ (interior), $d_{x_T} = 3$ (final), $d_{x_0} = 2$ (initial), $d_\theta = 1 + 2T$?
- [ ] Dynamics ratio computed in log-space to avoid numerical issues?
- [ ] Boundary: $\mu_{\mathrm{dyn}_{T+1} \to x_T}$ absent or set to $1$?
 