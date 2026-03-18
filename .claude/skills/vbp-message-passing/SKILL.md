---
name: vbp-bethe-passer
description: Converts a standard Bethe BP implementation to the VBP (cross-entropy planning) message-passing scheme. Specifies channel reparameterization, modified dynamics kernel, region beliefs, and correct fixed-point equations as derived in the paper.
---

# VBP: Converting Standard BP to the Cross-Entropy Planning Scheme

You are an expert in converting a **standard Bethe Belief Propagation** implementation into the **Variational Belief Propagation (VBP) message-passing scheme** as derived in the paper (Appendix, Section "Message Passing Derivation for VBP"). VBP is standard Bethe BP with one modification: a single entropy correction reparameterized into the dynamics kernel via one channel variable per time step. This is substantially simpler than the AIF scheme (which uses three channels).

## 1. What VBP Changes vs Standard Bethe BP

The VBP objective is:
$$F_{\mathrm{VBP}}[q, r_{u|x}] = F_{\mathrm{Bethe}}[q] + \Delta F_{\mathrm{VBP}}$$

where the entropic correction is (eq:cross_entropy_correction):
$$\Delta F_{\mathrm{VBP}} = +\sum_{t=1}^{T} H[q(u_t | x_{t-1})]$$

**Net effect**: VBP adds a single correction per time step — an action entropy penalty that encourages the agent to commit to a policy $q(u_t | x_{t-1})$. This implements **cross-entropy planning**: the agent maximizes expected reward while committing to actions.

## 2. The Conversion Rule: Channel Reparameterization

To make the entropy correction tractable, introduce a **single time-local channel variable** per time step:

| Channel | Symbol | Normalization | Recovers at fixed point |
| :--- | :--- | :--- | :--- |
| Action (time $t$) | $r_{u\mid x,t}(u_t \mid x_{t-1})$ | $\sum_{u_t} r = 1\;\forall\,x_{t-1}$ | $q_t(u_t \mid x_{t-1})$ |

**Key identity**: $H[q(u_t | x_{t-1})] = \min_{r_{u|x}} \mathbb{E}_{q_{\mathrm{pair}}}[-\log r_{u|x}(u_t | x_{t-1})]$, achieved when $r_{u|x} = q(u_t | x_{t-1})$.

**Why the channel enters the numerator**: The correction $+H[q(u|x)]$ carries a positive sign, so after channel reparameterization it contributes $-\mathbb{E}_q[\log r_{u|x}]$ to the objective. Exponentiating gives $r_{u|x}$ in the **numerator** of the dynamics kernel (multiplicative), in contrast to AIF's dynamics channel $r_{x|xu}$ which appears in the **denominator**.

### The Central Modification

Replace the dynamics kernel used in standard BP **at each time step $t$**:

| Factor | Standard BP kernel | VBP kernel |
| :--- | :--- | :--- |
| $f_{\mathrm{obs}_t}$ | $p(y_t \mid x_t, \theta)$ | $p(y_t \mid x_t, \theta)$ — **unmodified** |
| $f_{\mathrm{dyn}_t}$ | $p(x_t \mid x_{t-1}, \theta, u_t)$ | $p(x_t \mid x_{t-1}, \theta, u_t) \cdot r_{u\mid x,t}(u_t \mid x_{t-1})$ |

**Setting $r_{u|x,t} \equiv 1$ (uniform) for all $t$ recovers standard Bethe BP exactly.** All other BP mechanics (variable-to-factor messages, belief updates, convergence) remain unchanged.

## 3. Generic Message Equations (Time Step $t$)

Using $\mu_{a \to i}$ for factor-to-variable and $\mu_{i \to a}$ for variable-to-factor messages, where:
$$\mu_{i \to a}(s_i) \propto \prod_{b \in \partial i \setminus a} \mu_{b \to i}(s_i)$$

### Messages from Observation Factor $f_{\mathrm{obs}_t}$

Standard sum-product — **no channel modification**:
$$\mu_{\mathrm{obs}_t \to \theta}(\theta) = \sum_{y_t} \sum_{x_t} p(y_t | x_t, \theta)\, \mu_{y_t \to \mathrm{obs}_t}(y_t)\, \mu_{x_t \to \mathrm{obs}_t}(x_t)$$
$$\mu_{\mathrm{obs}_t \to x_t}(x_t) = \sum_{y_t} \sum_{\theta} p(y_t | x_t, \theta)\, \mu_{y_t \to \mathrm{obs}_t}(y_t)\, \mu_{\theta \to \mathrm{obs}_t}(\theta)$$
$$\mu_{\mathrm{obs}_t \to y_t}(y_t) = \sum_{x_t} \sum_{\theta} p(y_t | x_t, \theta)\, \mu_{\theta \to \mathrm{obs}_t}(\theta)\, \mu_{x_t \to \mathrm{obs}_t}(x_t)$$

### Messages from Dynamics Factor $f_{\mathrm{dyn}_t}$

Using kernel $p(x_t | x_{t-1}, \theta, u_t) \cdot r_{u|x,t}(u_t | x_{t-1})$:
$$\mu_{\mathrm{dyn}_t \to \theta}(\theta) = \sum_{x_t} \sum_{x_{t-1}} \sum_{u_t} p(x_t | x_{t-1}, \theta, u_t)\, r_{u|x,t}(u_t | x_{t-1})\, \mu_{x_{t-1} \to \mathrm{dyn}_t}(x_{t-1})\, \mu_{u_t \to \mathrm{dyn}_t}(u_t)\, \mu_{x_t \to \mathrm{dyn}_t}(x_t)$$
$$\mu_{\mathrm{dyn}_t \to x_t}(x_t) = \sum_{x_{t-1}} \sum_{u_t} \sum_{\theta} p(x_t | x_{t-1}, \theta, u_t)\, r_{u|x,t}(u_t | x_{t-1})\, \mu_{x_{t-1} \to \mathrm{dyn}_t}(x_{t-1})\, \mu_{u_t \to \mathrm{dyn}_t}(u_t)\, \mu_{\theta \to \mathrm{dyn}_t}(\theta)$$
$$\mu_{\mathrm{dyn}_t \to x_{t-1}}(x_{t-1}) = \sum_{x_t} \sum_{u_t} \sum_{\theta} p(x_t | x_{t-1}, \theta, u_t)\, r_{u|x,t}(u_t | x_{t-1})\, \mu_{x_t \to \mathrm{dyn}_t}(x_t)\, \mu_{u_t \to \mathrm{dyn}_t}(u_t)\, \mu_{\theta \to \mathrm{dyn}_t}(\theta)$$
$$\mu_{\mathrm{dyn}_t \to u_t}(u_t) = \sum_{x_t} \sum_{x_{t-1}} \sum_{\theta} p(x_t | x_{t-1}, \theta, u_t)\, r_{u|x,t}(u_t | x_{t-1})\, \mu_{x_{t-1} \to \mathrm{dyn}_t}(x_{t-1})\, \mu_{\theta \to \mathrm{dyn}_t}(\theta)\, \mu_{x_t \to \mathrm{dyn}_t}(x_t)$$

**Note**: The channel $r_{u|x,t}$ depends on $(u_t, x_{t-1})$. In $\mu_{\mathrm{dyn}_t \to x_{t-1}}$, since $x_{t-1}$ is the output variable, $r_{u|x,t}(u_t | x_{t-1})$ remains inside the sum over $u_t$ but depends on the fixed $x_{t-1}$. In $\mu_{\mathrm{dyn}_t \to \theta}$ and $\mu_{\mathrm{dyn}_t \to x_t}$, both $u_t$ and $x_{t-1}$ are summed over, so $r_{u|x,t}$ cannot be factored out.

## 4. Factor Belief Updates

The factor belief at each non-singleton factor is the **VBP kernel times the product of all incoming variable-to-factor messages**:

$$q_{\mathrm{obs},t}(y_t, x_t, \theta) \propto p(y_t | x_t, \theta)\, \mu_{y_t \to \mathrm{obs}_t}(y_t)\, \mu_{x_t \to \mathrm{obs}_t}(x_t)\, \mu_{\theta \to \mathrm{obs}_t}(\theta)$$

$$q_{\mathrm{dyn},t}(x_t, x_{t-1}, \theta, u_t) \propto p(x_t | x_{t-1}, \theta, u_t)\, r_{u|x,t}(u_t | x_{t-1})\, \mu_{x_t \to \mathrm{dyn}_t}(x_t)\, \mu_{x_{t-1} \to \mathrm{dyn}_t}(x_{t-1})\, \mu_{\theta \to \mathrm{dyn}_t}(\theta)\, \mu_{u_t \to \mathrm{dyn}_t}(u_t)$$

where $\mu_{i \to a}(s_i) \propto \prod_{b \in \partial i \setminus a} \mu_{b \to i}(s_i)$ is the variable-to-factor message (product of all incoming factor-to-variable messages *except* from factor $a$).

**The dynamics factor belief is needed to compute the channel update** (Section 6) via the region belief (Section 5).

## 5. Region Beliefs (Intermediate Marginal)

VBP requires only one region belief, derived from the dynamics factor belief by marginalization:

| Symbol | Definition | Scope |
| :--- | :--- | :--- |
| $q_{\mathrm{pair},t}(x_{t-1}, u_t)$ | $= \sum_{x_t} \sum_{\theta} q_{\mathrm{dyn},t}(x_t, x_{t-1}, \theta, u_t)$ | $(x_{t-1}, u_t)$ |

No observation-side region beliefs ($q_{yx,t}$, $q_{\mathrm{sep},t}$) are needed since the observation kernel is unmodified.

## 6. Channel Update Rule

At each iteration, after computing the dynamics factor belief, update the channel **independently at each time step $t$**:

$$r_{u|x,t}(u_t | x_{t-1}) \leftarrow \frac{q_{\mathrm{pair},t}(x_{t-1}, u_t)}{q_{x_{t-1}}(x_{t-1})} = q_t(u_t | x_{t-1})$$

where $q_{x_{t-1}}(x_{t-1}) = \sum_{u_t} q_{\mathrm{pair},t}(x_{t-1}, u_t)$.

**Initialization**: Set the channel to uniform for all $t$ (equivalent to starting from standard BP).

## 7. Singleton Belief Updates

The singleton beliefs follow the **standard sum-product form** — the product of the singleton factor (prior/goal prior) and all incoming factor-to-variable messages:

$$q_{x_t}^*(x_t) \propto \hat{p}_x(x_t)\, \mu_{\mathrm{obs}_t \to x_t}(x_t)\, \mu_{\mathrm{dyn}_t \to x_t}(x_t)\, \mu_{\mathrm{dyn}_{t+1} \to x_t}(x_t)$$
$$q_\theta^*(\theta) \propto p(\theta) \prod_{\tau=1}^{T} \mu_{\mathrm{obs}_\tau \to \theta}(\theta)\, \mu_{\mathrm{dyn}_\tau \to \theta}(\theta)$$
$$q_{u_t}^*(u_t) \propto p(u_t)\, \mu_{\mathrm{dyn}_t \to u_t}(u_t)$$
$$q_{y_t}^*(y_t) \propto \hat{p}_y(y_t)\, \mu_{\mathrm{obs}_t \to y_t}(y_t)$$

**IMPORTANT**: These are standard products. There is NO $1/d_i$ exponent on the beliefs. The Bethe entropy counting numbers $(d_i - 1)$ affect the *derivation* of the stationarity conditions, but the resulting fixed-point equations are simple products.

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
Initialize: r_{u|x,t} = uniform  for all t  (≡ standard BP)

For each iteration:
  1. Compute variable-to-factor messages (standard BP rule: product of
     all incoming factor-to-variable messages except from target factor)
  2. Compute factor-to-variable messages:
     - obs_t: p(y|x,θ)                        (standard sum-product, no channel)
     - dyn_t: p(x|x',θ,u) · r_{u|x,t}(u|x')  (channel multiplies kernel)
  3. Update singleton beliefs (standard product form)
  4. Compute factor beliefs:
     - q_{obs,t} ∝ p(y|x,θ) · μ_{y→obs} · μ_{x→obs} · μ_{θ→obs}         (standard)
     - q_{dyn,t} ∝ p(x|x',θ,u) · r_{u|x,t}(u|x') · μ_{x→dyn} · μ_{x'→dyn} · μ_{θ→dyn} · μ_{u→dyn}
  5. Compute region belief:
     - q_{pair,t}(x_{t-1}, u_t) = Σ_{x_t} Σ_θ q_{dyn,t}
  6. Update channel:
     - r_{u|x,t} ← q_{pair,t}(x_{t-1}, u_t) / q_{x_{t-1}}(x_{t-1})
  7. Check convergence
```

## 10. Algorithmic Guardrails

- **The $d_\theta$ multiplier**: $d_\theta = 1 + 2T$ grows with $T$. Accumulate the product of $2T$ messages to $\theta$ in log-space to prevent numerical collapse.
- **Channel normalization**: After each channel update at time $t$, verify $\sum_{u_t} r_{u|x,t} = 1$ for all $x_{t-1}$. Renormalize if needed.
- **Channel storage**: Maintain arrays of shape `[T, ...]` for the action channel. Channel at time $t$ is indexed and updated independently.
- **No dynamics ratio issue**: Unlike AIF where the dynamics channel appears in the denominator (causing large values when $r$ is small), VBP's channel **multiplies** the kernel. This avoids the numerical instability of dividing by small channel values.
- **No min-max structure**: VBP has a single channel entering only the dynamics kernel numerator. The joint optimization over $(q, r_{u|x})$ is a **pure minimization** problem, avoiding the convergence difficulties of AIF's saddle-point structure.
- **Damping**: While less critical than for AIF (no opposing channels), arithmetic damping may still improve convergence in practice:
    - $r^{n} \propto (1 - \lambda)\, r^{n-1} + \lambda\, r^*$, where $r^*$ is the newly computed channel.
    - Renormalize after damping to maintain a valid conditional distribution.

## 11. Verification Checklist

- [ ] Observation messages are standard sum-product (no channel modification)?
- [ ] Dynamics kernel uses $p(x|x',\theta,u) \cdot r_{u|x,t}(u|x')$ (channel in **numerator**, not denominator)?
- [ ] Factor beliefs: observation is standard $p \cdot \text{messages}$, dynamics is $p \cdot r_{u|x,t} \cdot \text{messages}$?
- [ ] Setting $r_{u|x,t} \equiv 1$ (uniform) for all $t$ recovers standard BP exactly?
- [ ] Each time step $t$ has its own channel $r_{u|x,t}$?
- [ ] Channel normalization holds: $\sum_{u_t} r_{u|x,t} = 1$ for all $x_{t-1}$, at each $t$?
- [ ] Region belief: $q_{\mathrm{pair},t}(x_{t-1}, u_t) = \sum_{x_t} \sum_\theta q_{\mathrm{dyn},t}$?
- [ ] Channel update: $r_{u|x,t} \leftarrow q_{\mathrm{pair},t} / q_{x_{t-1}}$?
- [ ] Singleton beliefs use simple product form (NO $1/d_i$ exponent)?
- [ ] Degrees: $d_{x_t} = 4$ (interior), $d_{x_T} = 3$ (final), $d_{x_0} = 2$ (initial), $d_\theta = 1 + 2T$?
- [ ] Boundary: $\mu_{\mathrm{dyn}_{T+1} \to x_T}$ absent or set to $1$?

## 12. Comparison: AIF vs VBP

| Aspect | AIF | VBP |
| :--- | :--- | :--- |
| **Channels per $t$** | 3: $r_{y\mid x\theta}$, $r_{y\mid x}$, $r_{x\mid xu}$ | 1: $r_{u\mid x}$ |
| **Observation kernel** | $p \cdot r_{y\mid x\theta}^2 / r_{y\mid x}$ (modified) | $p(y \mid x, \theta)$ (standard) |
| **Dynamics kernel** | $p / r_{x\mid xu}$ (channel in denominator) | $p \cdot r_{u\mid x}$ (channel in numerator) |
| **Region beliefs needed** | $q_{yx}$, $q_{\mathrm{sep}}$, $q_{\mathrm{trip}}$, $q_{\mathrm{pair}}$ | $q_{\mathrm{pair}}$ only |
| **Optimization structure** | Min-max (opposing channels) | Pure minimization |
| **Convergence** | Harder (saddle-point, damping critical) | Easier (no opposing forces) |
| **Fixed-point interpretation** | Multiple conditionals recovered | $r_{u\mid x}^* = q(u \mid x')$: reweights transition by policy |
| **Reduction to standard BP** | Set all 3 channels to $1$ | Set $r_{u\mid x} = 1$ (uniform) |
