---
name: discrete-factor-graph-implementer
description: Expert in implementing discrete message-passing schemes using specific JAX primitives. Translates LaTeX update rules into indexed or tensor-based contractions for state-space models and planning.
allowed-tools: Read, Write, Edit
---

# Discrete Message Passing Implementation Protocol

You are optimized to implement inference loops for discrete factor graphs using a specific library of JAX primitives. You operate primarily on **Categorical distributions** and **Transition Tensors/Indices**.

## 1. Notation

Following the paper's conventions:
- **Factor-to-variable messages**: $\mu_{a \to i}(s_i)$ — message from factor $a$ to variable $i$
- **Variable-to-factor messages**: $\mu_{i \to a}(s_i) \propto \prod_{b \in \partial i \setminus a} \mu_{b \to i}(s_i)$
- **Singleton beliefs**: $q_i^*(s_i) \propto f_i(s_i) \cdot \prod_{a \in \partial i,\, \text{non-singleton}} \mu_{a \to i}(s_i)$, where $f_i$ is the singleton factor (prior or goal prior)
- **Factor names**: $f_{\mathrm{obs}_t}$ (observation at time $t$), $f_{\mathrm{dyn}_t}$ (dynamics at time $t$)

## 2. Core Primitive Mapping

Use the following table to map LaTeX mathematical expressions to the provided source code functions:

| LaTeX Expression | Logic Type | Recommended Function |
| :--- | :--- | :--- |
| $\mu \propto \sum_{j,k,l} T_{i,j,k,l} \cdot q_j \cdot q_k \cdot a_l$ | **Forward (Indexed)** | `forward_message_indexed(...)` |
| $\mu \propto \sum_{j,k,l} T_{i,j,k,l} \cdot q_j \cdot q_k \cdot a_l$ | **Forward (Dense)** | `forward_message_4d(...)` |
| $\prod_{fov} P(y_{fov} \mid x, s)$ | **Obs Likelihood** | `backward_obs_message_indexed(...)` |
| $\mu_{f \to s} \propto \sum_{new, old} T \cdot q_{new} \cdot q_{old}$ | **Static Update** | `transition_message_to_static_indexed(...)` |
| $q(x) \propto \prod \mu_i$ | **Belief (Prob)** | `combine_messages(list_of_probs)` |
| $q(x) \propto \exp(\sum \log \mu_i)$ | **Belief (Log)** | `combine_messages_log(list_of_logs)` |

## 3. AIF Factor Kernel Modification

When implementing Active Inference (as opposed to standard BP), the factor kernels are modified by **channel variables** $r_{y|x\theta}$ and $r_{x|xu}$:

| Factor | Standard BP kernel | AIF-modified kernel |
| :--- | :--- | :--- |
| $f_{\mathrm{obs}_t}$ | $p(y_t \mid x_t, \theta)$ | $p(y_t \mid x_t, \theta) \cdot r_{y \mid x\theta}(y_t \mid x_t, \theta)$ |
| $f_{\mathrm{dyn}_t}$ | $p(x_t \mid x_{t-1}, \theta, u_t)$ | $p(x_t \mid x_{t-1}, \theta, u_t) \;/\; r_{x \mid xu}(x_t \mid x_{t-1}, u_t)$ |

**Setting $r_{y|x\theta} \equiv 1$ and $r_{x|xu} \equiv 1$ recovers standard Bethe BP.**

### In log-space:
- **Observation**: add `log_r_y` to `log_likelihood` before computing messages.
- **Dynamics**: subtract `log_r_x` from `log_transition` before computing messages. The dynamics channel does NOT depend on $\theta$, so compute the ratio on the reduced $(x_t, x_{t-1}, u_t)$ space.

## 4. Decision Logic: Indexed vs. Dense
When the generative model code defines a transition:
1.  **Use Indexed Primitives** (`_indexed`) if the transition $T$ is provided as a mapping of indices (e.g., `next_state_idx`). This is preferred for memory efficiency in large state spaces (like MiniGrid).
2.  **Use Tensor Primitives** (`_2d/3d/4d`) if the transition $T$ is provided as a dense probability/contingency table.

## 5. Implementation Pipeline

### Phase 1: Variable & Factor Extraction
* **Edges (variables):** Identify `q_state` ($x_t$), `q_static` ($\theta$), `q_action` ($u_t$), `q_obs` ($y_t$).
* **Nodes (factors):** Identify `Transition` ($f_{\mathrm{dyn}_t}$) and `Observation` ($f_{\mathrm{obs}_t}$) factor nodes.

### Phase 2: Message Computation
1.  **Forward messages** ($\mu_{\mathrm{dyn}_t \to x_t}$): Compute using `forward_message_indexed` with the (AIF-modified) dynamics kernel.
    * *Critical:* Always pass the `action_idx` as an integer/scalar if the scheme implies a known control input.
2.  **Backward messages** ($\mu_{\mathrm{obs}_t \to x_t}$, $\mu_{\mathrm{dyn}_t \to x_{t-1}}$):
    * For observations, use `backward_obs_message_indexed` with the (AIF-modified) observation kernel.
    * If the scheme includes a goal prior $\hat{p}_x(x_T)$, treat it as an incoming message at the final time step.
3.  **Static variable updates** ($\mu_{\mathrm{obs}_t \to \theta}$, $\mu_{\mathrm{dyn}_t \to \theta}$):
    * The parameter $\theta$ receives messages from *every* observation and *every* dynamics factor. Use `backward_obs_message_to_static_indexed` and `transition_message_to_static_indexed`.

### Phase 3: Channel Updates (AIF only)
After computing factor beliefs, update channels (see aif-message-passing skill for details):
* $r_{y|x\theta} \leftarrow q_{\mathrm{obs},t} / q_{\mathrm{sep},t}$
* $r_{x|xu} \leftarrow q_{\mathrm{trip},t} / q_{\mathrm{pair},t}$

## 6. Numerical & Algorithmic Guardrails
* **Log-Space Stability:** For observation-heavy models, always accumulate messages in log-space and use `combine_messages_log` with the provided `EPSILON` to prevent underflow.
* **Normalization:** All `forward_` functions in the library already normalize. However, when combining messages to form a marginal $q$, you MUST use `combine_messages` to apply the final `softmax`.
* **Einsum Strings:** When using dense tensors, strictly follow:
    * 3D: `'ijk,j,k->i'` (out, in1, in2)
    * 4D: `'ijkl,j,k,l->i'` (out, in1, in2, in3)
    * Backward 3D to State: `'ijk,i,k->j'`
