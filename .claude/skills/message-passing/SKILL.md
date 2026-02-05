---
name: discrete-factor-graph-implementer
description: Expert in implementing discrete message-passing schemes using specific JAX primitives. Translates LaTeX update rules into indexed or tensor-based contractions for state-space models and planning.
allowed-tools: Read, Write, Edit
---

# Discrete Message Passing Implementation Protocol

You are optimized to implement inference loops for discrete factor graphs using a specific library of JAX primitives. You operate primarily on **Categorical distributions** and **Transition Tensors/Indices**.

## 1. Core Primitive Mapping

Use the following table to map LaTeX mathematical expressions to the provided source code functions:

| LaTeX Expression | Logic Type | Recommended Function |
| :--- | :--- | :--- |
| $\mu \propto \sum_{j,k,l} T_{i,j,k,l} \cdot q_j \cdot q_k \cdot a_l$ | **Forward (Indexed)** | `forward_message_indexed(...)` |
| $\mu \propto \sum_{j,k,l} T_{i,j,k,l} \cdot q_j \cdot q_k \cdot a_l$ | **Forward (Dense)** | `forward_message_4d(...)` |
| $\prod_{fov} P(y_{fov} \mid x, s)$ | **Obs Likelihood** | `backward_obs_message_indexed(...)` |
| $\mu_{f \to s} \propto \sum_{new, old} T \cdot q_{new} \cdot q_{old}$ | **Static Update** | `transition_message_to_static_indexed(...)` |
| $q(x) \propto \prod \mu_i$ | **Belief (Prob)** | `combine_messages(list_of_probs)` |
| $q(x) \propto \exp(\sum \log \mu_i)$ | **Belief (Log)** | `combine_messages_log(list_of_logs)` |

## 2. Decision Logic: Indexed vs. Dense
When the generative model code defines a transition:
1.  **Use Indexed Primitives** (`_indexed`) if the transition $T$ is provided as a mapping of indices (e.g., `next_state_idx`). This is preferred for memory efficiency in large state spaces (like MiniGrid).
2.  **Use Tensor Primitives** (`_2d/3d/4d`) if the transition $T$ is provided as a dense probability/contingency table.

## 3. Implementation Pipeline

### Phase 1: Variable & Factor Extraction
* **Edges:** Identify `q_state`, `q_static`, `q_action`.
* **Nodes:** Identify `Transition` (between time steps) and `Observation` (likelihood factors).

### Phase 2: Implementation of the Scheme
1.  **Forward Pass (Prediction):** * Translate $\mu_{f \to x_t}$ into `forward_message_indexed`. 
    * *Critical:* Always pass the `action_idx` as an integer/scalar if the scheme implies a known control input.
2.  **Backward Pass (Inference/Planning):**
    * For observations, use `backward_obs_message_indexed` which internally handles the log-sum-exp over the field-of-view (49 positions).
    * If the scheme includes a "Goal" term $G(x_T)$, ensure it is treated as an incoming message at the final time step.
3.  **Static Variable Updates:**
    * The `static_state` (e.g., map layout) often receives messages from *every* observation and *every* transition. Use `backward_obs_message_to_static_indexed` and `transition_message_to_static_indexed`.

## 4. Numerical & Algorithmic Guardrails
* **Log-Space Stability:** For observation-heavy models, always accumulate messages in log-space and use `combine_messages_log` with the provided `EPSILON` to prevent underflow.
* **Normalization:** All `forward_` functions in the library already normalize. However, when combining messages to form a marginal $q$, you MUST use `combine_messages` to apply the final `softmax`.
* **Einsum Strings:** When using dense tensors, strictly follow:
    * 3D: `'ijk,j,k->i'` (out, in1, in2)
    * 4D: `'ijkl,j,k,l->i'` (out, in1, in2, in3)
    * Backward 3D to State: `'ijk,i,k->j'`