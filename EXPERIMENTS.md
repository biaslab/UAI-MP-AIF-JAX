# Experiment Descriptions

## 1. Environments

### 1.1 Frozen Lake

Our Frozen Lake environment is a grid world where an agent navigates from a start position (top-left) to a goal position (bottom-right) while avoiding hidden holes. It differs from the standard OpenAI Frozen Lake in three ways:

**Multiple configurations (theta).** Instead of a single fixed hole layout, the agent faces one of `n_configs` pre-sampled layouts and must infer which is active. Each configuration is a binary vector over grid cells indicating which cells contain holes. Configurations are generated with a path guarantee (a safe path from start to goal always exists) and a minimum pairwise Hamming distance to ensure diversity. This introduces meta-level partial observability: the agent must simultaneously plan a path and infer the identity of the active hazard map.

**SCAN action.** A fifth action (alongside LEFT, DOWN, RIGHT, UP) that transitions the agent from "unscanned" to "scanned" mode. This doubles the state space to `2 * n_pos` states. The SCAN action costs a timestep but does not move the agent. In scanned mode, all observation channels become near-deterministic (p = 0.999), giving the agent reliable sensory information at the cost of one planning step.

**Distance-dependent observation noise.** The agent receives binary observations from `3 * n_pos` channels at every timestep:
- *Position channels* (`2 * n_pos` channels): One per state (position x scan-mode combination). Near-deterministic (p = 0.999 for the correct state, p = 0.001 otherwise).
- *Grid cell channels* (`n_pos` channels): One per cell, reporting "hole" or "safe". In unscanned mode, the noise on each cell channel increases with the Manhattan distance from the agent to the observed cell: `noise = base_noise + noise_range * dist / max_dist`. In scanned mode, all cell channels become near-deterministic.

This observation structure means the agent has reliable information about nearby cells but noisy information about distant ones, incentivizing either cautious exploration or an early SCAN.

### 1.2 Wumpus World

Our Wumpus World environment is a grid world where an agent navigates from a start position (top-left) to a gold cell while avoiding hidden pits and a wumpus. It differs from the classic AI textbook Wumpus World (Russell & Norvig) in three ways:

**Multiple configurations (theta).** Instead of a single fixed layout, pit positions, wumpus position, and gold position are jointly sampled across `n_configs` configurations. Each configuration specifies a unique combination of hazards and gold. The agent must infer the active configuration from noisy observations, creating the same meta-level partial observability as in Frozen Lake.

**No arrow; SCAN action instead.** The classic "shoot the wumpus" action is removed. In its place, the agent has a SCAN action (identical to Frozen Lake): it transitions from unscanned to scanned mode, making all observations near-deterministic at the cost of one timestep. The action space is LEFT, DOWN, RIGHT, UP, SCAN.

**Stochastic indirect observations.** In the classic version, breeze/stench/glitter are deterministic binary signals. In our version, all observation channels are noisy:
- *Feature channels* (3 channels): breeze (adjacent to a pit), stench (adjacent to the wumpus), and glitter (on the gold cell). In unscanned mode: true-positive rate `1 - obs_noise`, false-positive rate `obs_noise * 0.1`. In scanned mode: near-deterministic (p_tp = 0.999, p_fp = 0.001).
- *Position channels* (`n_pos` channels): One per grid cell, indicating agent presence. In unscanned mode: true-positive rate `1 - pos_noise`, false-positive rate `pos_noise * 0.1`. In scanned mode: near-deterministic.

The combination of indirect hazard cues (breeze/stench rather than direct pit/wumpus observation), noisy channels, and multiple possible configurations makes this a challenging inference-and-planning problem.

---

## 2. Experimental Setup

### 2.1 Frozen Lake

**Environment parameters:**

| Parameter | Value |
|---|---|
| Grid size | 4 x 4 (16 positions, 32 states with scan mode) |
| Configurations (theta) | 15 |
| Hole fraction | 0.2 |
| Min Hamming distance | 4 |
| Base observation noise | 0.4 |
| Noise range | 0.1 |
| Slip probability | 0.1 |
| Hole penalty (goal prior) | 2.0 |
| Goal temperature | 1.0 |
| SCAN cost (action prior weight) | 0.1 |

**Experiment parameters:**

| Parameter | Value |
|---|---|
| Episodes | 1000 |
| Max steps per episode | 15 |
| Planning horizon | 15 |
| Seed | 0 |

**Methods compared (7):**

| Method | Iterations | Damping |
|---|---|---|
| BP | 1 | 1.0 |
| Loopy VBP | 30 | 1.0 |
| Loopy BP | 30 | 1.0 |
| Region-extended | 50 | 0.25 |
| Reduced region-extended | 30 | 0.3 |
| Dyn-channel | 25 | 0.25 |
| Nuijten | 30 | 1.0 |

The action prior is `[1, 1, 1, 1, scan_cost]` normalized, making SCAN relatively cheap (scan_cost = 0.1) to encourage information gathering. Each episode samples a random configuration; the agent starts unscanned at position 0 and terminates upon reaching the goal, falling into a hole, or exceeding max steps.

### 2.2 Wumpus World

**Environment parameters:**

| Parameter | Value |
|---|---|
| Grid size | 5 x 5 (25 positions, 50 states with scan mode) |
| Configurations (theta) | 25 |
| Pits per configuration | 4 |
| Feature observation noise | 0.1 |
| Position observation noise | 0.4 |
| Slip probability | 0.01 |
| Pit penalty (goal prior) | 1.0 |
| Wumpus penalty (goal prior) | 1.0 |
| Goal temperature | 1.0 |
| SCAN cost (action prior weight) | 0.7 |

**Experiment parameters:**

| Parameter | Value |
|---|---|
| Episodes | 1000 |
| Max steps per episode | 10 |
| Planning horizon | 7 |
| Global damping | 0.25 |
| Seed | 0 |

**Methods compared (6):**

| Method | Iterations |
|---|---|
| Loopy VBP | 30 |
| Loopy BP | 30 |
| Region-extended | 50 |
| Reduced region-extended | 30 |
| Dyn-channel | 50 |
| Nuijten | 30 |

All Wumpus World methods use a shared damping factor of 0.25. The action prior is `[1, 1, 1, 1, scan_cost]` normalized, with scan_cost = 0.7 making SCAN relatively more expensive than in Frozen Lake. Each episode samples a random configuration; the agent starts unscanned at position 0 and terminates upon reaching the gold, stepping on a pit or the wumpus, or exceeding max steps. Stepping on the gold yields reward +1; stepping on a pit or the wumpus yields reward -1.

---

## 3. Metrics

Both experiments report three metrics per method:

- **Success rate**: Fraction of episodes where the agent reached the goal (Frozen Lake) or the gold (Wumpus World).
- **Average steps**: Mean number of steps taken per episode (including failures and truncations).
- **Average reward**: Mean cumulative reward per episode. In Frozen Lake, reward is +1 for reaching the goal and 0 otherwise. In Wumpus World, reward is +1 for gold, -1 for pit/wumpus, and 0 otherwise.
