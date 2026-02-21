"""Wumpus World environment with hidden pits and wumpus.

Grid world where the agent must reach the gold while avoiding hidden pits and
the wumpus. The agent perceives breeze (adjacent to pit), stench (adjacent to
wumpus), and glitter (on the gold cell). Static state θ represents sampled
(pits, wumpus, gold) configurations.

State space:
    x: agent position (row * grid_size + col), flat index 0..n_states-1
    θ: configuration index, one of k pre-sampled configs

Actions:
    0: LEFT, 1: DOWN, 2: RIGHT, 3: UP

Observations:
    3 binary channels:
        - channel 0: breeze  (0=no, 1=yes) — adjacent to a pit
        - channel 1: stench  (0=no, 1=yes) — adjacent to the wumpus
        - channel 2: glitter (0=no, 1=yes) — on the gold cell
    Observation tensor shape: (3, 2, n_states, n_static)
"""

import numpy as np
from dataclasses import dataclass

# Actions (same as Frozen Lake)
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
N_ACTIONS = 4

# Observation channels
BREEZE = 0
STENCH = 1
GLITTER = 2
N_OBS_CHANNELS = 3
N_OBS_TYPES = 2  # binary: 0 or 1

# Movement deltas: (delta_row, delta_col)
MOVEMENT = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}


def pos_to_rc(pos: int, grid_size: int) -> tuple[int, int]:
    return divmod(pos, grid_size)


def rc_to_pos(row: int, col: int, grid_size: int) -> int:
    return row * grid_size + col


def get_neighbors(pos: int, grid_size: int) -> list[int]:
    """Get valid orthogonal neighbors of a position."""
    row, col = pos_to_rc(pos, grid_size)
    neighbors = []
    for dr, dc in MOVEMENT.values():
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            neighbors.append(rc_to_pos(nr, nc, grid_size))
    return neighbors


def sample_configs(
    grid_size: int,
    n_configs: int,
    n_pits: int = 2,
    seed: int = 42,
    start_pos: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample random Wumpus World configurations.

    Each config has: pit positions, wumpus position, gold position.
    Start position is always safe (no pit, no wumpus, no gold).

    Args:
        grid_size: Grid size
        n_configs: Number of configurations to sample
        n_pits: Number of pits per configuration
        seed: Random seed
        start_pos: Start position (always safe)

    Returns:
        pits: (n_configs, n_states) binary. pits[θ, pos] = 1 if pit at pos.
        wumpus: (n_configs, n_states) binary. wumpus[θ, pos] = 1 if wumpus at pos.
        gold: (n_configs, n_states) binary. gold[θ, pos] = 1 if gold at pos.
    """
    n_states = grid_size * grid_size
    rng = np.random.default_rng(seed)

    candidates = [i for i in range(n_states) if i != start_pos]

    pits = np.zeros((n_configs, n_states), dtype=np.float32)
    wumpus = np.zeros((n_configs, n_states), dtype=np.float32)
    gold = np.zeros((n_configs, n_states), dtype=np.float32)

    configs_seen = set()
    idx = 0
    max_attempts = n_configs * 100
    attempts = 0

    while idx < n_configs and attempts < max_attempts:
        attempts += 1

        # Sample n_pits + 2 unique positions (pits, wumpus, gold) from candidates
        chosen = rng.choice(candidates, size=n_pits + 2, replace=False)
        pit_positions = tuple(sorted(chosen[:n_pits]))
        wumpus_pos = int(chosen[n_pits])
        gold_pos = int(chosen[n_pits + 1])

        config_key = (pit_positions, wumpus_pos, gold_pos)
        if config_key in configs_seen:
            continue
        configs_seen.add(config_key)

        for p in pit_positions:
            pits[idx, p] = 1.0
        wumpus[idx, wumpus_pos] = 1.0
        gold[idx, gold_pos] = 1.0
        idx += 1

    if idx < n_configs:
        raise ValueError(
            f"Could only generate {idx}/{n_configs} unique configs. "
            f"Reduce n_configs or increase grid_size/n_pits."
        )

    return pits, wumpus, gold


def generate_transition_tensor(
    grid_size: int,
    pits: np.ndarray,
    wumpus: np.ndarray,
    slip_prob: float = 0.0,
) -> np.ndarray:
    """Generate transition tensor T(x_new, x_old, θ, action).

    Pits and wumpus cells are absorbing (death).

    Args:
        grid_size: Grid size
        pits: (n_configs, n_states) pit configurations
        wumpus: (n_configs, n_states) wumpus configurations
        slip_prob: Movement noise probability

    Returns:
        T: (n_states, n_states, n_static, n_actions)
    """
    n_states = grid_size * grid_size
    n_static = pits.shape[0]

    T = np.zeros((n_states, n_states, n_static, N_ACTIONS), dtype=np.float32)

    for theta in range(n_static):
        for x_old in range(n_states):
            # Absorbing: pits and wumpus
            if pits[theta, x_old] == 1.0 or wumpus[theta, x_old] == 1.0:
                T[x_old, x_old, theta, :] = 1.0
                continue

            for intended_action in range(N_ACTIONS):
                for actual_action in range(N_ACTIONS):
                    if actual_action == intended_action:
                        prob = 1.0 - slip_prob
                    else:
                        prob = slip_prob / (N_ACTIONS - 1)

                    if prob == 0.0:
                        continue

                    row, col = pos_to_rc(x_old, grid_size)
                    dr, dc = MOVEMENT[actual_action]
                    new_row, new_col = row + dr, col + dc

                    if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                        x_new = rc_to_pos(new_row, new_col, grid_size)
                    else:
                        x_new = x_old

                    T[x_new, x_old, theta, intended_action] += prob

    return T


def generate_observation_tensor(
    grid_size: int,
    pits: np.ndarray,
    wumpus: np.ndarray,
    gold: np.ndarray,
    obs_noise: float = 0.1,
) -> np.ndarray:
    """Generate observation tensor B(channel, obs_type, x, θ).

    3 binary observation channels with noise:
        - breeze:  P(breeze=1 | x, θ) = p_tp if adjacent to pit, p_fp otherwise
        - stench:  P(stench=1 | x, θ) = p_tp if adjacent to wumpus, p_fp otherwise
        - glitter: P(glitter=1 | x, θ) = p_tp if on gold, p_fp otherwise

    where p_tp = 1 - obs_noise (true positive rate) and
          p_fp = obs_noise * 0.1 (false positive rate).

    Args:
        grid_size: Grid size
        pits: (n_configs, n_states) pit configs
        wumpus: (n_configs, n_states) wumpus configs
        gold: (n_configs, n_states) gold configs
        obs_noise: Noise level in [0, 1]. 0 = nearly deterministic.

    Returns:
        B: (3, 2, n_states, n_static) observation tensor.
           B[channel, obs_value, x, θ] = P(obs=obs_value | x, θ)
    """
    n_states = grid_size * grid_size
    n_static = pits.shape[0]

    p_tp = np.clip(1.0 - obs_noise, 0.01, 0.99)  # true positive
    p_fp = np.clip(obs_noise * 0.1, 0.01, 0.99)   # false positive

    B = np.zeros((N_OBS_CHANNELS, N_OBS_TYPES, n_states, n_static), dtype=np.float32)

    for theta in range(n_static):
        for x in range(n_states):
            neighbors = get_neighbors(x, grid_size)

            # Breeze: adjacent to any pit
            has_breeze = any(pits[theta, n] == 1.0 for n in neighbors)
            p_b = p_tp if has_breeze else p_fp
            B[BREEZE, 1, x, theta] = p_b
            B[BREEZE, 0, x, theta] = 1.0 - p_b

            # Stench: adjacent to wumpus
            has_stench = any(wumpus[theta, n] == 1.0 for n in neighbors)
            p_s = p_tp if has_stench else p_fp
            B[STENCH, 1, x, theta] = p_s
            B[STENCH, 0, x, theta] = 1.0 - p_s

            # Glitter: on gold
            has_glitter = gold[theta, x] == 1.0
            p_g = p_tp if has_glitter else p_fp
            B[GLITTER, 1, x, theta] = p_g
            B[GLITTER, 0, x, theta] = 1.0 - p_g

    return B


def generate_goal(
    gold: np.ndarray,
) -> np.ndarray:
    """Generate goal distribution: reach the gold cell.

    Since gold position varies by config θ, the goal is the marginal
    probability over gold positions (uniform over configs).

    Args:
        gold: (n_configs, n_states) gold configurations

    Returns:
        goal: (n_states,) goal distribution
    """
    goal = gold.mean(axis=0)
    total = goal.sum()
    if total > 0:
        goal = goal / total
    return goal.astype(np.float32)


# ---------------------------------------------------------------------------
# Simple simulator
# ---------------------------------------------------------------------------


@dataclass
class WumpusStepResult:
    obs: np.ndarray  # (3,) binary observations [breeze, stench, glitter]
    reward: float
    terminated: bool
    truncated: bool


class WumpusWorldEnv:
    """Simple Wumpus World simulator.

    Args:
        grid_size: Grid size
        pits: (n_configs, n_states) pit configurations
        wumpus: (n_configs, n_states) wumpus configurations
        gold: (n_configs, n_states) gold configurations
        slip_prob: Movement noise
        max_steps: Maximum steps per episode
    """

    def __init__(
        self,
        grid_size: int = 4,
        pits: np.ndarray | None = None,
        wumpus: np.ndarray | None = None,
        gold: np.ndarray | None = None,
        obs_tensor: np.ndarray | None = None,
        slip_prob: float = 0.0,
        max_steps: int = 100,
    ):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.pits = pits
        self.wumpus = wumpus
        self.gold = gold
        self.obs_tensor = obs_tensor
        self.slip_prob = slip_prob
        self._max_steps = max_steps
        self.start_pos = 0

        self._rng = np.random.default_rng(0)
        self._position = self.start_pos
        self._config_idx = 0
        self._steps = 0

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def config_idx(self) -> int:
        return self._config_idx

    def reset(self, seed: int | None = None, config_idx: int | None = None) -> WumpusStepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if config_idx is not None:
            self._config_idx = config_idx
        else:
            self._config_idx = int(self._rng.integers(0, self.pits.shape[0]))

        self._position = self.start_pos
        self._steps = 0

        return WumpusStepResult(
            obs=self._get_obs(),
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    def step(self, action: int) -> WumpusStepResult:
        self._steps += 1

        # Slip
        if self.slip_prob > 0 and self._rng.random() < self.slip_prob:
            other_actions = [a for a in range(N_ACTIONS) if a != action]
            action = int(self._rng.choice(other_actions))

        # Move
        row, col = pos_to_rc(self._position, self.grid_size)
        dr, dc = MOVEMENT[action]
        new_row, new_col = row + dr, col + dc

        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
            self._position = rc_to_pos(new_row, new_col, self.grid_size)

        # Termination
        theta = self._config_idx
        on_pit = self.pits[theta, self._position] == 1.0
        on_wumpus = self.wumpus[theta, self._position] == 1.0
        on_gold = self.gold[theta, self._position] == 1.0

        dead = on_pit or on_wumpus
        terminated = dead or on_gold
        truncated = self._steps >= self._max_steps and not terminated
        reward = 1.0 if on_gold else (-1.0 if dead else 0.0)

        return WumpusStepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

    def _get_obs(self) -> np.ndarray:
        """Sample binary observations [breeze, stench, glitter] from obs model."""
        if self.obs_tensor is not None:
            obs = np.zeros(N_OBS_CHANNELS, dtype=np.float32)
            for c in range(N_OBS_CHANNELS):
                p_fire = self.obs_tensor[c, 1, self._position, self._config_idx]
                obs[c] = float(self._rng.random() < p_fire)
            return obs
        else:
            # Fallback: deterministic
            theta = self._config_idx
            neighbors = get_neighbors(self._position, self.grid_size)
            breeze = float(any(self.pits[theta, n] == 1.0 for n in neighbors))
            stench = float(any(self.wumpus[theta, n] == 1.0 for n in neighbors))
            glitter = float(self.gold[theta, self._position] == 1.0)
            return np.array([breeze, stench, glitter], dtype=np.float32)

    def render_ascii(self) -> str:
        """Render current state as ASCII grid."""
        lines = []
        theta = self._config_idx
        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                pos = rc_to_pos(r, c, self.grid_size)
                if pos == self._position:
                    row_chars.append("A")
                elif self.pits[theta, pos] == 1.0:
                    row_chars.append("P")
                elif self.wumpus[theta, pos] == 1.0:
                    row_chars.append("W")
                elif self.gold[theta, pos] == 1.0:
                    row_chars.append("G")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)
