"""Frozen Lake environment with hidden holes.

Grid world where the agent must reach a goal position while avoiding hidden holes.
The agent gets 4 directional line-of-sight sensors reporting whether there is a
hole anywhere along the row/column in each direction.
Static state θ represents sampled hole configurations.

State space:
    x: agent position (row * grid_size + col), flat index 0..n_states-1
    θ: hole configuration index, one of k pre-sampled configs

Actions:
    0: LEFT, 1: DOWN, 2: RIGHT, 3: UP

Observations:
    Two modalities:
    1. Position (n_states channels): deterministic, θ-independent.
    2. Directional (4 channels: LEFT, DOWN, RIGHT, UP): stochastic,
       θ-dependent line-of-sight sensors.
    Observation tensor shape: (n_states + 4, 2, n_states, n_static)
"""

import numpy as np
from dataclasses import dataclass


# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
N_ACTIONS = 4

# Movement deltas: (delta_row, delta_col)
MOVEMENT = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}


N_OBS_CHANNELS = 4
N_OBS_TYPES = 2


def get_cells_in_direction(pos: int, direction: int, grid_size: int) -> list[int]:
    """Return all cells along row/column in the given direction from pos (excluding pos).

    LEFT/RIGHT scan the same row; DOWN/UP scan the same column.
    """
    row, col = divmod(pos, grid_size)
    cells = []
    if direction == LEFT:
        for c in range(col - 1, -1, -1):
            cells.append(row * grid_size + c)
    elif direction == RIGHT:
        for c in range(col + 1, grid_size):
            cells.append(row * grid_size + c)
    elif direction == DOWN:
        for r in range(row + 1, grid_size):
            cells.append(r * grid_size + col)
    elif direction == UP:
        for r in range(row - 1, -1, -1):
            cells.append(r * grid_size + col)
    return cells


def pos_to_rc(pos: int, grid_size: int) -> tuple[int, int]:
    """Convert flat position to (row, col)."""
    return divmod(pos, grid_size)


def rc_to_pos(row: int, col: int, grid_size: int) -> int:
    """Convert (row, col) to flat position."""
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


def has_path(start: int, goal: int, hole_set: set, grid_size: int) -> bool:
    """BFS check: is there a path from start to goal avoiding holes?"""
    if start == goal:
        return True
    visited = {start}
    queue = [start]
    while queue:
        pos = queue.pop(0)
        for nb in get_neighbors(pos, grid_size):
            if nb == goal:
                return True
            if nb not in visited and nb not in hole_set:
                visited.add(nb)
                queue.append(nb)
    return False


def manhattan_distance(a: int, b: int, grid_size: int) -> int:
    """Manhattan distance between two positions."""
    ra, ca = pos_to_rc(a, grid_size)
    rb, cb = pos_to_rc(b, grid_size)
    return abs(ra - rb) + abs(ca - cb)


def sample_configs(
    grid_size: int,
    n_configs: int,
    hole_fraction: float = 0.25,
    seed: int = 42,
    start_pos: int = 0,
    goal_pos: int | None = None,
    min_hamming: int = 0,
) -> np.ndarray:
    """Sample hole configurations with path guarantee and optional diversity.

    Every config is guaranteed to have a safe path from start to goal.
    When min_hamming > 0, uses greedy max-min selection from a large pool
    to maximize diversity between configs.

    Args:
        grid_size: Grid size (grid_size x grid_size)
        n_configs: Number of configurations to sample (= n_static)
        hole_fraction: Fraction of non-start/goal cells that are holes
        seed: Random seed
        start_pos: Start position (always safe)
        goal_pos: Goal position (always safe). Defaults to last cell.
        min_hamming: Minimum pairwise Hamming distance between configs.
                     0 = no constraint (pure random).

    Returns:
        holes: (n_configs, n_states) binary array.
               holes[θ, pos] = 1 if pos is a hole in config θ.
    """
    n_states = grid_size * grid_size
    if goal_pos is None:
        goal_pos = n_states - 1

    rng = np.random.default_rng(seed)
    candidates = [i for i in range(n_states) if i != start_pos and i != goal_pos]
    n_holes = max(1, int(len(candidates) * hole_fraction))

    def _sample_valid():
        """Generate random configs that have a valid path."""
        pool_set = set()
        pool = []
        attempts = 0
        target = max(n_configs * 50, 500) if min_hamming > 0 else n_configs
        while len(pool) < target and attempts < target * 20:
            attempts += 1
            hp = tuple(sorted(rng.choice(candidates, size=n_holes, replace=False)))
            if hp in pool_set:
                continue
            if not has_path(start_pos, goal_pos, set(hp), grid_size):
                continue
            pool_set.add(hp)
            vec = np.zeros(n_states, dtype=np.float32)
            for p in hp:
                vec[p] = 1.0
            pool.append(vec)
        return np.array(pool) if pool else np.zeros((0, n_states), dtype=np.float32)

    pool = _sample_valid()

    if pool.shape[0] < n_configs:
        raise ValueError(
            f"Could only generate {pool.shape[0]} valid configs (need {n_configs}). "
            f"Reduce n_configs, hole_fraction, or increase grid_size."
        )

    if min_hamming <= 0:
        return pool[:n_configs]

    # Greedy max-min diverse selection
    selected = [int(rng.integers(0, len(pool)))]
    remaining = set(range(len(pool))) - set(selected)

    for _ in range(n_configs - 1):
        best_idx = -1
        best_min_dist = -1
        for r in remaining:
            min_dist = min(
                int(np.sum(pool[r] != pool[s])) for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = r
        selected.append(best_idx)
        remaining.discard(best_idx)

    return pool[selected]


def generate_transition_tensor(
    grid_size: int,
    holes: np.ndarray,
    slip_prob: float = 0.0,
    goal_pos: int | None = None,
) -> np.ndarray:
    """Generate transition tensor T(x_new, x_old, θ, action).

    Args:
        grid_size: Grid size
        holes: (n_configs, n_states) hole configurations
        slip_prob: Probability of slipping to a random other direction
        goal_pos: Goal position (absorbing). Defaults to last cell.

    Returns:
        T: (n_states, n_states, n_static, n_actions) transition tensor
    """
    n_states = grid_size * grid_size
    n_static = holes.shape[0]

    if goal_pos is None:
        goal_pos = n_states - 1

    T = np.zeros((n_states, n_states, n_static, N_ACTIONS), dtype=np.float32)

    for theta in range(n_static):
        for x_old in range(n_states):
            # Absorbing states: holes and goal
            if holes[theta, x_old] == 1.0 or x_old == goal_pos:
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
                        x_new = x_old  # wall collision

                    T[x_new, x_old, theta, intended_action] += prob

    return T


def generate_observation_tensor(
    grid_size: int,
    holes: np.ndarray,
    base_noise: float = 0.05,
    noise_range: float = 0.15,
    false_alarm_scale: float = 0.3,
) -> np.ndarray:
    """Generate observation tensor B(channel, obs_type, x, θ).

    Two modalities:
    1. Position channels (0..n_states-1): deterministic, θ-independent.
       Channel c fires iff agent is at position c.
    2. Directional channels (n_states..n_states+3): stochastic, θ-dependent.
       LEFT, DOWN, RIGHT, UP line-of-sight sensors reporting whether there
       is a hole anywhere along the agent's row/column in that direction.

    Directional noise model:
    - noise(x) = base_noise + noise_range * manhattan_dist_from_center(x) / max_dist
    - If hole in direction: P(fire) = 1 - noise(x)
    - If no hole in direction: P(fire) = noise(x) * false_alarm_scale

    Args:
        grid_size: Grid size
        holes: (n_configs, n_states) hole configurations
        base_noise: Minimum noise level (at grid center)
        noise_range: Additional noise at maximum distance from center
        false_alarm_scale: Multiplier for false alarm probability

    Returns:
        B: (n_states + 4, 2, n_states, n_static) observation tensor.
    """
    n_states = grid_size * grid_size
    n_configs = holes.shape[0]

    n_channels = n_states + N_OBS_CHANNELS
    B = np.zeros((n_channels, N_OBS_TYPES, n_states, n_configs), dtype=np.float32)

    # --- Position channels (deterministic, θ-independent) ---
    for x in range(n_states):
        for c in range(n_states):
            if c == x:
                B[c, 1, x, :] = 0.999
                B[c, 0, x, :] = 0.001
            else:
                B[c, 1, x, :] = 0.001
                B[c, 0, x, :] = 0.999

    # --- Directional channels (stochastic, θ-dependent) ---
    # Precompute position-dependent noise
    center = (grid_size - 1) / 2.0
    max_dist = abs(center) * 2  # maximum manhattan distance from center
    noise = np.zeros(n_states, dtype=np.float64)
    for x in range(n_states):
        row, col = divmod(x, grid_size)
        dist_from_center = abs(row - center) + abs(col - center)
        noise[x] = base_noise + noise_range * dist_from_center / max_dist

    for theta in range(n_configs):
        for x in range(n_states):
            for d in range(N_OBS_CHANNELS):
                cells = get_cells_in_direction(x, d, grid_size)
                has_hole = any(holes[theta, c] == 1.0 for c in cells)

                if has_hole:
                    p = 1.0 - noise[x]
                else:
                    p = noise[x] * false_alarm_scale

                p = np.clip(p, 0.01, 0.99)
                B[n_states + d, 1, x, theta] = p
                B[n_states + d, 0, x, theta] = 1.0 - p

    return B


def generate_goal(
    grid_size: int,
    holes: np.ndarray,
    goal_pos: int | None = None,
    goal_reward: float = 1.0,
    hole_penalty: float = 1.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Generate goal distribution via softmax over a reward vector.

    Positive reward at the goal position, negative penalty at expected hole
    positions (marginal over configs). Other positions get zero reward.

    Args:
        grid_size: Grid size
        holes: (n_configs, n_states) hole configurations
        goal_pos: Goal position. Defaults to last cell.
        goal_reward: Reward magnitude at goal position
        hole_penalty: Penalty magnitude scaled by hole probability
        temperature: Softmax temperature (lower = more peaked)

    Returns:
        goal: (n_states,) softmax goal distribution
    """
    n_states = grid_size * grid_size
    if goal_pos is None:
        goal_pos = n_states - 1

    rewards = np.zeros(n_states, dtype=np.float64)
    rewards[goal_pos] = goal_reward

    hole_marginal = holes.mean(axis=0)  # P(hole at pos) across configs
    rewards -= hole_penalty * hole_marginal

    scaled = rewards / temperature
    scaled -= scaled.max()  # numerical stability
    goal = np.exp(scaled)
    goal /= goal.sum()
    return goal.astype(np.float32)


# ---------------------------------------------------------------------------
# Simple simulator
# ---------------------------------------------------------------------------


@dataclass
class FrozenLakeStepResult:
    obs: np.ndarray  # (n_states + 4,) position one-hot + directional sensors
    reward: float
    terminated: bool
    truncated: bool


class FrozenLakeEnv:
    """Simple Frozen Lake simulator.

    Args:
        grid_size: Grid size
        holes: (n_configs, n_states) pre-sampled hole configurations
        obs_tensor: (n_states, 2, n_states, n_configs) observation tensor for sampling
        slip_prob: Movement noise
        max_steps: Maximum steps per episode
        goal_pos: Goal position (defaults to last cell)
    """

    def __init__(
        self,
        grid_size: int = 4,
        holes: np.ndarray | None = None,
        obs_tensor: np.ndarray | None = None,
        slip_prob: float = 0.0,
        max_steps: int = 100,
        goal_pos: int | None = None,
    ):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_obs_channels = obs_tensor.shape[0] if obs_tensor is not None else N_OBS_CHANNELS
        self.holes = holes
        self.obs_tensor = obs_tensor
        self.slip_prob = slip_prob
        self._max_steps = max_steps
        self.goal_pos = goal_pos if goal_pos is not None else self.n_states - 1
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
        """Index of the current hole configuration."""
        return self._config_idx

    def reset(self, seed: int | None = None, config_idx: int | None = None) -> FrozenLakeStepResult:
        """Reset environment.

        Args:
            seed: Random seed
            config_idx: Specific config to use. If None, sampled randomly.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if config_idx is not None:
            self._config_idx = config_idx
        else:
            self._config_idx = int(self._rng.integers(0, self.holes.shape[0]))

        self._position = self.start_pos
        self._steps = 0

        return FrozenLakeStepResult(
            obs=self._get_obs(),
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    def step(self, action: int) -> FrozenLakeStepResult:
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
        on_hole = self.holes[self._config_idx, self._position] == 1.0
        on_goal = self._position == self.goal_pos
        terminated = on_hole or on_goal
        truncated = self._steps >= self._max_steps and not terminated
        reward = 1.0 if on_goal else 0.0

        return FrozenLakeStepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

    def _get_obs(self) -> np.ndarray:
        """Sample binary sensor readings from the observation model."""
        if self.obs_tensor is not None:
            n_channels = self.obs_tensor.shape[0]
            obs = np.zeros(n_channels, dtype=np.float32)
            for c in range(n_channels):
                p_fire = self.obs_tensor[c, 1, self._position, self._config_idx]
                obs[c] = float(self._rng.random() < p_fire)
        else:
            # Fallback: deterministic position + directional observation
            obs = np.zeros(self.n_states + N_OBS_CHANNELS, dtype=np.float32)
            obs[self._position] = 1.0  # position one-hot
            config = self.holes[self._config_idx] if self.holes is not None else None
            if config is not None:
                for d in range(N_OBS_CHANNELS):
                    cells = get_cells_in_direction(self._position, d, self.grid_size)
                    obs[self.n_states + d] = float(any(config[c] == 1.0 for c in cells))
        return obs

    def render_ascii(self) -> str:
        """Render current state as ASCII grid."""
        lines = []
        config = self.holes[self._config_idx]
        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                pos = rc_to_pos(r, c, self.grid_size)
                if pos == self._position:
                    row_chars.append("A")
                elif pos == self.goal_pos:
                    row_chars.append("G")
                elif config[pos] == 1.0:
                    row_chars.append("H")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)
