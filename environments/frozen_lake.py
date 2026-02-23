"""Frozen Lake environment with hidden holes and SCAN action.

Grid world where the agent must reach a goal position while avoiding hidden holes.
The agent observes the entire grid at every timestep: one binary channel per cell
reporting "hole or safe". When unscanned, observation noise increases with Manhattan
distance from the agent to the observed cell. After SCAN, all cells are observed
near-deterministically.
Static state θ represents sampled hole configurations.

State space (doubled for scan mode):
    x: state_index(pos, scanned, n_pos) where pos = row * grid_size + col
    mode 0 = unscanned (noisy observations), mode 1 = scanned (near-deterministic)
    Total states: 2 * n_pos

Actions:
    0: LEFT, 1: DOWN, 2: RIGHT, 3: UP, 4: SCAN

Observations:
    Two modalities:
    1. Position (2*n_pos channels): deterministic, θ-independent.
    2. Grid cell (n_pos channels): one per cell, binary "hole/safe".
       Unscanned: noise grows with distance from agent to cell.
       Scanned: near-deterministic.
    Observation tensor shape: (2*n_pos + n_pos, 2, 2*n_pos, n_static)
"""

import numpy as np
from dataclasses import dataclass


# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
SCAN = 4
N_ACTIONS = 5
N_MOVEMENT_ACTIONS = 4

# Movement deltas: (delta_row, delta_col)
MOVEMENT = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}


N_OBS_TYPES = 2


def state_index(pos: int, scanned: int, n_pos: int) -> int:
    """Compute flat state index: x = pos + scanned * n_pos."""
    return pos + scanned * n_pos


def unpack_state(x: int, n_pos: int) -> tuple[int, int]:
    """Unpack flat state index into (pos, scanned)."""
    scanned = x // n_pos
    pos = x % n_pos
    return pos, scanned


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

    State space is doubled: x = pos + scanned * n_pos.
    Movement actions (0-3) preserve scan mode. SCAN action (4) transitions
    from unscanned to scanned (deterministic, no slip).

    Args:
        grid_size: Grid size
        holes: (n_configs, n_pos) hole configurations
        slip_prob: Probability of slipping to a random other movement direction
        goal_pos: Goal position (absorbing). Defaults to last cell.

    Returns:
        T: (2*n_pos, 2*n_pos, n_static, 5) transition tensor
    """
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos  # doubled for scan mode
    n_static = holes.shape[0]

    if goal_pos is None:
        goal_pos = n_pos - 1

    T = np.zeros((n_states, n_states, n_static, N_ACTIONS), dtype=np.float32)

    for theta in range(n_static):
        for x_old in range(n_states):
            pos_old, scanned_old = unpack_state(x_old, n_pos)

            # Absorbing states: holes and goal (in both modes)
            if holes[theta, pos_old] == 1.0 or pos_old == goal_pos:
                T[x_old, x_old, theta, :] = 1.0
                continue

            # Movement actions (0-3): preserve scan mode, apply slip
            for intended_action in range(N_MOVEMENT_ACTIONS):
                for actual_action in range(N_MOVEMENT_ACTIONS):
                    if actual_action == intended_action:
                        prob = 1.0 - slip_prob
                    else:
                        prob = slip_prob / (N_MOVEMENT_ACTIONS - 1)

                    if prob == 0.0:
                        continue

                    row, col = pos_to_rc(pos_old, grid_size)
                    dr, dc = MOVEMENT[actual_action]
                    new_row, new_col = row + dr, col + dc

                    if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                        pos_new = rc_to_pos(new_row, new_col, grid_size)
                    else:
                        pos_new = pos_old  # wall collision

                    x_new = state_index(pos_new, scanned_old, n_pos)
                    T[x_new, x_old, theta, intended_action] += prob

            # SCAN action (4): deterministic, no slip
            # unscanned -> scanned (same position), scanned -> stays scanned
            x_new_scan = state_index(pos_old, 1, n_pos)
            T[x_new_scan, x_old, theta, SCAN] = 1.0

    return T


def generate_observation_tensor(
    grid_size: int,
    holes: np.ndarray,
    base_noise: float = 0.05,
    noise_range: float = 0.15,
) -> np.ndarray:
    """Generate observation tensor B(channel, obs_type, x, θ).

    State space is doubled (2*n_pos) for scan mode.

    Two modalities:
    1. Position channels (0..2*n_pos-1): deterministic, θ-independent.
       Channel c fires iff agent is at state c.
    2. Grid cell channels (2*n_pos..3*n_pos-1): one per cell, binary
       "hole/safe" sensor. Fires if there is a hole at cell c.
       Unscanned mode: noise grows with Manhattan distance from agent to cell.
       Scanned mode: near-deterministic (p_tp=0.999, p_fp=0.001).

    Args:
        grid_size: Grid size
        holes: (n_configs, n_pos) hole configurations
        base_noise: Minimum noise level (agent at same cell) for unscanned mode
        noise_range: Additional noise at maximum Manhattan distance

    Returns:
        B: (2*n_pos + n_pos, 2, 2*n_pos, n_static) observation tensor.
    """
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos
    n_configs = holes.shape[0]
    max_dist = 2 * (grid_size - 1)  # max Manhattan distance on grid

    n_channels = n_states + n_pos  # position channels + grid cell channels
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

    # --- Grid cell channels (stochastic, θ-dependent) ---
    # Channel (n_states + cell) fires if there is a hole at `cell`.
    # Noise depends on Manhattan distance from agent position to observed cell.
    for theta in range(n_configs):
        for pos in range(n_pos):
            row_a, col_a = divmod(pos, grid_size)

            for cell in range(n_pos):
                row_c, col_c = divmod(cell, grid_size)
                dist = abs(row_a - row_c) + abs(col_a - col_c)
                noise = base_noise + noise_range * dist / max_dist if max_dist > 0 else base_noise

                has_hole = holes[theta, cell] == 1.0
                ch = n_states + cell

                # Unscanned mode: noisy
                x_unscanned = state_index(pos, 0, n_pos)
                if has_hole:
                    p = np.clip(1.0 - noise, 0.01, 0.99)
                else:
                    p = np.clip(noise, 0.01, 0.99)
                B[ch, 1, x_unscanned, theta] = p
                B[ch, 0, x_unscanned, theta] = 1.0 - p

                # Scanned mode: near-deterministic
                x_scanned = state_index(pos, 1, n_pos)
                if has_hole:
                    B[ch, 1, x_scanned, theta] = 0.999
                    B[ch, 0, x_scanned, theta] = 0.001
                else:
                    B[ch, 1, x_scanned, theta] = 0.001
                    B[ch, 0, x_scanned, theta] = 0.999

    return B


def generate_goal(
    grid_size: int,
    holes: np.ndarray,
    goal_pos: int | None = None,
    goal_reward: float = 1.0,
    hole_penalty: float = 1.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Generate per-config preference factor C(x, θ) via softmax over rewards.

    State space is doubled (2*n_pos). Goal and hole penalties apply in both
    scan modes. Each config θ gets its own reward vector based on which
    positions are holes in that config.

    Args:
        grid_size: Grid size
        holes: (n_configs, n_pos) hole configurations
        goal_pos: Goal position. Defaults to last cell.
        goal_reward: Reward magnitude at goal position
        hole_penalty: Penalty magnitude for hole positions
        temperature: Softmax temperature (lower = more peaked)

    Returns:
        goal: (2*n_pos, n_configs) per-config softmax preference
    """
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos
    n_static = holes.shape[0]
    if goal_pos is None:
        goal_pos = n_pos - 1

    rewards = np.zeros((n_states, n_static), dtype=np.float64)

    for theta in range(n_static):
        for mode in range(2):
            # Goal reward in both modes
            rewards[state_index(goal_pos, mode, n_pos), theta] = goal_reward
            # Hole penalty: full penalty for actual holes in this config
            for pos in range(n_pos):
                if holes[theta, pos] == 1.0:
                    rewards[state_index(pos, mode, n_pos), theta] -= hole_penalty

    scaled = rewards / temperature
    scaled -= scaled.max(axis=0, keepdims=True)  # numerical stability per config
    goal = np.exp(scaled)
    goal /= goal.sum(axis=0, keepdims=True)
    return goal.astype(np.float32)


# ---------------------------------------------------------------------------
# Simple simulator
# ---------------------------------------------------------------------------


@dataclass
class FrozenLakeStepResult:
    obs: np.ndarray  # (3*n_pos,) position one-hot + grid cell sensors
    reward: float
    terminated: bool
    truncated: bool


class FrozenLakeEnv:
    """Simple Frozen Lake simulator with SCAN action.

    State includes scan mode: agent starts unscanned, SCAN action switches
    to scanned mode with near-deterministic observations.

    Args:
        grid_size: Grid size
        holes: (n_configs, n_pos) pre-sampled hole configurations
        obs_tensor: (3*n_pos, 2, 2*n_pos, n_configs) observation tensor
        slip_prob: Movement noise (only affects movement actions, not SCAN)
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
        self.n_pos = grid_size * grid_size
        self.n_obs_channels = obs_tensor.shape[0] if obs_tensor is not None else (3 * self.n_pos)
        self.holes = holes
        self.obs_tensor = obs_tensor
        self.slip_prob = slip_prob
        self._max_steps = max_steps
        self.goal_pos = goal_pos if goal_pos is not None else self.n_pos - 1
        self.start_pos = 0

        self._rng = np.random.default_rng(0)
        self._position = self.start_pos
        self._scanned = 0  # 0 = unscanned, 1 = scanned
        self._config_idx = 0
        self._steps = 0

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def config_idx(self) -> int:
        """Index of the current hole configuration."""
        return self._config_idx

    @property
    def _state_idx(self) -> int:
        """Current flat state index (pos + scanned * n_pos)."""
        return state_index(self._position, self._scanned, self.n_pos)

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
        self._scanned = 0
        self._steps = 0

        return FrozenLakeStepResult(
            obs=self._get_obs(),
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    def step(self, action: int) -> FrozenLakeStepResult:
        self._steps += 1

        if action == SCAN:
            # SCAN: switch to scanned mode, no movement, no slip
            self._scanned = 1
        else:
            # Movement action: apply slip (only among movement actions 0-3)
            if self.slip_prob > 0 and self._rng.random() < self.slip_prob:
                other_actions = [a for a in range(N_MOVEMENT_ACTIONS) if a != action]
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
        x = self._state_idx
        if self.obs_tensor is not None:
            n_channels = self.obs_tensor.shape[0]
            obs = np.zeros(n_channels, dtype=np.float32)
            for c in range(n_channels):
                p_fire = self.obs_tensor[c, 1, x, self._config_idx]
                obs[c] = float(self._rng.random() < p_fire)
        else:
            # Fallback: deterministic position + grid cell observation
            n_states = 2 * self.n_pos
            obs = np.zeros(n_states + self.n_pos, dtype=np.float32)
            obs[x] = 1.0  # position one-hot
            config = self.holes[self._config_idx] if self.holes is not None else None
            if config is not None:
                for cell in range(self.n_pos):
                    obs[n_states + cell] = float(config[cell] == 1.0)
        return obs

    def render_ascii(self) -> str:
        """Render current state as ASCII grid."""
        lines = []
        config = self.holes[self._config_idx]
        mode_str = "SCANNED" if self._scanned else "UNSCANNED"
        lines.append(f"[{mode_str}]")
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
