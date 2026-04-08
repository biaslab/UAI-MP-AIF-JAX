"""RockSample[n,k] environment with distance-dependent observations.

Grid world where the agent must collect good rocks and exit the east edge.
Rocks have unknown quality (good/bad) inferred from distance-dependent noisy
observations. The agent can SCAN the nearest unscanned rock for clarity,
SAMPLE rocks to collect them, and move in four directions.

Static state θ represents sampled rock quality configurations.

State space:
    x: state_index(pos, collected_mask, scanned_mask, n_pos, n_collect, n_scan)
    pos = row * grid_size + col (row-major)
    collected_mask = 0..2^k-1 (bitmask of collected rocks)
    scanned_mask = 0..2^k-1 (bitmask of which rocks have been scanned)
    Total states: n_pos × 2^k × 2^k

Actions:
    0: LEFT, 1: DOWN, 2: RIGHT, 3: UP, 4: SCAN, 5: SAMPLE

Observations:
    Two modalities:
    1. Position channels (0..n_pos-1): θ-independent, fixed noise (pos_noise).
    2. Rock quality channels (n_pos..n_pos+k-1): θ-dependent.
       Unscanned: accuracy = 0.5 + 0.5 * 2^(-d / half_eff_dist).
       Scanned: near-deterministic.
    Observation tensor shape: (n_pos + k, 2, n_states, n_configs)
"""

import numpy as np
from dataclasses import dataclass

# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
SCAN = 4
SAMPLE = 5
N_ACTIONS = 6
N_MOVEMENT_ACTIONS = 4

# Movement deltas: (delta_row, delta_col)
MOVEMENT = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}

N_OBS_TYPES = 2


def state_index(pos: int, collected_mask: int, scanned_mask: int,
                n_pos: int, n_collect: int, n_scan: int) -> int:
    """Compute flat state index.

    x = pos + collected_mask * n_pos + scanned_mask * n_pos * n_collect
    """
    return pos + collected_mask * n_pos + scanned_mask * n_pos * n_collect


def unpack_state(x: int, n_pos: int, n_collect: int, n_scan: int) -> tuple[int, int, int]:
    """Unpack flat state index into (pos, collected_mask, scanned_mask)."""
    scanned_mask = x // (n_pos * n_collect)
    remainder = x % (n_pos * n_collect)
    collected_mask = remainder // n_pos
    pos = remainder % n_pos
    return pos, collected_mask, scanned_mask


def pos_to_rc(pos: int, grid_size: int) -> tuple[int, int]:
    """Convert flat position to (row, col)."""
    return divmod(pos, grid_size)


def rc_to_pos(row: int, col: int, grid_size: int) -> int:
    """Convert (row, col) to flat position."""
    return row * grid_size + col


def euclidean_distance(pos_a: int, pos_b: int, grid_size: int) -> float:
    """Euclidean distance between two grid positions."""
    ra, ca = pos_to_rc(pos_a, grid_size)
    rb, cb = pos_to_rc(pos_b, grid_size)
    return ((ra - rb) ** 2 + (ca - cb) ** 2) ** 0.5


def is_exit(pos: int, grid_size: int) -> bool:
    """Check if position is in the exit column (rightmost)."""
    _, col = pos_to_rc(pos, grid_size)
    return col == grid_size - 1


def nearest_unscanned_rock(
    pos: int,
    scanned_mask: int,
    rock_positions: np.ndarray,
    grid_size: int,
) -> int:
    """Find the nearest unscanned rock by Euclidean distance.

    Args:
        pos: Current grid position
        scanned_mask: Bitmask of already-scanned rocks
        rock_positions: (k,) array of rock positions
        grid_size: Grid size

    Returns:
        Index j of nearest unscanned rock, or -1 if all scanned.
    """
    best_j = -1
    best_dist = float('inf')
    for j, rp in enumerate(rock_positions):
        if scanned_mask & (1 << j):
            continue  # already scanned
        d = euclidean_distance(pos, int(rp), grid_size)
        if d < best_dist:
            best_dist = d
            best_j = j
    return best_j


def sample_rock_positions(
    grid_size: int,
    n_rocks: int,
    seed: int = 42,
) -> np.ndarray:
    """Sample rock positions on the grid.

    Rocks are placed at random non-start, non-exit positions.

    Args:
        grid_size: Grid size (n x n)
        n_rocks: Number of rocks (k)
        seed: Random seed

    Returns:
        rock_positions: (k,) array of grid positions for each rock
    """
    rng = np.random.default_rng(seed)

    # Start position: middle-left
    start_row = grid_size // 2
    start_pos = rc_to_pos(start_row, 0, grid_size)

    # Candidate positions: not start, not exit column
    candidates = []
    for r in range(grid_size):
        for c in range(grid_size):
            pos = rc_to_pos(r, c, grid_size)
            if pos != start_pos and not is_exit(pos, grid_size):
                candidates.append(pos)

    if len(candidates) < n_rocks:
        raise ValueError(
            f"Not enough candidate positions ({len(candidates)}) for {n_rocks} rocks. "
            f"Increase grid_size or reduce n_rocks."
        )

    return np.array(
        sorted(rng.choice(candidates, size=n_rocks, replace=False)),
        dtype=np.int32,
    )


def all_quality_configs(n_rocks: int) -> np.ndarray:
    """Enumerate all 2^k rock quality configurations.

    Each config θ is a k-bit assignment: qualities[θ, j] = 1 means rock j
    is good. This exhaustive enumeration ensures rock qualities are
    independent across rocks — no spurious correlations.

    Args:
        n_rocks: Number of rocks (k). Must be small enough that 2^k is manageable.

    Returns:
        qualities: (2^k, k) binary array.
    """
    n_configs = 2 ** n_rocks
    qualities = np.zeros((n_configs, n_rocks), dtype=np.float32)
    for i in range(n_configs):
        for j in range(n_rocks):
            qualities[i, j] = float((i >> j) & 1)
    return qualities


def generate_transition_tensor(
    grid_size: int,
    rock_positions: np.ndarray,
    n_rocks: int,
    slip_prob: float = 0.0,
) -> np.ndarray:
    """Generate transition tensor T(x_new, x_old, θ, action).

    θ-independent (same dynamics regardless of rock quality) but tiled across
    all 2^k configs for framework compatibility.

    Args:
        grid_size: Grid size
        rock_positions: (k,) rock positions
        n_rocks: Number of rocks
        slip_prob: Movement slip probability
    Returns:
        T: (n_states, n_states, 2^k, 6) transition tensor
    """
    n_pos = grid_size * grid_size
    n_collect = 2 ** n_rocks
    n_scan = 2 ** n_rocks
    n_configs = n_collect  # exhaustive: one config per quality assignment
    n_states = n_pos * n_collect * n_scan

    # Build rock position lookup: pos -> rock index
    rock_at_pos = {}
    for j, rp in enumerate(rock_positions):
        rock_at_pos[int(rp)] = j

    # Exit positions
    exit_positions = set()
    for r in range(grid_size):
        exit_positions.add(rc_to_pos(r, grid_size - 1, grid_size))

    # Build for θ=0 then tile
    T_single = np.zeros((n_states, n_states, 1, N_ACTIONS), dtype=np.float32)

    for x_old in range(n_states):
        pos_old, coll_old, scanned_old = unpack_state(x_old, n_pos, n_collect, n_scan)

        # Exit positions are absorbing
        if pos_old in exit_positions:
            T_single[x_old, x_old, 0, :] = 1.0
            continue

        # Movement actions (0-3)
        for intended in range(N_MOVEMENT_ACTIONS):
            for actual in range(N_MOVEMENT_ACTIONS):
                if actual == intended:
                    prob = 1.0 - slip_prob
                else:
                    prob = slip_prob / (N_MOVEMENT_ACTIONS - 1)
                if prob == 0.0:
                    continue

                row, col = pos_to_rc(pos_old, grid_size)
                dr, dc = MOVEMENT[actual]
                nr, nc = row + dr, col + dc

                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    pos_new = rc_to_pos(nr, nc, grid_size)
                else:
                    pos_new = pos_old  # wall collision

                x_new = state_index(pos_new, coll_old, scanned_old, n_pos, n_collect, n_scan)
                T_single[x_new, x_old, 0, intended] += prob

        # SCAN (4): find nearest unscanned rock, set its bit
        j = nearest_unscanned_rock(pos_old, scanned_old, rock_positions, grid_size)
        if j >= 0:
            scanned_new = scanned_old | (1 << j)
            x_scan = state_index(pos_old, coll_old, scanned_new, n_pos, n_collect, n_scan)
        else:
            # All rocks already scanned: self-loop
            x_scan = x_old
        T_single[x_scan, x_old, 0, SCAN] = 1.0

        # SAMPLE (5): if at rock j and rock j not collected, set bit j
        if pos_old in rock_at_pos:
            j = rock_at_pos[pos_old]
            if not (coll_old & (1 << j)):
                # Collect rock j
                coll_new = coll_old | (1 << j)
                x_new = state_index(pos_old, coll_new, scanned_old, n_pos, n_collect, n_scan)
                T_single[x_new, x_old, 0, SAMPLE] = 1.0
            else:
                # Already collected: self-loop
                T_single[x_old, x_old, 0, SAMPLE] = 1.0
        else:
            # Not at a rock: self-loop
            T_single[x_old, x_old, 0, SAMPLE] = 1.0

    # Tile across θ
    T = np.tile(T_single, (1, 1, n_configs, 1))
    return T


def generate_observation_tensor(
    grid_size: int,
    rock_positions: np.ndarray,
    qualities: np.ndarray,
    n_rocks: int,
    half_eff_dist: float = 2.0,
    pos_noise: float = 0.1,
) -> np.ndarray:
    """Generate observation tensor B(channel, obs_type, x, θ).

    Two modalities:
    1. Position channels (0..n_pos-1): θ-independent.
       Always use fixed noise (pos_noise).
    2. Rock quality channels (n_pos..n_pos+k-1): θ-dependent.
       P(correct | d) = 0.5 + 0.5 * 2^(-d / half_eff_dist)
       Scanned (bit j set in scanned_mask): near-deterministic.

    Args:
        grid_size: Grid size
        rock_positions: (k,) rock positions
        qualities: (n_configs, k) binary rock quality
        n_rocks: Number of rocks
        half_eff_dist: Distance at which observation accuracy halves toward 0.5
        pos_noise: Position channel noise

    Returns:
        B: (n_pos + k, 2, n_states, n_configs)
    """
    n_pos = grid_size * grid_size
    n_collect = 2 ** n_rocks
    n_scan = 2 ** n_rocks
    n_states = n_pos * n_collect * n_scan
    n_configs = qualities.shape[0]
    n_channels = n_pos + n_rocks

    p_tp_pos = np.clip(1.0 - pos_noise, 0.01, 0.99)
    p_fp_pos = np.clip(pos_noise * 0.1, 0.01, 0.99)
    p_tp_s = 0.999
    p_fp_s = 0.001

    B = np.zeros((n_channels, N_OBS_TYPES, n_states, n_configs), dtype=np.float32)

    # --- Position channels (θ-independent, always fixed noise) ---
    for target_pos in range(n_pos):
        ch = target_pos
        for x in range(n_states):
            pos, _, _ = unpack_state(x, n_pos, n_collect, n_scan)
            p = p_tp_pos if pos == target_pos else p_fp_pos
            B[ch, 1, x, :] = p
            B[ch, 0, x, :] = 1.0 - p

    # --- Rock quality channels (θ-dependent) ---
    for j in range(n_rocks):
        ch = n_pos + j
        rock_pos = int(rock_positions[j])
        for theta in range(n_configs):
            rock_good = qualities[theta, j] == 1.0
            for x in range(n_states):
                pos, _, scanned_mask = unpack_state(x, n_pos, n_collect, n_scan)
                if scanned_mask & (1 << j):
                    # This rock has been scanned: near-deterministic
                    if rock_good:
                        B[ch, 1, x, theta] = p_tp_s
                        B[ch, 0, x, theta] = p_fp_s
                    else:
                        B[ch, 1, x, theta] = p_fp_s
                        B[ch, 0, x, theta] = p_tp_s
                else:
                    # Distance-dependent accuracy
                    d = euclidean_distance(pos, rock_pos, grid_size)
                    p_correct = 0.5 + 0.5 * (2.0 ** (-d / half_eff_dist))
                    p_correct = np.clip(p_correct, 0.01, 0.99)
                    if rock_good:
                        B[ch, 1, x, theta] = p_correct
                        B[ch, 0, x, theta] = 1.0 - p_correct
                    else:
                        B[ch, 1, x, theta] = 1.0 - p_correct
                        B[ch, 0, x, theta] = p_correct

    return B


def generate_goal(
    grid_size: int,
    rock_positions: np.ndarray,
    qualities: np.ndarray,
    n_rocks: int,
    exit_reward: float = 10.0,
    good_reward: float = 10.0,
    bad_penalty: float = 10.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Generate per-config preference factor C(x, θ) via softmax over rewards.

    Reward encodes: reaching exit with good rocks collected is desirable,
    having collected bad rocks is undesirable. scanned_mask is ignored
    (scanning is purely informational).

    Args:
        grid_size: Grid size
        rock_positions: (k,) rock positions
        qualities: (n_configs, k) binary rock quality
        n_rocks: Number of rocks
        exit_reward: Reward for being at exit
        good_reward: Reward per collected good rock
        bad_penalty: Penalty per collected bad rock
        temperature: Softmax temperature

    Returns:
        goal: (n_states, n_configs) per-config softmax preference
    """
    n_pos = grid_size * grid_size
    n_collect = 2 ** n_rocks
    n_scan = 2 ** n_rocks
    n_states = n_pos * n_collect * n_scan
    n_configs = qualities.shape[0]

    rewards = np.zeros((n_states, n_configs), dtype=np.float64)

    for theta in range(n_configs):
        for x in range(n_states):
            pos, coll, _ = unpack_state(x, n_pos, n_collect, n_scan)

            r = 0.0
            if is_exit(pos, grid_size):
                r += exit_reward

            for j in range(n_rocks):
                if coll & (1 << j):
                    if qualities[theta, j] == 1.0:
                        r += good_reward
                    else:
                        r -= bad_penalty

            rewards[x, theta] = r

    scaled = rewards / temperature
    scaled -= scaled.max(axis=0, keepdims=True)
    goal = np.exp(scaled)
    goal /= goal.sum(axis=0, keepdims=True)
    return goal.astype(np.float32)


# ---------------------------------------------------------------------------
# Simple simulator
# ---------------------------------------------------------------------------


@dataclass
class RockSampleStepResult:
    obs: np.ndarray  # (n_pos + k,) binary observations
    reward: float
    terminated: bool
    truncated: bool


class RockSampleEnv:
    """Simple RockSample simulator.

    Args:
        grid_size: Grid size
        rock_positions: (k,) rock positions
        qualities: (n_configs, k) rock quality configs
        n_rocks: Number of rocks
        obs_tensor: (n_pos + k, 2, n_states, n_configs) observation tensor
        slip_prob: Movement noise
        max_steps: Maximum steps per episode
        good_reward: Reward for sampling a good rock
        bad_penalty: Penalty for sampling a bad rock
        exit_reward: Reward for reaching exit
    """

    def __init__(
        self,
        grid_size: int = 5,
        rock_positions: np.ndarray | None = None,
        qualities: np.ndarray | None = None,
        n_rocks: int = 3,
        obs_tensor: np.ndarray | None = None,
        slip_prob: float = 0.0,
        max_steps: int = 50,
        good_reward: float = 10.0,
        bad_penalty: float = 10.0,
        exit_reward: float = 10.0,
    ):
        self.grid_size = grid_size
        self.n_pos = grid_size * grid_size
        self.n_rocks = n_rocks
        self.n_collect = 2 ** n_rocks
        self.n_scan = 2 ** n_rocks
        self.rock_positions = rock_positions
        self.qualities = qualities
        self.obs_tensor = obs_tensor
        self.slip_prob = slip_prob
        self._max_steps = max_steps
        self.good_reward = good_reward
        self.bad_penalty = bad_penalty
        self.exit_reward = exit_reward

        # Start position: middle-left
        start_row = grid_size // 2
        self.start_pos = rc_to_pos(start_row, 0, grid_size)

        # Rock position lookup
        self._rock_at_pos = {}
        if rock_positions is not None:
            for j, rp in enumerate(rock_positions):
                self._rock_at_pos[int(rp)] = j

        self._rng = np.random.default_rng(0)
        self._position = self.start_pos
        self._collected = 0  # bitmask
        self._scanned_mask = 0  # bitmask of scanned rocks
        self._config_idx = 0
        self._steps = 0

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def config_idx(self) -> int:
        return self._config_idx

    @property
    def _state_idx(self) -> int:
        return state_index(
            self._position, self._collected, self._scanned_mask,
            self.n_pos, self.n_collect, self.n_scan,
        )

    def reset(self, seed: int | None = None, config_idx: int | None = None) -> RockSampleStepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if config_idx is not None:
            self._config_idx = config_idx
        else:
            self._config_idx = int(self._rng.integers(0, self.qualities.shape[0]))

        self._position = self.start_pos
        self._collected = 0
        self._scanned_mask = 0
        self._steps = 0

        return RockSampleStepResult(
            obs=self._get_obs(),
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    def step(self, action: int) -> RockSampleStepResult:
        self._steps += 1
        reward = 0.0
        terminated = False

        if action == SCAN:
            j = nearest_unscanned_rock(
                self._position, self._scanned_mask,
                self.rock_positions, self.grid_size,
            )
            if j >= 0:
                self._scanned_mask |= (1 << j)
            # else: all scanned, no-op
        elif action == SAMPLE:
            if self._position in self._rock_at_pos:
                j = self._rock_at_pos[self._position]
                if not (self._collected & (1 << j)):
                    # Collect rock
                    self._collected |= (1 << j)
                    rock_good = self.qualities[self._config_idx, j] == 1.0
                    reward = self.good_reward if rock_good else -self.bad_penalty
            # else: no rock or already collected, self-loop with 0 reward
        else:
            # Movement
            if self.slip_prob > 0 and self._rng.random() < self.slip_prob:
                other = [a for a in range(N_MOVEMENT_ACTIONS) if a != action]
                action = int(self._rng.choice(other))

            row, col = pos_to_rc(self._position, self.grid_size)
            dr, dc = MOVEMENT[action]
            nr, nc = row + dr, col + dc

            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                self._position = rc_to_pos(nr, nc, self.grid_size)

        # Check exit
        if is_exit(self._position, self.grid_size):
            reward += self.exit_reward
            terminated = True

        truncated = self._steps >= self._max_steps and not terminated

        return RockSampleStepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

    def _get_obs(self) -> np.ndarray:
        """Sample binary observations from observation model."""
        x = self._state_idx
        if self.obs_tensor is not None:
            n_channels = self.obs_tensor.shape[0]
            obs = np.zeros(n_channels, dtype=np.float32)
            for c in range(n_channels):
                p_fire = self.obs_tensor[c, 1, x, self._config_idx]
                obs[c] = float(self._rng.random() < p_fire)
            return obs
        else:
            # Fallback: deterministic
            n_channels = self.n_pos + self.n_rocks
            obs = np.zeros(n_channels, dtype=np.float32)
            obs[self._position] = 1.0
            for j in range(self.n_rocks):
                obs[self.n_pos + j] = float(
                    self.qualities[self._config_idx, j] == 1.0
                )
            return obs

    def render_ascii(self) -> str:
        """Render current state as ASCII grid."""
        lines = []
        theta = self._config_idx
        scan_str = bin(self._scanned_mask)[2:].zfill(self.n_rocks)
        coll_str = bin(self._collected)[2:].zfill(self.n_rocks)
        lines.append(f"[scanned={scan_str}] collected={coll_str}")

        rock_set = {int(rp): j for j, rp in enumerate(self.rock_positions)}

        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                pos = rc_to_pos(r, c, self.grid_size)
                if pos == self._position:
                    row_chars.append("A")
                elif is_exit(pos, self.grid_size):
                    row_chars.append("E")
                elif pos in rock_set:
                    j = rock_set[pos]
                    if self._collected & (1 << j):
                        row_chars.append("x")  # collected
                    elif self.qualities[theta, j] == 1.0:
                        ch = "G*" if self._scanned_mask & (1 << j) else "G"
                        row_chars.append(ch)
                    else:
                        ch = "B*" if self._scanned_mask & (1 << j) else "B"
                        row_chars.append(ch)
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)
