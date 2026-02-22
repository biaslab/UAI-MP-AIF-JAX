"""Wumpus World environment with hidden pits, wumpus, and SCAN action.

Grid world where the agent must reach the gold while avoiding hidden pits and
the wumpus. The agent perceives breeze (adjacent to pit), stench (adjacent to
wumpus), and glitter (on the gold cell), plus noisy position channels.
Static state θ represents sampled (pits, wumpus, gold) configurations.

State space (doubled for scan mode):
    x: state_index(pos, scanned, n_pos) where pos = row * grid_size + col
    mode 0 = unscanned (noisy observations), mode 1 = scanned (near-deterministic)
    Total states: 2 * n_pos

Actions:
    0: LEFT, 1: DOWN, 2: RIGHT, 3: UP, 4: SCAN

Observations:
    3 + n_pos binary channels:
        - channel 0: breeze  (0=no, 1=yes) — adjacent to a pit
        - channel 1: stench  (0=no, 1=yes) — adjacent to the wumpus
        - channel 2: glitter (0=no, 1=yes) — on the gold cell
        - channels 3..3+n_pos-1: position channels (θ-independent, noisy)
    Noisy when unscanned, near-deterministic when scanned.
    Observation tensor shape: (3 + n_pos, 2, 2*n_pos, n_static)
"""

import numpy as np
from dataclasses import dataclass

# Actions (same as Frozen Lake)
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
SCAN = 4
N_ACTIONS = 5
N_MOVEMENT_ACTIONS = 4

# Observation channels
BREEZE = 0
STENCH = 1
GLITTER = 2
N_FEATURE_CHANNELS = 3  # breeze, stench, glitter
N_OBS_TYPES = 2  # binary: 0 or 1

# Movement deltas: (delta_row, delta_col)
MOVEMENT = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}


def state_index(pos: int, scanned: int, n_pos: int) -> int:
    """Compute flat state index: x = pos + scanned * n_pos."""
    return pos + scanned * n_pos


def unpack_state(x: int, n_pos: int) -> tuple[int, int]:
    """Unpack flat state index into (pos, scanned)."""
    scanned = x // n_pos
    pos = x % n_pos
    return pos, scanned


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

    State space is doubled: x = pos + scanned * n_pos.
    Pits and wumpus cells are absorbing (death). Gold is NOT absorbing in T.
    Movement preserves scan mode. SCAN switches to scanned (deterministic, no slip).

    Args:
        grid_size: Grid size
        pits: (n_configs, n_pos) pit configurations
        wumpus: (n_configs, n_pos) wumpus configurations
        slip_prob: Movement noise probability (only affects movement actions)

    Returns:
        T: (2*n_pos, 2*n_pos, n_static, 5)
    """
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos
    n_static = pits.shape[0]

    T = np.zeros((n_states, n_states, n_static, N_ACTIONS), dtype=np.float32)

    for theta in range(n_static):
        for x_old in range(n_states):
            pos_old, scanned_old = unpack_state(x_old, n_pos)

            # Absorbing: pits and wumpus (in both modes)
            if pits[theta, pos_old] == 1.0 or wumpus[theta, pos_old] == 1.0:
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
                        pos_new = pos_old

                    x_new = state_index(pos_new, scanned_old, n_pos)
                    T[x_new, x_old, theta, intended_action] += prob

            # SCAN action (4): deterministic, no slip
            x_new_scan = state_index(pos_old, 1, n_pos)
            T[x_new_scan, x_old, theta, SCAN] = 1.0

    return T


def generate_observation_tensor(
    grid_size: int,
    pits: np.ndarray,
    wumpus: np.ndarray,
    gold: np.ndarray,
    obs_noise: float = 0.1,
    pos_noise: float = 0.1,
) -> np.ndarray:
    """Generate observation tensor B(channel, obs_type, x, θ).

    State space is doubled (2*n_pos) for scan mode.

    3 + n_pos binary observation channels:
        - breeze:  adjacent to pit
        - stench:  adjacent to wumpus
        - glitter: on gold cell
        - position channels: one per grid position (θ-independent)

    Unscanned mode: p_tp = 1 - noise, p_fp = noise * 0.1
    Scanned mode: p_tp = 0.999, p_fp = 0.001

    Args:
        grid_size: Grid size
        pits: (n_configs, n_pos) pit configs
        wumpus: (n_configs, n_pos) wumpus configs
        gold: (n_configs, n_pos) gold configs
        obs_noise: Noise level for feature channels (unscanned mode)
        pos_noise: Noise level for position channels (unscanned mode)

    Returns:
        B: (3 + n_pos, 2, 2*n_pos, n_static) observation tensor.
    """
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos
    n_static = pits.shape[0]
    n_channels = N_FEATURE_CHANNELS + n_pos

    # Unscanned noise parameters (feature channels)
    p_tp = np.clip(1.0 - obs_noise, 0.01, 0.99)
    p_fp = np.clip(obs_noise * 0.1, 0.01, 0.99)

    # Unscanned noise parameters (position channels)
    p_tp_pos = np.clip(1.0 - pos_noise, 0.01, 0.99)
    p_fp_pos = np.clip(pos_noise * 0.1, 0.01, 0.99)

    # Scanned (near-deterministic) parameters
    p_tp_s = 0.999
    p_fp_s = 0.001

    B = np.zeros((n_channels, N_OBS_TYPES, n_states, n_static), dtype=np.float32)

    # --- Feature channels (breeze, stench, glitter) — θ-dependent ---
    for theta in range(n_static):
        for pos in range(n_pos):
            neighbors = get_neighbors(pos, grid_size)

            has_breeze = any(pits[theta, n] == 1.0 for n in neighbors)
            has_stench = any(wumpus[theta, n] == 1.0 for n in neighbors)
            has_glitter = gold[theta, pos] == 1.0

            for scanned in range(2):
                x = state_index(pos, scanned, n_pos)
                if scanned == 0:
                    tp, fp = p_tp, p_fp
                else:
                    tp, fp = p_tp_s, p_fp_s

                p_b = tp if has_breeze else fp
                B[BREEZE, 1, x, theta] = p_b
                B[BREEZE, 0, x, theta] = 1.0 - p_b

                p_s = tp if has_stench else fp
                B[STENCH, 1, x, theta] = p_s
                B[STENCH, 0, x, theta] = 1.0 - p_s

                p_g = tp if has_glitter else fp
                B[GLITTER, 1, x, theta] = p_g
                B[GLITTER, 0, x, theta] = 1.0 - p_g

    # --- Position channels (θ-independent) ---
    for target_pos in range(n_pos):
        ch = N_FEATURE_CHANNELS + target_pos
        for pos in range(n_pos):
            for scanned in range(2):
                x = state_index(pos, scanned, n_pos)
                if scanned == 0:
                    p = p_tp_pos if pos == target_pos else p_fp_pos
                else:
                    p = p_tp_s if pos == target_pos else p_fp_s
                B[ch, 1, x, :] = p
                B[ch, 0, x, :] = 1.0 - p

    return B


def generate_goal(
    grid_size: int,
    pits: np.ndarray,
    wumpus: np.ndarray,
    gold: np.ndarray,
    gold_reward: float = 1.0,
    pit_penalty: float = 1.0,
    wumpus_penalty: float = 1.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Generate per-config preference factor C(x, θ) via softmax over rewards.

    State space is doubled (2*n_pos). Gold reward and pit/wumpus penalties
    apply in both scan modes. Each config θ gets its own reward vector based
    on which positions have gold, pits, and wumpus in that config.

    Args:
        grid_size: Grid size
        pits: (n_configs, n_pos) pit configurations
        wumpus: (n_configs, n_pos) wumpus configurations
        gold: (n_configs, n_pos) gold configurations
        gold_reward: Reward magnitude at gold position
        pit_penalty: Penalty magnitude for pit positions
        wumpus_penalty: Penalty magnitude for wumpus position
        temperature: Softmax temperature (lower = more peaked)

    Returns:
        goal: (2*n_pos, n_configs) per-config softmax preference
    """
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos
    n_static = pits.shape[0]

    rewards = np.zeros((n_states, n_static), dtype=np.float64)

    for theta in range(n_static):
        for mode in range(2):
            for pos in range(n_pos):
                idx = state_index(pos, mode, n_pos)
                if gold[theta, pos] == 1.0:
                    rewards[idx, theta] += gold_reward
                if pits[theta, pos] == 1.0:
                    rewards[idx, theta] -= pit_penalty
                if wumpus[theta, pos] == 1.0:
                    rewards[idx, theta] -= wumpus_penalty

    scaled = rewards / temperature
    scaled -= scaled.max(axis=0, keepdims=True)  # numerical stability per config
    goal = np.exp(scaled)
    goal /= goal.sum(axis=0, keepdims=True)
    return goal.astype(np.float32)


# ---------------------------------------------------------------------------
# Simple simulator
# ---------------------------------------------------------------------------


@dataclass
class WumpusStepResult:
    obs: np.ndarray  # (3 + n_pos,) binary observations
    reward: float
    terminated: bool
    truncated: bool


class WumpusWorldEnv:
    """Simple Wumpus World simulator with SCAN action.

    State includes scan mode: agent starts unscanned, SCAN action switches
    to scanned mode with near-deterministic observations.

    Args:
        grid_size: Grid size
        pits: (n_configs, n_pos) pit configurations
        wumpus: (n_configs, n_pos) wumpus configurations
        gold: (n_configs, n_pos) gold configurations
        obs_tensor: (n_channels, 2, 2*n_pos, n_static) observation tensor
        slip_prob: Movement noise (only affects movement actions, not SCAN)
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
        self.n_pos = grid_size * grid_size
        self.pits = pits
        self.wumpus = wumpus
        self.gold = gold
        self.obs_tensor = obs_tensor
        self.slip_prob = slip_prob
        self._max_steps = max_steps
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
        return self._config_idx

    @property
    def _state_idx(self) -> int:
        """Current flat state index (pos + scanned * n_pos)."""
        return state_index(self._position, self._scanned, self.n_pos)

    def reset(self, seed: int | None = None, config_idx: int | None = None) -> WumpusStepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if config_idx is not None:
            self._config_idx = config_idx
        else:
            self._config_idx = int(self._rng.integers(0, self.pits.shape[0]))

        self._position = self.start_pos
        self._scanned = 0
        self._steps = 0

        return WumpusStepResult(
            obs=self._get_obs(),
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    def step(self, action: int) -> WumpusStepResult:
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
        """Sample binary observations from obs model."""
        x = self._state_idx
        if self.obs_tensor is not None:
            n_channels = self.obs_tensor.shape[0]
            obs = np.zeros(n_channels, dtype=np.float32)
            for c in range(n_channels):
                p_fire = self.obs_tensor[c, 1, x, self._config_idx]
                obs[c] = float(self._rng.random() < p_fire)
            return obs
        else:
            # Fallback: deterministic (feature channels only)
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
        mode_str = "SCANNED" if self._scanned else "UNSCANNED"
        lines.append(f"[{mode_str}]")
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
