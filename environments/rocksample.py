"""Canonical RockSample[n,k] environment with per-rock SENSE actions.

Grid world where the agent must collect good rocks and exit the east edge.
Rocks have unknown quality (good/bad). The agent can SENSE individual rocks
(noisy, distance-dependent readings), SAMPLE the rock at its current position
(which reveals its quality), and move in four directions.

Observations are event-gated: rock channel r only emits an informative
reading when the last action was SENSE_r (or a SAMPLE that revealed rock r);
otherwise the channel emits a NO_INFO outcome. This removes the old "magic
scan" oracle — information must be gathered one noisy reading at a time.

Static state θ represents rock quality configurations (quality-only,
exhaustive: n_static = 2^k).

State space:
    x: state_index(pos, mask, event, n_pos, n_mask, n_events)
    pos = row * grid_size + col (row-major)
    mask = 0..2^k-1 (bitmask of sampled/collected rocks)
    event = 0 (OTHER) | 1+r (SENSE_r) | k+1 (SAMPLE) — last action's role;
            movement and no-op SAMPLEs reset it to OTHER
    Total states: n_pos × 2^k × (k+2)

Actions (k+5):
    0: LEFT, 1: DOWN, 2: RIGHT, 3: UP, 4..3+k: SENSE_r, 4+k: SAMPLE

Observations:
    Two modalities, three outcomes per channel (0, 1, NO_INFO=2):
    1. Position channels (0..n_pos-1): θ-independent, fixed noise
       (outcome 2 never emitted).
    2. Rock quality channels (n_pos..n_pos+k-1): θ-dependent, event-gated.
       SENSE_r: accuracy α(d) = 0.5 + 0.5 * 2^(-d / half_eff_dist) with
       d = Chebyshev distance from agent to rock r.
       SAMPLE-reveal (at rock r's cell, bit r set): near-deterministic.
       Otherwise: NO_INFO with probability 1.
    Observation tensor shape: (n_pos + k, 3, n_states, n_configs)
"""

import numpy as np
from dataclasses import dataclass

# Movement actions; SENSE_r = 4 + r for r in 0..k-1; SAMPLE = 4 + k
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
N_MOVEMENT_ACTIONS = 4

# Event encoding: EVENT_SENSE_r = 1 + r for r in 0..k-1
EVENT_OTHER = 0

# Movement deltas: (delta_row, delta_col)
MOVEMENT = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}

# Rock channel outcomes
ROCK_BIT_0 = 0
ROCK_BIT_1 = 1
ROCK_NO_INFO = 2
N_OBS_TYPES = 3


def n_actions_for(n_rocks: int) -> int:
    """Number of actions: 4 moves + k per-rock senses + 1 sample."""
    return N_MOVEMENT_ACTIONS + n_rocks + 1


def sense_action(rock: int) -> int:
    """Action index of SENSE_rock."""
    return N_MOVEMENT_ACTIONS + rock


def sample_action(n_rocks: int) -> int:
    """Action index of SAMPLE."""
    return N_MOVEMENT_ACTIONS + n_rocks


def n_events_for(n_rocks: int) -> int:
    """Number of event values: OTHER + k senses + SAMPLE."""
    return n_rocks + 2


def event_sense(rock: int) -> int:
    """Event value for SENSE_rock."""
    return 1 + rock


def event_sample(n_rocks: int) -> int:
    """Event value for SAMPLE."""
    return n_rocks + 1


def state_index(pos: int, mask: int, event: int,
                n_pos: int, n_mask: int, n_events: int) -> int:
    """Compute flat state index.

    x = pos + mask * n_pos + event * n_pos * n_mask
    """
    return pos + mask * n_pos + event * n_pos * n_mask


def unpack_state(x: int, n_pos: int, n_mask: int, n_events: int) -> tuple[int, int, int]:
    """Unpack flat state index into (pos, mask, event)."""
    event = x // (n_pos * n_mask)
    remainder = x % (n_pos * n_mask)
    mask = remainder // n_pos
    pos = remainder % n_pos
    return pos, mask, event


def pos_to_rc(pos: int, grid_size: int) -> tuple[int, int]:
    """Convert flat position to (row, col)."""
    return divmod(pos, grid_size)


def rc_to_pos(row: int, col: int, grid_size: int) -> int:
    """Convert (row, col) to flat position."""
    return row * grid_size + col


def chebyshev_distance(pos_a: int, pos_b: int, grid_size: int) -> int:
    """Chebyshev distance between two grid positions (canonical RockSample)."""
    ra, ca = pos_to_rc(pos_a, grid_size)
    rb, cb = pos_to_rc(pos_b, grid_size)
    return max(abs(ra - rb), abs(ca - cb))


def sense_accuracy(distance: float, half_eff_dist: float) -> float:
    """Distance-dependent sensor accuracy α(d) = 0.5 + 0.5 * 2^(-d/d0)."""
    return 0.5 + 0.5 * (2.0 ** (-distance / half_eff_dist))


def is_exit(pos: int, grid_size: int) -> bool:
    """Check if position is in the exit column (rightmost)."""
    _, col = pos_to_rc(pos, grid_size)
    return col == grid_size - 1


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

    Semantics:
        Movement: slips among the 4 moves only; resets event to OTHER.
        SENSE_r: deterministic, keeps position/mask, sets event to 1+r.
        SAMPLE at uncollected rock r's cell: sets mask bit r, event SAMPLE.
        SAMPLE elsewhere / on a collected rock: no-op, resets event to OTHER.
        Exit column: absorbing for all actions (event preserved).

    Args:
        grid_size: Grid size
        rock_positions: (k,) rock positions
        n_rocks: Number of rocks
        slip_prob: Movement slip probability
    Returns:
        T: (n_states, n_states, 2^k, k+5) transition tensor
    """
    n_pos = grid_size * grid_size
    n_mask = 2 ** n_rocks
    n_events = n_events_for(n_rocks)
    n_configs = n_mask  # exhaustive: one config per quality assignment
    n_states = n_pos * n_mask * n_events
    n_actions = n_actions_for(n_rocks)

    # Build rock position lookup: pos -> rock index
    rock_at_pos = {}
    for j, rp in enumerate(rock_positions):
        rock_at_pos[int(rp)] = j

    # Exit positions
    exit_positions = set()
    for r in range(grid_size):
        exit_positions.add(rc_to_pos(r, grid_size - 1, grid_size))

    # Build for θ=0 then tile
    T_single = np.zeros((n_states, n_states, 1, n_actions), dtype=np.float32)

    for x_old in range(n_states):
        pos_old, mask_old, _ = unpack_state(x_old, n_pos, n_mask, n_events)

        # Exit positions are absorbing
        if pos_old in exit_positions:
            T_single[x_old, x_old, 0, :] = 1.0
            continue

        # Movement actions (0-3): reset event to OTHER
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

                x_new = state_index(pos_new, mask_old, EVENT_OTHER,
                                    n_pos, n_mask, n_events)
                T_single[x_new, x_old, 0, intended] += prob

        # SENSE_r actions: deterministic, set event to 1+r
        for r in range(n_rocks):
            x_new = state_index(pos_old, mask_old, event_sense(r),
                                n_pos, n_mask, n_events)
            T_single[x_new, x_old, 0, sense_action(r)] = 1.0

        # SAMPLE: at uncollected rock -> set mask bit + SAMPLE event;
        # otherwise no-op with event reset to OTHER
        a_sample = sample_action(n_rocks)
        if pos_old in rock_at_pos and not (mask_old & (1 << rock_at_pos[pos_old])):
            j = rock_at_pos[pos_old]
            mask_new = mask_old | (1 << j)
            x_new = state_index(pos_old, mask_new, event_sample(n_rocks),
                                n_pos, n_mask, n_events)
        else:
            x_new = state_index(pos_old, mask_old, EVENT_OTHER,
                                n_pos, n_mask, n_events)
        T_single[x_new, x_old, 0, a_sample] = 1.0

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

    Two modalities with three outcomes (0, 1, NO_INFO=2):
    1. Position channels (0..n_pos-1): θ-independent, fixed noise.
       Outcome 2 has probability 0.
    2. Rock quality channels (n_pos..n_pos+k-1): θ-dependent, event-gated.
       event == SENSE_r: emits rock r's quality bit with accuracy
           α(d) = 0.5 + 0.5 * 2^(-d / half_eff_dist), d = Chebyshev distance.
       event == SAMPLE at rock r's cell with bit r set: near-deterministic
           reveal of rock r's quality.
       otherwise: NO_INFO with probability 1.

    Args:
        grid_size: Grid size
        rock_positions: (k,) rock positions
        qualities: (n_configs, k) binary rock quality
        n_rocks: Number of rocks
        half_eff_dist: Distance d0 at which accuracy halves toward 0.5
        pos_noise: Position channel noise

    Returns:
        B: (n_pos + k, 3, n_states, n_configs)
    """
    n_pos = grid_size * grid_size
    n_mask = 2 ** n_rocks
    n_events = n_events_for(n_rocks)
    n_states = n_pos * n_mask * n_events
    n_configs = qualities.shape[0]
    n_channels = n_pos + n_rocks

    p_tp_pos = np.clip(1.0 - pos_noise, 0.01, 0.99)
    p_fp_pos = np.clip(pos_noise * 0.1, 0.01, 0.99)
    p_reveal = 0.999

    B = np.zeros((n_channels, N_OBS_TYPES, n_states, n_configs), dtype=np.float32)

    # --- Position channels (θ-independent, fixed noise, never NO_INFO) ---
    for target_pos in range(n_pos):
        ch = target_pos
        for x in range(n_states):
            pos, _, _ = unpack_state(x, n_pos, n_mask, n_events)
            p = p_tp_pos if pos == target_pos else p_fp_pos
            B[ch, 1, x, :] = p
            B[ch, 0, x, :] = 1.0 - p

    # --- Rock quality channels (θ-dependent, event-gated) ---
    for j in range(n_rocks):
        ch = n_pos + j
        rock_pos = int(rock_positions[j])
        ev_sense_j = event_sense(j)
        ev_sample = event_sample(n_rocks)

        for x in range(n_states):
            pos, mask, event = unpack_state(x, n_pos, n_mask, n_events)

            if event == ev_sense_j:
                # Noisy distance-dependent reading of rock j
                d = chebyshev_distance(pos, rock_pos, grid_size)
                alpha = np.clip(sense_accuracy(d, half_eff_dist), 0.01, 0.99)
                for theta in range(n_configs):
                    q = int(qualities[theta, j])
                    B[ch, q, x, theta] = alpha
                    B[ch, 1 - q, x, theta] = 1.0 - alpha
            elif event == ev_sample and pos == rock_pos and (mask & (1 << j)):
                # Sampling rock j just revealed its quality
                for theta in range(n_configs):
                    q = int(qualities[theta, j])
                    B[ch, q, x, theta] = p_reveal
                    B[ch, 1 - q, x, theta] = 1.0 - p_reveal
            else:
                # No information about rock j this step
                B[ch, ROCK_NO_INFO, x, :] = 1.0

    return B


def generate_goal(
    grid_size: int,
    rock_positions: np.ndarray,
    qualities: np.ndarray,
    n_rocks: int,
    good_logit: float = 2.0,
    bad_logit: float = 4.0,
    exit_logit: float = 2.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Generate per-config preference factor C(x, θ) via softmax over logits.

    logits(x, θ) = good_logit * n_good(mask, θ)
                 - bad_logit * n_bad(mask, θ)
                 + exit_logit * [pos is exit]

    The asymmetry bad_logit > good_logit is the canonical guard against
    sampling-for-information: under a uniform quality belief the expected
    logit of sampling is negative, so SAMPLE only pays off once the agent
    actually believes a rock is good. The goal is flat over the event
    component (sensing is valued epistemically, never via preference).

    Args:
        grid_size: Grid size
        rock_positions: (k,) rock positions
        qualities: (n_configs, k) binary rock quality
        n_rocks: Number of rocks
        good_logit: Logit per collected good rock
        bad_logit: Logit penalty per collected bad rock
        exit_logit: Logit for being at the exit column
        temperature: Softmax temperature

    Returns:
        goal: (n_states, n_configs) per-config softmax preference
    """
    n_pos = grid_size * grid_size
    n_mask = 2 ** n_rocks
    n_events = n_events_for(n_rocks)
    n_states = n_pos * n_mask * n_events
    n_configs = qualities.shape[0]

    logits = np.zeros((n_states, n_configs), dtype=np.float64)

    for theta in range(n_configs):
        for x in range(n_states):
            pos, mask, _ = unpack_state(x, n_pos, n_mask, n_events)

            v = 0.0
            if is_exit(pos, grid_size):
                v += exit_logit

            for j in range(n_rocks):
                if mask & (1 << j):
                    if qualities[theta, j] == 1.0:
                        v += good_logit
                    else:
                        v -= bad_logit

            logits[x, theta] = v

    scaled = logits / temperature
    scaled -= scaled.max(axis=0, keepdims=True)
    goal = np.exp(scaled)
    goal /= goal.sum(axis=0, keepdims=True)
    return goal.astype(np.float32)


# ---------------------------------------------------------------------------
# Simple simulator
# ---------------------------------------------------------------------------


@dataclass
class RockSampleStepResult:
    obs: np.ndarray  # (n_pos + k,) per-channel outcome indices (0, 1, or 2)
    reward: float
    terminated: bool
    truncated: bool


class RockSampleEnv:
    """Simple canonical RockSample simulator.

    Args:
        grid_size: Grid size
        rock_positions: (k,) rock positions
        qualities: (n_configs, k) rock quality configs
        n_rocks: Number of rocks
        obs_tensor: (n_pos + k, 3, n_states, n_configs) observation tensor
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
        self.n_mask = 2 ** n_rocks
        self.n_events = n_events_for(n_rocks)
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
        self._mask = 0  # bitmask of sampled rocks
        self._event = EVENT_OTHER
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
            self._position, self._mask, self._event,
            self.n_pos, self.n_mask, self.n_events,
        )

    def reset(self, seed: int | None = None, config_idx: int | None = None) -> RockSampleStepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if config_idx is not None:
            self._config_idx = config_idx
        else:
            self._config_idx = int(self._rng.integers(0, self.qualities.shape[0]))

        self._position = self.start_pos
        self._mask = 0
        self._event = EVENT_OTHER
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

        a_sample = sample_action(self.n_rocks)

        if N_MOVEMENT_ACTIONS <= action < a_sample:
            # SENSE_r: no movement, set event
            r = action - N_MOVEMENT_ACTIONS
            self._event = event_sense(r)
        elif action == a_sample:
            if (self._position in self._rock_at_pos
                    and not (self._mask & (1 << self._rock_at_pos[self._position]))):
                j = self._rock_at_pos[self._position]
                self._mask |= (1 << j)
                self._event = event_sample(self.n_rocks)
                rock_good = self.qualities[self._config_idx, j] == 1.0
                reward = self.good_reward if rock_good else -self.bad_penalty
            else:
                # No rock or already collected: no-op
                self._event = EVENT_OTHER
        else:
            # Movement: resets event
            self._event = EVENT_OTHER
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
        """Sample per-channel categorical outcomes from the observation model."""
        x = self._state_idx
        if self.obs_tensor is not None:
            n_channels = self.obs_tensor.shape[0]
            n_obs = self.obs_tensor.shape[1]
            obs = np.zeros(n_channels, dtype=np.float32)
            for c in range(n_channels):
                p = np.asarray(self.obs_tensor[c, :, x, self._config_idx], dtype=np.float64)
                p = p / p.sum()
                obs[c] = float(self._rng.choice(n_obs, p=p))
            return obs
        else:
            # Fallback: deterministic position one-hot, rocks NO_INFO unless
            # the last event reveals them
            n_channels = self.n_pos + self.n_rocks
            obs = np.full(n_channels, float(ROCK_NO_INFO), dtype=np.float32)
            obs[:self.n_pos] = 0.0
            obs[self._position] = 1.0
            for j in range(self.n_rocks):
                if self._event == event_sense(j) or (
                    self._event == event_sample(self.n_rocks)
                    and self._position == int(self.rock_positions[j])
                    and (self._mask & (1 << j))
                ):
                    obs[self.n_pos + j] = float(
                        self.qualities[self._config_idx, j] == 1.0
                    )
            return obs

    def render_ascii(self) -> str:
        """Render current state as ASCII grid."""
        lines = []
        theta = self._config_idx
        mask_str = bin(self._mask)[2:].zfill(self.n_rocks)
        if self._event == EVENT_OTHER:
            event_str = "other"
        elif self._event == event_sample(self.n_rocks):
            event_str = "sample"
        else:
            event_str = f"sense_{self._event - 1}"
        lines.append(f"[sampled={mask_str}] last_event={event_str}")

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
                    if self._mask & (1 << j):
                        row_chars.append("x")  # collected
                    elif self.qualities[theta, j] == 1.0:
                        row_chars.append("G")
                    else:
                        row_chars.append("B")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)
