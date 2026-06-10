from enum import IntEnum
import numpy as np


class ActionType(IntEnum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6


class CellType(IntEnum):
    UNSEEN = 0
    EMPTY = 1
    WALL = 2
    FLOOR = 3
    DOOR = 4
    KEY = 5
    BALL = 6
    BOX = 7
    GOAL = 8
    LAVA = 9
    AGENT = 10


class Orientation(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


N_CELL_TYPES = 11
N_ORIENTATIONS = 4
N_ACTIONS = 7
N_DOOR_KEY_STATES = 3


def state_to_coords(s: int, n: int) -> tuple[int, int]:
    return (s // n, s % n)


def coords_to_state(x: int, y: int, n: int) -> int:
    return x * n + y


def flatten_state_index(
    state: int,
    orientation: int,
    door_key_state: int,
    n_states: int,
    n_orientations: int,
    n_door_key_states: int,
) -> int:
    return (
        state * (n_orientations * n_door_key_states)
        + orientation * n_door_key_states
        + door_key_state
    )


def unflatten_state_index(
    flat_idx: int, n_states: int, n_orientations: int, n_door_key_states: int
) -> tuple[int, int, int]:
    door_key_state = flat_idx % n_door_key_states
    flat_idx = flat_idx // n_door_key_states
    orientation = flat_idx % n_orientations
    state = flat_idx // n_orientations
    return (state, orientation, door_key_state)


def key_position(key_pos: int, n: int) -> tuple[int, int]:
    return (key_pos // n, key_pos % n)


def door_position(door_pos: int, n: int) -> tuple[int, int]:
    return (door_pos // n + 1, door_pos % n)


def get_valid_static_configs(n: int) -> list[tuple[int, int]]:
    """Return list of (key_pos, door_pos) where key_x < door_x."""
    n_positions = n * n - 2 * n
    configs = []
    for key_pos in range(n_positions):
        key_x, _ = key_position(key_pos, n)
        for door_pos in range(n_positions):
            door_x, _ = door_position(door_pos, n)
            if key_x < door_x:
                configs.append((key_pos, door_pos))
    return configs


def get_relative_coords(
    agent_x: int, agent_y: int, orientation: int, target_x: int, target_y: int
) -> tuple[int, int]:
    dx = target_x - agent_x
    dy = target_y - agent_y

    if orientation == Orientation.RIGHT:
        return (-dy, dx)
    elif orientation == Orientation.DOWN:
        return (dx, dy)
    elif orientation == Orientation.LEFT:
        return (dy, -dx)
    else:  # UP
        return (-dx, -dy)


def in_fov(rel_x: int, rel_y: int, fov_size: int = 7) -> bool:
    half = fov_size // 2
    return -half <= rel_x <= half and 0 <= rel_y <= fov_size - 1


def relative_to_fov_coords(rel_x: int, rel_y: int, fov_size: int = 7) -> tuple[int, int]:
    half = fov_size // 2
    fov_x = half - rel_x  # Agent at column half (0-indexed)
    fov_y = (fov_size - 1) - rel_y  # Agent at last row (0-indexed)
    return (fov_x, fov_y)


def relative_to_absolute_coords(
    agent_x: int, agent_y: int, orientation: int, rel_x: int, rel_y: int
) -> tuple[int, int]:
    if orientation == Orientation.RIGHT:
        dx = rel_y
        dy = -rel_x
    elif orientation == Orientation.DOWN:
        dx = rel_x
        dy = rel_y
    elif orientation == Orientation.LEFT:
        dx = -rel_y
        dy = rel_x
    else:  # UP
        dx = -rel_x
        dy = -rel_y

    return (agent_x + dx, agent_y + dy)


def generate_visibility_mask(
    agent_x: int, agent_y: int, width: int, height: int, walls: set[tuple[int, int]]
) -> np.ndarray:
    mask = np.zeros((width, height), dtype=bool)
    mask[agent_x, agent_y] = True

    for j in range(height - 1, -1, -1):
        for i in range(width - 1):
            if not mask[i, j]:
                continue
            if (i, j) in walls:
                continue
            mask[i + 1, j] = True
            if j > 0:
                mask[i + 1, j - 1] = True
                mask[i, j - 1] = True

        for i in range(width - 1, 0, -1):
            if not mask[i, j]:
                continue
            if (i, j) in walls:
                continue
            mask[i - 1, j] = True
            if j > 0:
                mask[i - 1, j - 1] = True
                mask[i, j - 1] = True

    return mask


def create_wall_set(door_x: int, door_y: int, n: int) -> set[tuple[int, int]]:
    walls = set()
    for y in range(n):
        if y != door_y:
            walls.add((door_x, y))
    for x in range(n):
        walls.add((x, -1))
        walls.add((x, n))
    for y in range(n):
        walls.add((-1, y))
        walls.add((n, y))
    walls.add((-1, -1))
    walls.add((n, -1))
    walls.add((-1, n))
    walls.add((n, n))
    return walls


def get_fov(
    agent_x: int,
    agent_y: int,
    orientation: int,
    key_x: int,
    key_y: int,
    door_x: int,
    door_y: int,
    door_key_state: int,
    n: int,
    fov_size: int = 7,
) -> np.ndarray:
    half = fov_size // 2
    fov = np.full((fov_size, fov_size), CellType.EMPTY, dtype=np.int32)
    walls = create_wall_set(door_x, door_y, n)

    for wall_x, wall_y in walls:
        rel_wall = get_relative_coords(agent_x, agent_y, orientation, wall_x, wall_y)
        if in_fov(*rel_wall, fov_size):
            fov_x, fov_y = relative_to_fov_coords(*rel_wall, fov_size)
            fov[fov_x, fov_y] = CellType.WALL

    # Place goal
    goal_x, goal_y = n - 1, n - 1
    rel_goal = get_relative_coords(agent_x, agent_y, orientation, goal_x, goal_y)
    if in_fov(*rel_goal, fov_size):
        fov_x, fov_y = relative_to_fov_coords(*rel_goal, fov_size)
        fov[fov_x, fov_y] = CellType.GOAL

    # Place key on ground (only when not carrying)
    if door_key_state == 0:
        rel_key = get_relative_coords(agent_x, agent_y, orientation, key_x, key_y)
        if in_fov(*rel_key, fov_size):
            fov_x, fov_y = relative_to_fov_coords(*rel_key, fov_size)
            fov[fov_x, fov_y] = CellType.KEY

    # Place door
    rel_door = get_relative_coords(agent_x, agent_y, orientation, door_x, door_y)
    if in_fov(*rel_door, fov_size):
        fov_x, fov_y = relative_to_fov_coords(*rel_door, fov_size)
        fov[fov_x, fov_y] = CellType.DOOR

    # Carried key at agent position — last so it overrides door when agent is on door cell
    if door_key_state >= 1:
        fov[half, fov_size - 1] = CellType.KEY

    if door_key_state != 2:  # Door not open - blocks visibility
        walls.add((door_x, door_y))

    relative_walls = set()
    for wall in walls:
        rel = get_relative_coords(agent_x, agent_y, orientation, wall[0], wall[1])
        if in_fov(*rel, fov_size):
            relative_walls.add(relative_to_fov_coords(*rel, fov_size))

    visibility_mask = generate_visibility_mask(half, fov_size - 1, fov_size, fov_size, relative_walls)
    for x in range(-half, half + 1):
        for y in range(fov_size):
            fov_x, fov_y = relative_to_fov_coords(x, y, fov_size)
            if not visibility_mask[fov_x, fov_y]:
                fov[fov_x, fov_y] = CellType.UNSEEN

    return fov


def get_next_orientation(orientation: int, action: int) -> int:
    if action == ActionType.TURN_LEFT:
        return (orientation + 3) % 4  # Counter-clockwise
    elif action == ActionType.TURN_RIGHT:
        return (orientation + 1) % 4  # Clockwise
    else:
        return orientation


def get_next_door_key_state(
    agent_x: int,
    agent_y: int,
    orientation: int,
    key_x: int,
    key_y: int,
    door_x: int,
    door_y: int,
    action: int,
    door_key_state: int,
) -> int:
    if action == ActionType.PICKUP:
        if door_key_state > 0:
            return door_key_state
        rel_x, rel_y = get_relative_coords(agent_x, agent_y, orientation, key_x, key_y)
        if rel_x == 0 and rel_y == 1:
            return 1  # Key picked up
        return door_key_state

    if action != ActionType.TOGGLE:
        return door_key_state

    if door_key_state != 1:  # Need to have key but door not open
        return door_key_state

    rel_x, rel_y = get_relative_coords(agent_x, agent_y, orientation, door_x, door_y)
    if rel_x == 0 and rel_y == 1:
        return 2  # Door opened
    return door_key_state


def get_next_agent_position(
    agent_x: int,
    agent_y: int,
    orientation: int,
    door_x: int,
    door_y: int,
    key_x: int,
    key_y: int,
    door_key_state: int,
    action: int,
    n: int,
) -> int:
    if action == ActionType.FORWARD:
        new_x, new_y = agent_x, agent_y
        if orientation == Orientation.RIGHT:
            new_x += 1
        elif orientation == Orientation.DOWN:
            new_y += 1
        elif orientation == Orientation.LEFT:
            new_x -= 1
        else:  # UP
            new_y -= 1

        if new_x < 0 or new_x >= n or new_y < 0 or new_y >= n:
            return coords_to_state(agent_x, agent_y, n)
        if new_x == key_x and new_y == key_y and door_key_state == 0:
            return coords_to_state(agent_x, agent_y, n)
        if new_x == door_x and door_key_state != 2:
            return coords_to_state(agent_x, agent_y, n)
        if new_x == door_x and new_y != door_y:
            return coords_to_state(agent_x, agent_y, n)
        return coords_to_state(new_x, new_y, n)
    else:
        return coords_to_state(agent_x, agent_y, n)


def generate_observation_tensor(n: int, valid_configs: list[tuple[int, int]], fov_size: int = 7, dtype=np.float16) -> np.ndarray:
    """Generate full observation tensor (memory-intensive, for reference/testing)."""
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = len(valid_configs)

    B = np.zeros((fov_size, fov_size, N_CELL_TYPES, n_total_states, n_static_states), dtype=dtype)

    for agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for static_idx, (key_pos, door_pos) in enumerate(valid_configs):
                key_x, key_y = key_position(key_pos, n)
                door_x, door_y = door_position(door_pos, n)

                for door_key_state in range(N_DOOR_KEY_STATES):
                    fov = get_fov(
                        agent_x,
                        agent_y,
                        orientation,
                        key_x,
                        key_y,
                        door_x,
                        door_y,
                        door_key_state,
                        n,
                        fov_size,
                    )
                    flat_state = flatten_state_index(
                        agent_state,
                        orientation,
                        door_key_state,
                        n_location_states,
                        N_ORIENTATIONS,
                        N_DOOR_KEY_STATES,
                    )
                    for i in range(fov_size):
                        for j in range(fov_size):
                            B[i, j, fov[i, j], flat_state, static_idx] = 1.0

    return B


def soften_observation_tensor(B_hard: np.ndarray, fov_size: int, alpha: float) -> np.ndarray:
    """Soften observation tensor based on Manhattan distance from the cell in front of the agent.

    Nearby cells keep high precision (near one-hot), distant cells approach uniform.
    UNSEEN entries (occluded cells) are preserved unchanged.

    Args:
        B_hard: Hard one-hot observation tensor, shape (fov, fov, N_CELL_TYPES, n_states, n_static).
        fov_size: FOV grid size (must be odd).
        alpha: Noise rate per unit Manhattan distance. 0 = no softening.

    Returns:
        Softened observation tensor with same shape and dtype as B_hard.
    """
    if alpha == 0.0:
        return B_hard

    half = fov_size // 2
    ref_x, ref_y = half, fov_size - 2  # cell directly in front of agent
    agent_x, agent_y = half, fov_size - 1  # agent position

    # Manhattan distance grid: min distance to agent or cell in front
    ix = np.arange(fov_size)
    dist_ref = np.abs(ix[:, None] - ref_x) + np.abs(ref_y - ix[None, :])
    dist_agent = np.abs(ix[:, None] - agent_x) + np.abs(agent_y - ix[None, :])
    dist = np.minimum(dist_ref, dist_agent)  # (fov, fov)

    precision = np.maximum(0.0, 1.0 - alpha * dist)  # (fov, fov)

    # Broadcast to (fov, fov, 1, 1, 1) for element-wise ops
    p = precision[:, :, None, None, None]
    uniform = np.float64(1.0 / N_CELL_TYPES)

    B_soft = (p * B_hard + (1.0 - p) * uniform).astype(B_hard.dtype)

    return B_soft


def generate_orientation_observation_tensor(n: int, dtype=np.float16) -> np.ndarray:
    """Generate full orientation observation tensor (for reference/testing)."""
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES

    B = np.zeros((N_ORIENTATIONS, n_total_states), dtype=dtype)

    for state_idx in range(n_total_states):
        _, orientation, _ = unflatten_state_index(
            state_idx, n_location_states, N_ORIENTATIONS, N_DOOR_KEY_STATES
        )
        B[orientation, state_idx] = 1.0

    return B



def generate_transition_tensor(n: int, valid_configs: list[tuple[int, int]], dtype=np.float16) -> np.ndarray:
    """Generate full transition tensor (memory-intensive, for reference/testing)."""
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = len(valid_configs)

    T = np.zeros(
        (n_total_states, n_total_states, n_static_states, N_ACTIONS), dtype=dtype
    )

    for old_agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(old_agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for door_key_state in range(N_DOOR_KEY_STATES):
                for static_idx, (key_pos, door_pos) in enumerate(valid_configs):
                    key_x, key_y = key_position(key_pos, n)
                    door_x, door_y = door_position(door_pos, n)

                    old_idx = flatten_state_index(
                        old_agent_state,
                        orientation,
                        door_key_state,
                        n_location_states,
                        N_ORIENTATIONS,
                        N_DOOR_KEY_STATES,
                    )

                    if agent_x == door_x and agent_y != door_y:
                        T[old_idx, old_idx, static_idx, :] = 1.0
                        continue

                    for action in range(N_ACTIONS):
                        new_agent_state = get_next_agent_position(
                            agent_x,
                            agent_y,
                            orientation,
                            door_x,
                            door_y,
                            key_x,
                            key_y,
                            door_key_state,
                            action,
                            n,
                        )
                        new_door_key_state = get_next_door_key_state(
                            agent_x,
                            agent_y,
                            orientation,
                            key_x,
                            key_y,
                            door_x,
                            door_y,
                            action,
                            door_key_state,
                        )
                        new_orientation = get_next_orientation(orientation, action)
                        new_idx = flatten_state_index(
                            new_agent_state,
                            new_orientation,
                            new_door_key_state,
                            n_location_states,
                            N_ORIENTATIONS,
                            N_DOOR_KEY_STATES,
                        )
                        T[new_idx, old_idx, static_idx, action] = 1.0

    return T



def observation_to_onehot(image: np.ndarray) -> np.ndarray:
    fov_w, fov_h = image.shape[0], image.shape[1]
    onehot = np.zeros((fov_w, fov_h, N_CELL_TYPES), dtype=np.float16)
    for i in range(fov_w):
        for j in range(fov_h):
            onehot[i, j, image[i, j]] = 1.0
    return onehot


def direction_to_onehot(direction: int) -> np.ndarray:
    onehot = np.zeros(N_ORIENTATIONS, dtype=np.float16)
    onehot[direction] = 1.0
    return onehot


def action_to_onehot(action: int) -> np.ndarray:
    onehot = np.zeros(N_ACTIONS, dtype=np.float16)
    onehot[action] = 1.0
    return onehot


def convert_action(action: int) -> int:
    return action


def contains_key(image: np.ndarray) -> bool:
    return CellType.KEY in image[:, :, 0]


def contains_door(image: np.ndarray) -> bool:
    return CellType.DOOR in image[:, :, 0]
