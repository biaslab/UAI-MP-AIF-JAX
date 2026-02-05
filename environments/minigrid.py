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


def flatten_position_index(
    key_pos: int, door_pos: int, n_key_positions: int, n_door_positions: int
) -> int:
    return key_pos * n_door_positions + door_pos


def unflatten_position_index(
    flat_idx: int, n_key_positions: int, n_door_positions: int
) -> tuple[int, int]:
    door_pos = flat_idx % n_door_positions
    key_pos = flat_idx // n_door_positions
    return (key_pos, door_pos)


def key_position(key_pos: int, n: int) -> tuple[int, int]:
    return (key_pos // n, key_pos % n)


def door_position(door_pos: int, n: int) -> tuple[int, int]:
    return (door_pos // n + 1, door_pos % n)


def get_relative_coords(
    agent_x: int, agent_y: int, orientation: int, target_x: int, target_y: int
) -> tuple[int, int]:
    dx = target_x - agent_x
    dy = target_y - agent_y

    if orientation == Orientation.RIGHT:
        return (-dy, dx)
    elif orientation == Orientation.DOWN:
        return (-dx, -dy)
    elif orientation == Orientation.LEFT:
        return (dy, -dx)
    else:  # UP
        return (dx, dy)


def in_fov(rel_x: int, rel_y: int) -> bool:
    return -3 <= rel_x <= 3 and 0 <= rel_y <= 6


def relative_to_fov_coords(rel_x: int, rel_y: int) -> tuple[int, int]:
    fov_x = 3 + rel_x  # Agent at column 3 (0-indexed)
    fov_y = 6 - rel_y  # Agent at row 6 (0-indexed)
    return (fov_x, fov_y)


def relative_to_absolute_coords(
    agent_x: int, agent_y: int, orientation: int, rel_x: int, rel_y: int
) -> tuple[int, int]:
    if orientation == Orientation.RIGHT:
        dx = rel_y
        dy = -rel_x
    elif orientation == Orientation.DOWN:
        dx = -rel_x
        dy = -rel_y
    elif orientation == Orientation.LEFT:
        dx = -rel_y
        dy = rel_x
    else:  # UP
        dx = rel_x
        dy = rel_y

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
) -> np.ndarray:
    fov = np.full((7, 7), CellType.EMPTY, dtype=np.int32)
    walls = create_wall_set(door_x, door_y, n)

    for wall_x, wall_y in walls:
        rel_wall = get_relative_coords(agent_x, agent_y, orientation, wall_x, wall_y)
        if in_fov(*rel_wall):
            fov_x, fov_y = relative_to_fov_coords(*rel_wall)
            fov[fov_x, fov_y] = CellType.WALL

    if door_key_state == 0:  # Don't have the key
        rel_key = get_relative_coords(agent_x, agent_y, orientation, key_x, key_y)
        if in_fov(*rel_key):
            fov_x, fov_y = relative_to_fov_coords(*rel_key)
            fov[fov_x, fov_y] = CellType.KEY
    else:  # Have the key - it appears at agent position
        fov[3, 6] = CellType.KEY

    rel_door = get_relative_coords(agent_x, agent_y, orientation, door_x, door_y)
    if in_fov(*rel_door):
        fov_x, fov_y = relative_to_fov_coords(*rel_door)
        fov[fov_x, fov_y] = CellType.DOOR

    if door_key_state != 2:  # Door not open - blocks visibility
        walls.add((door_x, door_y))

    relative_walls = set()
    for wall in walls:
        rel = get_relative_coords(agent_x, agent_y, orientation, wall[0], wall[1])
        if in_fov(*rel):
            relative_walls.add(relative_to_fov_coords(*rel))

    visibility_mask = generate_visibility_mask(3, 6, 7, 7, relative_walls)
    for x in range(-3, 4):
        for y in range(7):
            fov_x, fov_y = relative_to_fov_coords(x, y)
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
            new_y -= 1
        elif orientation == Orientation.LEFT:
            new_x -= 1
        else:  # UP
            new_y += 1

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


def generate_observation_tensor(n: int, dtype=np.float16) -> np.ndarray:
    """Generate full observation tensor (memory-intensive, for reference/testing)."""
    n_location_states = n * n
    n_key_positions = n_location_states - 2 * n
    n_door_positions = n_location_states - 2 * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = n_key_positions * n_door_positions

    B = np.zeros((7, 7, N_CELL_TYPES, n_total_states, n_static_states), dtype=dtype)

    for agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for key_pos in range(n_key_positions):
                key_x, key_y = key_position(key_pos, n)

                for door_pos in range(n_door_positions):
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
                        )
                        flat_state = flatten_state_index(
                            agent_state,
                            orientation,
                            door_key_state,
                            n_location_states,
                            N_ORIENTATIONS,
                            N_DOOR_KEY_STATES,
                        )
                        flat_static = flatten_position_index(
                            key_pos, door_pos, n_key_positions, n_door_positions
                        )
                        for i in range(7):
                            for j in range(7):
                                B[i, j, fov[i, j], flat_state, flat_static] = 1.0

    return B


def generate_observation_indices(n: int) -> np.ndarray:
    """
    Generate index-based observation tensor (memory-efficient).
    
    Instead of storing B[fov_x, fov_y, cell_type, state, static] as one-hot,
    store obs_idx[fov_x, fov_y, state, static] -> cell_type directly.
    
    Returns:
        obs_idx: (7, 7, n_total_states, n_static_states) uint8 array
        where obs_idx[i, j, state, static] = expected cell type at FOV position (i,j)
    """
    n_location_states = n * n
    n_key_positions = n_location_states - 2 * n
    n_door_positions = n_location_states - 2 * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = n_key_positions * n_door_positions

    # Index array: for each (fov_x, fov_y, state, static), store the cell type
    obs_idx = np.zeros((7, 7, n_total_states, n_static_states), dtype=np.uint8)

    for agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for key_pos in range(n_key_positions):
                key_x, key_y = key_position(key_pos, n)

                for door_pos in range(n_door_positions):
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
                        )
                        flat_state = flatten_state_index(
                            agent_state,
                            orientation,
                            door_key_state,
                            n_location_states,
                            N_ORIENTATIONS,
                            N_DOOR_KEY_STATES,
                        )
                        flat_static = flatten_position_index(
                            key_pos, door_pos, n_key_positions, n_door_positions
                        )
                        obs_idx[:, :, flat_state, flat_static] = fov

    return obs_idx


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


def generate_orientation_indices(n: int) -> np.ndarray:
    """
    Generate index-based orientation tensor (memory-efficient).
    
    Instead of storing B[orientation, state] as one-hot,
    store ori_idx[state] -> orientation directly.
    
    Returns:
        ori_idx: (n_total_states,) uint8 array
        where ori_idx[state] = orientation of that state
    """
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES

    ori_idx = np.zeros(n_total_states, dtype=np.uint8)

    for state_idx in range(n_total_states):
        _, orientation, _ = unflatten_state_index(
            state_idx, n_location_states, N_ORIENTATIONS, N_DOOR_KEY_STATES
        )
        ori_idx[state_idx] = orientation

    return ori_idx


def generate_transition_tensor(n: int, dtype=np.float16) -> np.ndarray:
    """Generate full transition tensor (memory-intensive, for reference/testing)."""
    n_location_states = n * n
    n_key_positions = n_location_states - 2 * n
    n_door_positions = n_location_states - 2 * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = n_key_positions * n_door_positions

    T = np.zeros(
        (n_total_states, n_total_states, n_static_states, N_ACTIONS), dtype=dtype
    )

    for old_agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(old_agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for door_key_state in range(N_DOOR_KEY_STATES):
                for door_pos in range(n_door_positions):
                    door_x, door_y = door_position(door_pos, n)

                    for key_pos in range(n_key_positions):
                        key_x, key_y = key_position(key_pos, n)

                        static_idx = flatten_position_index(
                            key_pos, door_pos, n_key_positions, n_door_positions
                        )
                        old_idx = flatten_state_index(
                            old_agent_state,
                            orientation,
                            door_key_state,
                            n_location_states,
                            N_ORIENTATIONS,
                            N_DOOR_KEY_STATES,
                        )

                        if key_x == door_x or (
                            agent_x == door_x and agent_y != door_y
                        ):
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


def generate_transition_indices(n: int) -> np.ndarray:
    """
    Generate index-based transition tensor (memory-efficient).
    
    Instead of storing T[new_state, old_state, static, action] as a one-hot tensor,
    store next_state_idx[old_state, static, action] -> new_state directly.
    
    Returns:
        next_state_idx: (n_total_states, n_static_states, N_ACTIONS) int32 array
        where next_state_idx[old, static, action] = new_state
    """
    n_location_states = n * n
    n_key_positions = n_location_states - 2 * n
    n_door_positions = n_location_states - 2 * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = n_key_positions * n_door_positions

    # Index array: for each (old_state, static, action), store the new_state
    next_state_idx = np.zeros(
        (n_total_states, n_static_states, N_ACTIONS), dtype=np.int32
    )

    for old_agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(old_agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for door_key_state in range(N_DOOR_KEY_STATES):
                old_idx = flatten_state_index(
                    old_agent_state,
                    orientation,
                    door_key_state,
                    n_location_states,
                    N_ORIENTATIONS,
                    N_DOOR_KEY_STATES,
                )

                for door_pos in range(n_door_positions):
                    door_x, door_y = door_position(door_pos, n)

                    for key_pos in range(n_key_positions):
                        key_x, key_y = key_position(key_pos, n)

                        static_idx = flatten_position_index(
                            key_pos, door_pos, n_key_positions, n_door_positions
                        )

                        # Invalid configurations: stay in place
                        if key_x == door_x or (
                            agent_x == door_x and agent_y != door_y
                        ):
                            next_state_idx[old_idx, static_idx, :] = old_idx
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
                            next_state_idx[old_idx, static_idx, action] = new_idx

    return next_state_idx


def observation_to_onehot(image: np.ndarray) -> np.ndarray:
    onehot = np.zeros((7, 7, N_CELL_TYPES), dtype=np.float16)
    for i in range(7):
        for j in range(7):
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
