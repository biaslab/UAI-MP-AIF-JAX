import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from environments.minigrid import (
    flatten_state_index,
    unflatten_state_index,
    get_valid_static_configs,
    get_next_orientation,
    get_next_door_key_state,
    get_next_agent_position,
    state_to_coords,
    coords_to_state,
    key_position,
    door_position,
    get_relative_coords,
    in_fov,
    relative_to_fov_coords,
    get_fov,
    generate_observation_tensor,
    soften_observation_tensor,
    ActionType,
    Orientation,
    CellType,
    N_CELL_TYPES,
)


class TestIndexMappingFunctions:
    def test_flatten_unflatten_state_index_roundtrip(self):
        n_states = 4
        n_orientations = 4
        n_door_key_states = 3

        for state in range(n_states):
            for orientation in range(n_orientations):
                for door_key_state in range(n_door_key_states):
                    flat_idx = flatten_state_index(
                        state,
                        orientation,
                        door_key_state,
                        n_states,
                        n_orientations,
                        n_door_key_states,
                    )
                    assert flat_idx >= 0
                    assert flat_idx < n_states * n_orientations * n_door_key_states

                    state_out, orientation_out, door_key_state_out = (
                        unflatten_state_index(
                            flat_idx, n_states, n_orientations, n_door_key_states
                        )
                    )
                    assert state_out == state
                    assert orientation_out == orientation
                    assert door_key_state_out == door_key_state

    def test_flatten_state_index_unique(self):
        n_states = 4
        n_orientations = 4
        n_door_key_states = 3

        all_indices = set()
        for state in range(n_states):
            for orientation in range(n_orientations):
                for door_key_state in range(n_door_key_states):
                    flat_idx = flatten_state_index(
                        state,
                        orientation,
                        door_key_state,
                        n_states,
                        n_orientations,
                        n_door_key_states,
                    )
                    assert flat_idx not in all_indices
                    all_indices.add(flat_idx)

        assert len(all_indices) == n_states * n_orientations * n_door_key_states

    def test_flatten_state_index_boundary(self):
        n_states = 4
        n_orientations = 4
        n_door_key_states = 3

        assert (
            flatten_state_index(0, 0, 0, n_states, n_orientations, n_door_key_states)
            == 0
        )
        max_flat = flatten_state_index(
            n_states - 1,
            n_orientations - 1,
            n_door_key_states - 1,
            n_states,
            n_orientations,
            n_door_key_states,
        )
        assert max_flat == n_states * n_orientations * n_door_key_states - 1

    def test_integration_realistic_grid_sizes(self):
        n = 5
        n_states = n * n  # 25
        n_orientations = 4
        n_door_key_states = 3

        for state in range(n_states):
            for orientation in range(n_orientations):
                for door_key_state in range(n_door_key_states):
                    flat_idx = flatten_state_index(
                        state,
                        orientation,
                        door_key_state,
                        n_states,
                        n_orientations,
                        n_door_key_states,
                    )
                    state_out, orientation_out, door_key_state_out = (
                        unflatten_state_index(
                            flat_idx, n_states, n_orientations, n_door_key_states
                        )
                    )
                    assert state_out == state
                    assert orientation_out == orientation
                    assert door_key_state_out == door_key_state


class TestValidStaticConfigs:
    def test_all_configs_have_key_before_door(self):
        n = 5
        configs = get_valid_static_configs(n)
        for key_pos, door_pos in configs:
            key_x, _ = key_position(key_pos, n)
            door_x, _ = door_position(door_pos, n)
            assert key_x < door_x, f"key_x={key_x} >= door_x={door_x}"

    def test_fewer_configs_than_full_product(self):
        n = 5
        n_positions = n * n - 2 * n
        configs = get_valid_static_configs(n)
        assert len(configs) < n_positions * n_positions

    def test_configs_nonempty(self):
        for n in [3, 4, 5]:
            configs = get_valid_static_configs(n)
            assert len(configs) > 0


class TestGetNextOrientation:
    def test_turn_left_rotates_counter_clockwise(self):
        assert (
            get_next_orientation(Orientation.RIGHT, ActionType.TURN_LEFT)
            == Orientation.UP
        )
        assert (
            get_next_orientation(Orientation.UP, ActionType.TURN_LEFT)
            == Orientation.LEFT
        )
        assert (
            get_next_orientation(Orientation.LEFT, ActionType.TURN_LEFT)
            == Orientation.DOWN
        )
        assert (
            get_next_orientation(Orientation.DOWN, ActionType.TURN_LEFT)
            == Orientation.RIGHT
        )

    def test_turn_right_rotates_clockwise(self):
        assert (
            get_next_orientation(Orientation.RIGHT, ActionType.TURN_RIGHT)
            == Orientation.DOWN
        )
        assert (
            get_next_orientation(Orientation.DOWN, ActionType.TURN_RIGHT)
            == Orientation.LEFT
        )
        assert (
            get_next_orientation(Orientation.LEFT, ActionType.TURN_RIGHT)
            == Orientation.UP
        )
        assert (
            get_next_orientation(Orientation.UP, ActionType.TURN_RIGHT)
            == Orientation.RIGHT
        )

    def test_other_actions_dont_change_orientation(self):
        for orientation in [
            Orientation.RIGHT,
            Orientation.DOWN,
            Orientation.LEFT,
            Orientation.UP,
        ]:
            assert get_next_orientation(orientation, ActionType.FORWARD) == orientation
            assert get_next_orientation(orientation, ActionType.PICKUP) == orientation
            assert get_next_orientation(orientation, ActionType.TOGGLE) == orientation


class TestGetNextDoorKeyState:
    # door_key_state: 0 = key on ground, 1 = key held, 2 = door open

    def test_pickup_when_facing_key(self):
        # Agent at (1,1), facing RIGHT, key at (2,1) - key is directly in front
        assert (
            get_next_door_key_state(
                1, 1, Orientation.RIGHT, 2, 1, 3, 1, ActionType.PICKUP, 0
            )
            == 1
        )

        # Agent at (1,1), facing DOWN, key at (1,2) - key is directly in front
        assert (
            get_next_door_key_state(
                1, 1, Orientation.DOWN, 1, 2, 3, 1, ActionType.PICKUP, 0
            )
            == 1
        )

        # Agent at (1,1), facing UP, key at (1,0) - key is directly in front
        assert (
            get_next_door_key_state(
                1, 1, Orientation.UP, 1, 0, 3, 1, ActionType.PICKUP, 0
            )
            == 1
        )

        # Agent at (1,1), facing LEFT, key at (0,1) - key is directly in front
        assert (
            get_next_door_key_state(
                1, 1, Orientation.LEFT, 0, 1, 3, 1, ActionType.PICKUP, 0
            )
            == 1
        )

    def test_pickup_when_not_facing_key(self):
        # Agent at (1,1), facing RIGHT, key at (1,2) - key is not in front
        assert (
            get_next_door_key_state(
                1, 1, Orientation.RIGHT, 1, 2, 3, 1, ActionType.PICKUP, 0
            )
            == 0
        )

        # Agent at (1,1), facing RIGHT, key at (3,1) - key is 2 cells away, not adjacent
        assert (
            get_next_door_key_state(
                1, 1, Orientation.RIGHT, 3, 1, 4, 1, ActionType.PICKUP, 0
            )
            == 0
        )

    def test_pickup_when_key_already_held(self):
        # Even if facing key position, state stays at 1
        assert (
            get_next_door_key_state(
                1, 1, Orientation.RIGHT, 2, 1, 3, 1, ActionType.PICKUP, 1
            )
            == 1
        )

    def test_toggle_when_facing_door_with_key(self):
        # Agent at (2,1), facing RIGHT, door at (3,1), has key (state=1)
        assert (
            get_next_door_key_state(
                2, 1, Orientation.RIGHT, 0, 0, 3, 1, ActionType.TOGGLE, 1
            )
            == 2
        )

        # Agent at (2,1), facing DOWN, door at (2,2), has key (state=1)
        assert (
            get_next_door_key_state(
                2, 1, Orientation.DOWN, 0, 0, 2, 2, ActionType.TOGGLE, 1
            )
            == 2
        )

    def test_toggle_when_not_facing_door(self):
        # Agent at (2,1), facing RIGHT, door at (2,2) - not facing door
        assert (
            get_next_door_key_state(
                2, 1, Orientation.RIGHT, 0, 0, 2, 2, ActionType.TOGGLE, 1
            )
            == 1
        )

    def test_toggle_without_key(self):
        # Agent at (2,1), facing RIGHT, door at (3,1), no key (state=0)
        assert (
            get_next_door_key_state(
                2, 1, Orientation.RIGHT, 3, 1, 3, 1, ActionType.TOGGLE, 0
            )
            == 0
        )

    def test_toggle_when_door_already_open(self):
        # Door already open, stays open
        assert (
            get_next_door_key_state(
                2, 1, Orientation.RIGHT, 0, 0, 3, 1, ActionType.TOGGLE, 2
            )
            == 2
        )

    def test_other_actions_dont_change_door_key_state(self):
        for state in [0, 1, 2]:
            assert (
                get_next_door_key_state(
                    1, 1, Orientation.RIGHT, 2, 1, 3, 1, ActionType.TURN_LEFT, state
                )
                == state
            )
            assert (
                get_next_door_key_state(
                    1, 1, Orientation.RIGHT, 2, 1, 3, 1, ActionType.TURN_RIGHT, state
                )
                == state
            )
            assert (
                get_next_door_key_state(
                    1, 1, Orientation.RIGHT, 2, 1, 3, 1, ActionType.FORWARD, state
                )
                == state
            )


class TestCoordinateFunctions:
    def test_state_to_coords_roundtrip(self):
        n = 5
        for s in range(n * n):
            x, y = state_to_coords(s, n)
            assert coords_to_state(x, y, n) == s

    def test_state_to_coords_specific(self):
        n = 5
        assert state_to_coords(0, n) == (0, 0)
        assert state_to_coords(1, n) == (0, 1)
        assert state_to_coords(n, n) == (1, 0)
        assert state_to_coords(n * n - 1, n) == (n - 1, n - 1)

    def test_key_position(self):
        n = 5
        assert key_position(0, n) == (0, 0)
        assert key_position(1, n) == (0, 1)
        assert key_position(n, n) == (1, 0)

    def test_door_position(self):
        n = 5
        # Door positions start in column 1 (0-indexed), not column 0
        assert door_position(0, n) == (1, 0)
        assert door_position(1, n) == (1, 1)
        assert door_position(n, n) == (2, 0)


class TestRelativeCoords:
    def test_get_relative_coords_facing_right(self):
        # Agent at (2,2) facing RIGHT
        # Target at (3,2) should be (0, 1) - directly in front
        assert get_relative_coords(2, 2, Orientation.RIGHT, 3, 2) == (0, 1)
        # Target at (2,3) should be (-1, 0) - to the left
        assert get_relative_coords(2, 2, Orientation.RIGHT, 2, 3) == (-1, 0)
        # Target at (2,1) should be (1, 0) - to the right
        assert get_relative_coords(2, 2, Orientation.RIGHT, 2, 1) == (1, 0)

    def test_get_relative_coords_facing_down(self):
        # Agent at (2,2) facing DOWN
        # Target at (2,3) should be (0, 1) - directly in front
        assert get_relative_coords(2, 2, Orientation.DOWN, 2, 3) == (0, 1)

    def test_get_relative_coords_facing_up(self):
        # Agent at (2,2) facing UP
        # Target at (2,1) should be (0, 1) - directly in front
        assert get_relative_coords(2, 2, Orientation.UP, 2, 1) == (0, 1)

    def test_in_fov(self):
        assert in_fov(0, 0) == True
        assert in_fov(0, 6) == True
        assert in_fov(-3, 0) == True
        assert in_fov(3, 0) == True
        assert in_fov(-4, 0) == False
        assert in_fov(4, 0) == False
        assert in_fov(0, -1) == False
        assert in_fov(0, 7) == False

    def test_relative_to_fov_coords(self):
        # Agent is at (3, 6) in FOV coordinates (0-indexed)
        assert relative_to_fov_coords(0, 0) == (3, 6)  # Agent position
        assert relative_to_fov_coords(0, 1) == (3, 5)  # One step forward
        assert relative_to_fov_coords(-1, 0) == (4, 6)  # One step left
        assert relative_to_fov_coords(1, 0) == (2, 6)  # One step right


class TestGetNextAgentPosition:
    def test_forward_open_space(self):
        n = 5
        # Agent at (2,2), facing RIGHT, no obstacles in the way
        new_state = get_next_agent_position(
            2, 2, Orientation.RIGHT, 4, 2, 0, 0, 2, ActionType.FORWARD, n
        )
        assert state_to_coords(new_state, n) == (3, 2)

    def test_forward_into_wall(self):
        n = 5
        # Agent at (4,2), facing RIGHT, would go out of bounds
        new_state = get_next_agent_position(
            4, 2, Orientation.RIGHT, 3, 2, 0, 0, 2, ActionType.FORWARD, n
        )
        assert state_to_coords(new_state, n) == (4, 2)  # Stays in place

    def test_forward_into_closed_door(self):
        n = 5
        # Agent at (2,2), facing RIGHT, door at (3,2), door not open
        new_state = get_next_agent_position(
            2, 2, Orientation.RIGHT, 3, 2, 0, 0, 0, ActionType.FORWARD, n
        )
        assert state_to_coords(new_state, n) == (2, 2)  # Stays in place

    def test_forward_through_open_door(self):
        n = 5
        # Agent at (2,2), facing RIGHT, door at (3,2), door open (state=2)
        new_state = get_next_agent_position(
            2, 2, Orientation.RIGHT, 3, 2, 0, 0, 2, ActionType.FORWARD, n
        )
        assert state_to_coords(new_state, n) == (3, 2)

    def test_turn_doesnt_move(self):
        n = 5
        for action in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT]:
            new_state = get_next_agent_position(
                2, 2, Orientation.RIGHT, 4, 2, 0, 0, 0, action, n
            )
            assert state_to_coords(new_state, n) == (2, 2)


class TestFOV:
    def test_fov_basic_shape(self):
        n = 5
        fov = get_fov(2, 2, Orientation.RIGHT, 0, 0, 3, 2, 0, n)
        assert fov.shape == (7, 7)

    def test_fov_contains_door_when_visible(self):
        n = 5
        # Agent at (2,2), facing RIGHT, door at (3,2) - should be visible
        fov = get_fov(2, 2, Orientation.RIGHT, 0, 0, 3, 2, 0, n)
        # Door should be at relative (0,1), which is FOV (3, 5)
        assert fov[3, 5] == CellType.DOOR

    def test_fov_contains_key_when_visible(self):
        n = 5
        # Agent at (2,2), facing DOWN, key at (2,3) - should be visible (one step forward)
        fov = get_fov(2, 2, Orientation.DOWN, 2, 3, 3, 2, 0, n)
        # Key should be at relative (0,1), which is FOV (3, 5)
        assert fov[3, 5] == CellType.KEY

    def test_fov_key_at_agent_when_held(self):
        n = 5
        # Agent has key (door_key_state=1), key should appear at agent position
        fov = get_fov(2, 2, Orientation.RIGHT, 0, 0, 3, 2, 1, n)
        # Agent is at FOV (3, 6)
        assert fov[3, 6] == CellType.KEY

    def test_fov_contains_goal(self):
        n = 5
        # Agent at (3,3), facing RIGHT, goal at (4,4) - should be visible
        fov = get_fov(3, 3, Orientation.RIGHT, 0, 0, 2, 2, 2, n)
        # Goal at (4,4): dx=1, dy=1; RIGHT: (-1, 1) → FOV (3-(-1), 6-1) = (4, 5)
        assert fov[4, 5] == CellType.GOAL


class TestTensorGeneration:
    def test_observation_tensor_shape(self):
        from environments.minigrid import (
            generate_observation_tensor,
            N_CELL_TYPES,
            N_ORIENTATIONS,
            N_DOOR_KEY_STATES,
        )

        n = 5
        valid_configs = get_valid_static_configs(n)
        n_location_states = n * n  # 25
        n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES  # 300
        n_static_states = len(valid_configs)

        B = generate_observation_tensor(n, valid_configs)
        assert B.shape == (7, 7, N_CELL_TYPES, n_total_states, n_static_states)

    def test_observation_tensor_is_onehot(self):
        from environments.minigrid import generate_observation_tensor

        n = 4  # Use smaller grid for speed
        valid_configs = get_valid_static_configs(n)
        B = generate_observation_tensor(n, valid_configs)

        # For each (fov_x, fov_y, state, static_state), exactly one cell type should be 1
        for state_idx in range(B.shape[3]):
            for static_idx in range(B.shape[4]):
                for x in range(7):
                    for y in range(7):
                        cell_probs = B[x, y, :, state_idx, static_idx]
                        assert np.sum(cell_probs) == 1.0, f"Not one-hot at ({x},{y},{state_idx},{static_idx})"
                        assert np.max(cell_probs) == 1.0

    def test_orientation_observation_tensor_shape(self):
        from environments.minigrid import (
            generate_orientation_observation_tensor,
            N_ORIENTATIONS,
            N_DOOR_KEY_STATES,
        )

        n = 5
        n_location_states = n * n
        n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES

        B = generate_orientation_observation_tensor(n)
        assert B.shape == (N_ORIENTATIONS, n_total_states)

    def test_orientation_observation_tensor_is_onehot(self):
        from environments.minigrid import generate_orientation_observation_tensor

        n = 5
        B = generate_orientation_observation_tensor(n)

        # Each column should be one-hot
        for state_idx in range(B.shape[1]):
            assert np.sum(B[:, state_idx]) == 1.0
            assert np.max(B[:, state_idx]) == 1.0

    def test_transition_tensor_shape(self):
        from environments.minigrid import (
            generate_transition_tensor,
            N_ORIENTATIONS,
            N_DOOR_KEY_STATES,
            N_ACTIONS,
        )

        n = 5
        valid_configs = get_valid_static_configs(n)
        n_location_states = n * n
        n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
        n_static_states = len(valid_configs)

        T = generate_transition_tensor(n, valid_configs)
        assert T.shape == (n_total_states, n_total_states, n_static_states, N_ACTIONS)

    def test_transition_tensor_is_stochastic(self):
        from environments.minigrid import generate_transition_tensor

        n = 4  # Use smaller grid for speed
        valid_configs = get_valid_static_configs(n)
        T = generate_transition_tensor(n, valid_configs)

        # For each (old_state, static_state, action), probabilities over new_state should sum to 1
        for old_idx in range(T.shape[1]):
            for static_idx in range(T.shape[2]):
                for action in range(T.shape[3]):
                    prob_sum = np.sum(T[:, old_idx, static_idx, action])
                    assert np.isclose(
                        prob_sum, 1.0
                    ), f"Transition probs don't sum to 1: {prob_sum} at ({old_idx},{static_idx},{action})"


class TestCustomFOVSize:
    def test_in_fov_size5(self):
        assert in_fov(0, 0, fov_size=5) == True
        assert in_fov(0, 4, fov_size=5) == True
        assert in_fov(-2, 0, fov_size=5) == True
        assert in_fov(2, 0, fov_size=5) == True
        assert in_fov(-3, 0, fov_size=5) == False
        assert in_fov(3, 0, fov_size=5) == False
        assert in_fov(0, -1, fov_size=5) == False
        assert in_fov(0, 5, fov_size=5) == False

    def test_relative_to_fov_coords_size5(self):
        # Agent is at (2, 4) in 5x5 FOV
        assert relative_to_fov_coords(0, 0, fov_size=5) == (2, 4)
        assert relative_to_fov_coords(0, 1, fov_size=5) == (2, 3)
        assert relative_to_fov_coords(-1, 0, fov_size=5) == (3, 4)
        assert relative_to_fov_coords(1, 0, fov_size=5) == (1, 4)

    def test_get_fov_shape_size5(self):
        n = 5
        fov = get_fov(2, 2, Orientation.RIGHT, 0, 0, 3, 2, 0, n, fov_size=5)
        assert fov.shape == (5, 5)

    def test_get_fov_shape_size3(self):
        n = 5
        fov = get_fov(2, 2, Orientation.RIGHT, 0, 0, 3, 2, 0, n, fov_size=3)
        assert fov.shape == (3, 3)

    def test_get_fov_key_at_agent_size5(self):
        n = 5
        fov = get_fov(2, 2, Orientation.RIGHT, 0, 0, 3, 2, 1, n, fov_size=5)
        # Agent at (half=2, fov_size-1=4)
        assert fov[2, 4] == CellType.KEY

    def test_get_fov_door_visible_size5(self):
        n = 5
        # Agent at (2,2), facing RIGHT, door at (3,2) — relative (0,1) → FOV (2,3)
        fov = get_fov(2, 2, Orientation.RIGHT, 0, 0, 3, 2, 0, n, fov_size=5)
        assert fov[2, 3] == CellType.DOOR

    def test_observation_tensor_shape_size5(self):
        from environments.minigrid import (
            N_ORIENTATIONS,
            N_DOOR_KEY_STATES,
        )

        n = 3
        fov_size = 5
        valid_configs = get_valid_static_configs(n)
        n_location_states = n * n
        n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
        n_static_states = len(valid_configs)

        B = generate_observation_tensor(n, valid_configs, fov_size=fov_size)
        assert B.shape == (fov_size, fov_size, N_CELL_TYPES, n_total_states, n_static_states)


class TestObservationSoftening:
    def setup_method(self):
        self.n = 3
        self.fov_size = 7
        self.valid_configs = get_valid_static_configs(self.n)
        self.B_hard = generate_observation_tensor(self.n, self.valid_configs, fov_size=self.fov_size)

    def test_shape_preserved(self):
        B_soft = soften_observation_tensor(self.B_hard, self.fov_size, alpha=0.1)
        assert B_soft.shape == self.B_hard.shape

    def test_sums_to_one(self):
        B_soft = soften_observation_tensor(self.B_hard, self.fov_size, alpha=0.1)
        # Sum over cell-type axis (axis=2) should be 1 for every (fov_x, fov_y, state, static)
        sums = np.sum(B_soft.astype(np.float64), axis=2)
        assert np.allclose(sums, 1.0, atol=1e-3), f"Max deviation: {np.max(np.abs(sums - 1.0))}"

    def test_reference_cell_unchanged(self):
        """Cell directly in front of agent (d=0) should be unchanged."""
        alpha = 0.15
        B_soft = soften_observation_tensor(self.B_hard, self.fov_size, alpha=alpha)
        half = self.fov_size // 2
        ref_x, ref_y = half, self.fov_size - 2
        np.testing.assert_array_equal(
            B_soft[ref_x, ref_y, :, :, :],
            self.B_hard[ref_x, ref_y, :, :, :],
        )

    def test_alpha_zero_recovers_hard(self):
        B_soft = soften_observation_tensor(self.B_hard, self.fov_size, alpha=0.0)
        np.testing.assert_array_equal(B_soft, self.B_hard)

    def test_large_alpha_approaches_uniform(self):
        """Visible cells far from reference should approach uniform with large alpha."""
        alpha = 0.5  # At d>=2, precision=0 → fully uniform
        B_soft = soften_observation_tensor(self.B_hard, self.fov_size, alpha=alpha)
        uniform = 1.0 / N_CELL_TYPES

        # Check a far-away visible cell (corner at (0,0), d = 3+5 = 8 from ref (3,5))
        # But only if it's not UNSEEN in the hard tensor
        for s in range(B_soft.shape[3]):
            for th in range(B_soft.shape[4]):
                if self.B_hard[0, 0, CellType.UNSEEN, s, th] == 1.0:
                    continue  # skip occluded
                cell_probs = B_soft[0, 0, :, s, th].astype(np.float64)
                assert np.allclose(cell_probs, uniform, atol=1e-3), \
                    f"Far cell not uniform: {cell_probs}"
                break  # one visible example is enough
            else:
                continue
            break

    def test_unseen_softened(self):
        """UNSEEN entries are softened like any other cell type."""
        alpha = 0.2
        B_soft = soften_observation_tensor(self.B_hard, self.fov_size, alpha=alpha)

        unseen_mask = self.B_hard[:, :, CellType.UNSEEN, :, :] == 1.0
        # Where hard tensor had UNSEEN=1, soft tensor should NOT be identical
        for s in range(B_soft.shape[3]):
            for th in range(B_soft.shape[4]):
                for i in range(self.fov_size):
                    for j in range(self.fov_size):
                        if unseen_mask[i, j, s, th]:
                            soft_probs = B_soft[i, j, :, s, th].astype(np.float64)
                            assert np.allclose(soft_probs.sum(), 1.0, atol=1e-2), \
                                f"UNSEEN soft probs don't sum to 1: {soft_probs.sum()}"
                            assert soft_probs[CellType.UNSEEN] > soft_probs.max() - 1e-3, \
                                "UNSEEN should still be the most likely cell type"
                            assert not np.array_equal(
                                B_soft[i, j, :, s, th],
                                self.B_hard[i, j, :, s, th],
                            ), "UNSEEN entry should be softened, not preserved exactly"
                            return  # one example is enough
