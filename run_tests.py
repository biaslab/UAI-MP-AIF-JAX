#!/usr/bin/env python
"""Run all JAX implementation tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_minigrid_tests():
    from tests.test_minigrid import (
        TestIndexMappingFunctions,
        TestGetNextOrientation,
        TestGetNextDoorKeyState,
        TestCoordinateFunctions,
        TestRelativeCoords,
        TestGetNextAgentPosition,
        TestFOV,
        TestTensorGeneration,
        TestCustomFOVSize,
    )

    print("=" * 60)
    print("MINIGRID ENVIRONMENT TESTS")
    print("=" * 60)

    print("Running index mapping tests...")
    t = TestIndexMappingFunctions()
    t.test_flatten_unflatten_state_index_roundtrip()
    t.test_flatten_state_index_unique()
    t.test_flatten_state_index_boundary()
    t.test_flatten_unflatten_position_index_roundtrip()
    t.test_flatten_position_index_unique()
    t.test_flatten_position_index_boundary()
    t.test_integration_realistic_grid_sizes()
    print("  Index mapping: PASSED")

    print("Running orientation tests...")
    t = TestGetNextOrientation()
    t.test_turn_left_rotates_counter_clockwise()
    t.test_turn_right_rotates_clockwise()
    t.test_other_actions_dont_change_orientation()
    print("  Orientation: PASSED")

    print("Running door key state tests...")
    t = TestGetNextDoorKeyState()
    t.test_pickup_when_facing_key()
    t.test_pickup_when_not_facing_key()
    t.test_pickup_when_key_already_held()
    t.test_toggle_when_facing_door_with_key()
    t.test_toggle_when_not_facing_door()
    t.test_toggle_without_key()
    t.test_toggle_when_door_already_open()
    t.test_other_actions_dont_change_door_key_state()
    print("  Door key state: PASSED")

    print("Running coordinate tests...")
    t = TestCoordinateFunctions()
    t.test_state_to_coords_roundtrip()
    t.test_state_to_coords_specific()
    t.test_key_position()
    t.test_door_position()
    print("  Coordinates: PASSED")

    print("Running relative coords tests...")
    t = TestRelativeCoords()
    t.test_get_relative_coords_facing_right()
    t.test_get_relative_coords_facing_up()
    t.test_in_fov()
    t.test_relative_to_fov_coords()
    print("  Relative coords: PASSED")

    print("Running agent position tests...")
    t = TestGetNextAgentPosition()
    t.test_forward_open_space()
    t.test_forward_into_wall()
    t.test_forward_into_closed_door()
    t.test_forward_through_open_door()
    t.test_turn_doesnt_move()
    print("  Agent position: PASSED")

    print("Running FOV tests...")
    t = TestFOV()
    t.test_fov_basic_shape()
    t.test_fov_contains_door_when_visible()
    t.test_fov_contains_key_when_visible()
    t.test_fov_key_at_agent_when_held()
    print("  FOV: PASSED")

    print("Running tensor generation tests (may take a moment)...")
    t = TestTensorGeneration()
    t.test_observation_tensor_shape()
    t.test_observation_tensor_is_onehot()
    t.test_orientation_observation_tensor_shape()
    t.test_orientation_observation_tensor_is_onehot()
    t.test_transition_tensor_shape()
    t.test_transition_tensor_is_stochastic()
    print("  Tensor generation: PASSED")

    print("Running custom FOV size tests...")
    t = TestCustomFOVSize()
    t.test_in_fov_size5()
    t.test_relative_to_fov_coords_size5()
    t.test_get_fov_shape_size5()
    t.test_get_fov_shape_size3()
    t.test_get_fov_key_at_agent_size5()
    t.test_get_fov_door_visible_size5()
    t.test_observation_indices_shape_size5()
    t.test_observation_tensor_shape_size5()
    print("  Custom FOV size: PASSED")


def run_inference_tests():
    from tests.test_inference import (
        TestMessages,
        TestStateInference,
        TestPlanning,
        TestLoopyBPPlanning,
        TestRegionExtendedLoopyBP,
        TestNumericalStability,
        TestReducedRegionExtended,
        TestAgentIntegration,
        TestCustomFOVSizeInference,
    )

    print()
    print("=" * 60)
    print("INFERENCE TESTS")
    print("=" * 60)

    print("Running message tests...")
    t = TestMessages()
    t.test_forward_message_2d_shape()
    t.test_forward_message_2d_normalized()
    t.test_forward_message_2d_deterministic()
    t.test_forward_message_4d_shape()
    t.test_backward_message_2d_shape()
    t.test_backward_message_2d_onehot_observation()
    t.test_combine_messages_normalized()
    t.test_combine_messages_single()
    print("  Messages: PASSED")

    print("Running state inference tests (generating tensors)...")
    t = TestStateInference()
    t.setup_method()
    t.test_state_inference_shapes()
    t.test_state_inference_converges()
    print("  State inference: PASSED")

    print("Running planning tests...")
    t = TestPlanning()
    t.setup_method()
    t.test_planning_output_shape()
    t.test_planning_respects_action_mask()
    t.test_marginalize_static_shape()
    t.test_marginalize_static_is_stochastic()
    print("  Planning: PASSED")

    print("Running loopy BP planning tests...")
    t = TestLoopyBPPlanning()
    t.setup_method()
    t.test_output_shape()
    t.test_single_iter_matches_standard_bp()
    t.test_respects_action_mask()
    t.test_theta_cavities_shape_and_normalization()
    t.test_forward_backward_messages_shape()
    t.test_multi_iteration_changes_result()
    t.test_dyn_to_theta_messages_finite()
    print("  Loopy BP Planning: PASSED")

    print("Running region-extended loopy BP tests...")
    t = TestRegionExtendedLoopyBP()
    t.setup_method()
    t.test_output_shape()
    t.test_respects_action_mask()
    t.test_theta_cavities_extended_shape()
    print("  Region-Extended Loopy BP: PASSED")

    print("Running numerical stability tests...")
    t = TestNumericalStability()
    t.test_safe_log_div_zero_over_zero()
    t.test_dyn_channel_with_deterministic_transitions()
    t.test_dyn_channel_naive_division_creates_bogus_transitions()
    t.test_obs_channel_with_deterministic_observations()
    t.test_naive_subtraction_produces_zero_for_impossible()
    t.test_safe_log_on_float16_tensor()
    t.test_safe_log_on_float32_tensor()
    t.test_region_extended_multi_iteration_no_nan()
    t.test_reduced_region_extended_multi_iteration_no_nan()
    print("  Numerical stability: PASSED")

    print("Running reduced region-extended tests...")
    t = TestReducedRegionExtended()
    t.setup_method()
    t.test_output_shape()
    t.test_respects_action_mask()
    t.test_single_iter_matches_region_extended()
    print("  Reduced Region-Extended: PASSED")

    print("Running agent integration tests...")
    t = TestAgentIntegration()
    t.setup_method()
    t.test_agent_creation()
    t.test_agent_step()
    t.test_agent_reset()
    print("  Agent integration: PASSED")

    print("Running custom FOV size inference tests...")
    t = TestCustomFOVSizeInference()
    t.setup_method()
    t.test_obs_idx_shape()
    t.test_state_inference_with_fov5()
    t.test_region_extended_with_fov5()
    t.test_reduced_region_extended_with_fov5()
    t.test_agent_step_with_fov5()
    print("  Custom FOV size inference: PASSED")


def run_groundtruth_tests():
    from tests.test_minigrid_groundtruth import TestFOVSizeAgainstMiniGrid

    print()
    print("=" * 60)
    print("GROUNDTRUTH FOV SIZE TESTS")
    print("=" * 60)

    print("Running FOV size=5 ground truth tests...")
    t = TestFOVSizeAgainstMiniGrid()
    t.test_fov_size5_matches_minigrid()
    print("  FOV size=5 ground truth: PASSED")


def main():
    run_minigrid_tests()
    run_inference_tests()
    run_groundtruth_tests()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
