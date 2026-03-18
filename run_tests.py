#!/usr/bin/env python
"""Run all JAX implementation tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_minigrid_tests():
    from tests.test_minigrid import (
        TestIndexMappingFunctions,
        TestValidStaticConfigs,
        TestGetNextOrientation,
        TestGetNextDoorKeyState,
        TestCoordinateFunctions,
        TestRelativeCoords,
        TestGetNextAgentPosition,
        TestFOV,
        TestTensorGeneration,
        TestCustomFOVSize,
        TestObservationSoftening,
    )

    print("=" * 60)
    print("MINIGRID ENVIRONMENT TESTS")
    print("=" * 60)

    print("Running index mapping tests...")
    t = TestIndexMappingFunctions()
    t.test_flatten_unflatten_state_index_roundtrip()
    t.test_flatten_state_index_unique()
    t.test_flatten_state_index_boundary()
    t.test_integration_realistic_grid_sizes()
    print("  Index mapping: PASSED")

    print("Running valid static configs tests...")
    t = TestValidStaticConfigs()
    t.test_all_configs_have_key_before_door()
    t.test_fewer_configs_than_full_product()
    t.test_configs_nonempty()
    print("  Valid static configs: PASSED")

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
    t.test_get_relative_coords_facing_down()
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
    t.test_fov_contains_goal()
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
    t.test_observation_tensor_shape_size5()
    print("  Custom FOV size: PASSED")

    print("Running observation softening tests...")
    t = TestObservationSoftening()
    t.setup_method()
    t.test_shape_preserved()
    t.test_sums_to_one()
    t.test_reference_cell_unchanged()
    t.test_alpha_zero_recovers_hard()
    t.test_large_alpha_approaches_uniform()
    t.test_unseen_softened()
    print("  Observation softening: PASSED")


def run_inference_tests():
    from tests.test_inference import (
        TestMessages,
        TestStateInference,
        TestLoopyBPPlanning,
        TestRegionExtendedLoopyBP,
        TestNumericalStability,
        TestNuijtenMP,
        TestVBPChannel,
        TestPreciseInfoSeeking,
        TestAgentIntegration,
        TestCustomFOVSizeInference,
        TestPerformanceRefactorEquivalence,
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

    print("Running loopy BP planning tests...")
    t = TestLoopyBPPlanning()
    t.setup_method()
    t.test_output_shape()
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
    print("  Numerical stability: PASSED")

    print("Running Nuijten MP tests...")
    t = TestNuijtenMP()
    t.setup_method()
    t.test_obs_region_beliefs_shape_and_normalization()
    t.test_obs_region_beliefs_matches_reference()
    t.test_efe_action_prior_shape_and_valid()
    t.test_efe_action_prior_matches_reference()
    t.test_obs_efe_to_x_shape_and_normalized()
    t.test_obs_efe_to_x_matches_reference()
    t.test_obs_efe_to_theta_shape_and_finite()
    t.test_obs_efe_to_theta_matches_reference()
    t.test_nuijten_output_shape()
    t.test_nuijten_respects_action_mask()
    t.test_nuijten_multi_iteration_no_nan()
    t.test_nuijten_region_beliefs_shapes()
    print("  Nuijten MP: PASSED")

    print("Running VBP channel tests...")
    t = TestVBPChannel()
    t.setup_method()
    t.test_output_shape()
    t.test_action_dist_normalized()
    t.test_respects_action_mask()
    t.test_action_channel_is_conditional()
    t.test_multi_iteration_changes_result()
    print("  VBP Channel: PASSED")

    print("Running precise info-seeking tests...")
    t = TestPreciseInfoSeeking()
    t.setup_method()
    t.test_output_shape()
    t.test_respects_action_mask()
    t.test_multi_iteration_changes_result()
    t.test_obs_channels_shape()
    t.test_action_channels_shape()
    print("  Precise Info-Seeking: PASSED")

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
    t.test_obs_tensor_shape()
    t.test_state_inference_with_fov5()
    t.test_region_extended_with_fov5()
    t.test_agent_step_with_fov5()
    print("  Custom FOV size inference: PASSED")

    print("Running performance refactor equivalence tests...")
    t = TestPerformanceRefactorEquivalence()
    t.setup_method()
    t.test_region_extended_equivalence()
    t.test_loopy_bp_equivalence()
    t.test_loopy_vbp_equivalence()
    print("  Performance refactor equivalence: PASSED")


def run_groundtruth_tests():
    try:
        import pytest
    except ImportError:
        print()
        print("=" * 60)
        print("GROUNDTRUTH FOV TESTS (SKIPPED — pytest not installed)")
        print("=" * 60)
        return

    from tests.test_minigrid_groundtruth import (
        test_fov_initial_state,
        test_fov_full_episode,
    )

    print()
    print("=" * 60)
    print("GROUNDTRUTH FOV TESTS")
    print("=" * 60)

    print("Running FOV ground truth tests (grid=5, fov=3, seeds 0-4)...")
    for seed in range(5):
        test_fov_initial_state(grid_size=5, fov_size=3, seed=seed)
    print("  Initial state FOV: PASSED")

    print("Running FOV full episode ground truth (grid=5, fov=3, seed=0)...")
    test_fov_full_episode(grid_size=5, fov_size=3, seed=0)
    print("  Full episode FOV: PASSED")


def run_rocksample_tests():
    from tests.test_rocksample import (
        TestRockSampleTensors,
        TestRockSampleEnv,
        TestRockSampleAgents,
        TestRockSampleEpisode,
    )

    print()
    print("=" * 60)
    print("ROCKSAMPLE ENVIRONMENT TESTS")
    print("=" * 60)

    print("Running RockSample tensor tests...")
    t = TestRockSampleTensors()
    t.setup_method()
    t.test_config_shapes()
    t.test_exhaustive_configs()
    t.test_rock_quality_independence()
    t.test_rocks_not_at_start_or_exit()
    t.test_state_indexing_roundtrip()
    t.test_transition_shape()
    t.test_transition_stochastic()
    t.test_exit_absorbing()
    t.test_scan_transitions()
    t.test_scan_nearest_rock()
    t.test_scan_all_scanned_selfloop()
    t.test_sample_at_rock()
    t.test_sample_no_rock()
    t.test_sample_already_collected()
    t.test_theta_independent_transitions()
    t.test_observation_shape()
    t.test_observation_is_stochastic()
    t.test_position_channels_correctness()
    t.test_position_channels_scan_independent()
    t.test_rock_quality_distance_dependent()
    t.test_rock_quality_scanned_deterministic()
    t.test_rock_quality_unscanned_not_deterministic()
    t.test_goal_shape()
    t.test_slip_stochastic()
    print("  RockSample tensors: PASSED")

    print("Running RockSample env tests...")
    t = TestRockSampleEnv()
    t.setup_method()
    t.test_reset()
    t.test_movement_right()
    t.test_wall_collision()
    t.test_scan_action()
    t.test_scan_second_rock()
    t.test_scan_all_scanned_noop()
    t.test_sample_at_rock()
    t.test_exit_termination()
    t.test_ascii_render()
    print("  RockSample env: PASSED")

    print("Running RockSample agent tests...")
    t = TestRockSampleAgents()
    t.setup_method()
    t.test_bp_agent_produces_valid_action()
    t.test_loopy_bp_agent_produces_valid_action()
    t.test_region_extended_agent_produces_valid_action()
    t.test_static_belief_update()
    t.test_all_methods_run()
    print("  RockSample agents: PASSED")

    print("Running RockSample episode test...")
    t = TestRockSampleEpisode()
    t.test_episode_completes()
    print("  RockSample episode: PASSED")


def main():
    run_minigrid_tests()
    run_inference_tests()
    run_groundtruth_tests()
    run_rocksample_tests()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
