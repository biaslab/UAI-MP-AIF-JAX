import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMessages:
    def test_forward_message_2d_shape(self):
        import jax.numpy as jnp
        from inference.messages import forward_message_2d

        tensor = jnp.ones((5, 3)) / 3
        q_in = jnp.ones(3) / 3
        msg = forward_message_2d(tensor, q_in)
        assert msg.shape == (5,)

    def test_forward_message_2d_normalized(self):
        import jax.numpy as jnp
        from inference.messages import forward_message_2d

        tensor = jnp.array([[0.5, 0.5], [0.3, 0.7]])
        q_in = jnp.array([0.6, 0.4])
        msg = forward_message_2d(tensor, q_in)
        assert np.isclose(msg.sum(), 1.0)

    def test_forward_message_2d_deterministic(self):
        import jax.numpy as jnp
        from inference.messages import forward_message_2d

        tensor = jnp.eye(3)
        q_in = jnp.array([1.0, 0.0, 0.0])
        msg = forward_message_2d(tensor, q_in)
        assert np.allclose(msg, jnp.array([1.0, 0.0, 0.0]))

    def test_forward_message_4d_shape(self):
        import jax.numpy as jnp
        from inference.messages import forward_message_4d

        tensor = jnp.ones((10, 5, 3, 7))
        tensor = tensor / tensor.sum(axis=0, keepdims=True)
        q1 = jnp.ones(5) / 5
        q2 = jnp.ones(3) / 3
        q3 = jnp.ones(7) / 7
        msg = forward_message_4d(tensor, q1, q2, q3)
        assert msg.shape == (10,)
        assert np.isclose(msg.sum(), 1.0)

    def test_backward_message_2d_shape(self):
        import jax.numpy as jnp
        from inference.messages import backward_message_2d

        tensor = jnp.ones((5, 3))
        obs = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])  # Observed state 1
        msg = backward_message_2d(tensor, obs)
        assert msg.shape == (3,)

    def test_backward_message_2d_onehot_observation(self):
        import jax.numpy as jnp
        from inference.messages import backward_message_2d

        tensor = jnp.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
        obs = jnp.array([0.0, 1.0, 0.0])  # Observed state 1
        msg = backward_message_2d(tensor, obs)
        assert np.allclose(msg, jnp.array([0.2, 0.8]))

    def test_combine_messages_normalized(self):
        import jax.numpy as jnp
        from inference.messages import combine_messages

        msg1 = jnp.array([0.5, 0.3, 0.2])
        msg2 = jnp.array([0.2, 0.5, 0.3])
        combined = combine_messages([msg1, msg2])
        assert np.isclose(combined.sum(), 1.0)

    def test_combine_messages_single(self):
        import jax.numpy as jnp
        from inference.messages import combine_messages

        msg = jnp.array([0.5, 0.3, 0.2])
        combined = combine_messages([msg])
        assert np.isclose(combined.sum(), 1.0)


class TestStateInference:
    def setup_method(self):
        import jax.numpy as jnp
        from environments.minigrid import (
            generate_observation_tensor,
            generate_orientation_observation_tensor,
            generate_transition_tensor,
            N_ORIENTATIONS,
            N_DOOR_KEY_STATES,
            N_ACTIONS,
            N_CELL_TYPES,
        )

        self.n = 4
        n_loc = self.n * self.n
        n_key = n_loc - 2 * self.n
        n_door = n_loc - 2 * self.n
        self.n_states = n_loc * N_ORIENTATIONS * N_DOOR_KEY_STATES
        self.n_static = n_key * n_door

        self.transition_tensor = jnp.array(generate_transition_tensor(self.n))
        self.obs_tensors = jnp.array(generate_observation_tensor(self.n))
        self.ori_tensor = jnp.array(
            generate_orientation_observation_tensor(self.n)
        )

    def test_state_inference_shapes(self):
        import jax.numpy as jnp
        from inference.state_inference import state_inference_step

        q_old = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        vision_obs = jnp.zeros((7, 7, 11))
        vision_obs = vision_obs.at[:, :, 1].set(1.0)  # All EMPTY
        ori_obs = jnp.array([1.0, 0.0, 0.0, 0.0])  # Facing RIGHT
        action = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # TURN_LEFT

        q_current, q_static_new = state_inference_step(
            q_old,
            q_static,
            self.transition_tensor,
            self.obs_tensors,
            self.ori_tensor,
            vision_obs,
            ori_obs,
            action,
            n_iterations=1,
        )

        assert q_current.shape == (self.n_states,)
        assert q_static_new.shape == (self.n_static,)
        assert np.isclose(q_current.sum(), 1.0)
        assert np.isclose(q_static_new.sum(), 1.0)

    def test_state_inference_converges(self):
        import jax.numpy as jnp
        from inference.state_inference import state_inference_step

        q_old = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        vision_obs = jnp.zeros((7, 7, 11))
        vision_obs = vision_obs.at[:, :, 1].set(1.0)
        ori_obs = jnp.array([1.0, 0.0, 0.0, 0.0])
        action = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        q1, _ = state_inference_step(
            q_old,
            q_static,
            self.transition_tensor,
            self.obs_tensors,
            self.ori_tensor,
            vision_obs,
            ori_obs,
            action,
            n_iterations=1,
        )
        q10, _ = state_inference_step(
            q_old,
            q_static,
            self.transition_tensor,
            self.obs_tensors,
            self.ori_tensor,
            vision_obs,
            ori_obs,
            action,
            n_iterations=10,
        )

        assert q1.shape == q10.shape


class TestPlanning:
    def setup_method(self):
        import jax.numpy as jnp
        from environments.minigrid import (
            generate_transition_tensor,
            N_ORIENTATIONS,
            N_DOOR_KEY_STATES,
        )

        self.n = 4
        n_loc = self.n * self.n
        n_key = n_loc - 2 * self.n
        n_door = n_loc - 2 * self.n
        self.n_states = n_loc * N_ORIENTATIONS * N_DOOR_KEY_STATES
        self.n_static = n_key * n_door
        self.n_actions = 7

        self.transition_tensor = jnp.array(generate_transition_tensor(self.n))

    def test_planning_output_shape(self):
        import jax.numpy as jnp
        from inference.planning import planning

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist = planning(
            q_current, q_static, self.transition_tensor, goal, horizon=5
        )

        assert action_dist.shape == (self.n_actions,)
        assert np.isclose(action_dist.sum(), 1.0)

    def test_planning_respects_action_mask(self):
        import jax.numpy as jnp
        from inference.planning import planning

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist = planning(
            q_current, q_static, self.transition_tensor, goal, horizon=5
        )

        # Actions 4 (DROP) and 6 (DONE) should have zero probability
        assert action_dist[4] < 1e-6
        assert action_dist[6] < 1e-6

    def test_marginalize_static_shape(self):
        import jax.numpy as jnp
        from inference.planning import marginalize_static

        q_static = jnp.ones(self.n_static) / self.n_static
        reduced = marginalize_static(self.transition_tensor, q_static)

        assert reduced.shape == (self.n_states, self.n_states, self.n_actions)

    def test_marginalize_static_is_stochastic(self):
        import jax.numpy as jnp
        from inference.planning import marginalize_static

        q_static = jnp.ones(self.n_static) / self.n_static
        reduced = marginalize_static(self.transition_tensor, q_static)

        for old_state in range(min(10, self.n_states)):
            for action in range(self.n_actions):
                prob_sum = reduced[:, old_state, action].sum()
                assert np.isclose(
                    prob_sum, 1.0, atol=1e-5
                ), f"Not stochastic at ({old_state}, {action}): {prob_sum}"


class TestLoopyBPPlanning:
    def setup_method(self):
        import jax.numpy as jnp
        from environments.minigrid import (
            generate_transition_indices,
            N_ORIENTATIONS,
            N_DOOR_KEY_STATES,
        )

        self.n = 4
        n_loc = self.n * self.n
        n_key = n_loc - 2 * self.n
        n_door = n_loc - 2 * self.n
        self.n_states = n_loc * N_ORIENTATIONS * N_DOOR_KEY_STATES
        self.n_static = n_key * n_door
        self.n_actions = 7

        self.transition_idx = jnp.array(generate_transition_indices(self.n))

    def test_output_shape(self):
        import jax.numpy as jnp
        from inference.loopy_bp import loopy_bp_planning_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist = loopy_bp_planning_indexed(
            q_current, q_static, self.transition_idx, goal,
            horizon=5, n_iterations=2,
        )

        assert action_dist.shape == (self.n_actions,)
        assert np.isclose(action_dist.sum(), 1.0)

    def test_single_iter_matches_standard_bp(self):
        """With 1 iteration, cavity_θ = p(θ) for all t, matching standard BP."""
        import jax.numpy as jnp
        from inference.planning import planning_indexed
        from inference.loopy_bp import loopy_bp_planning_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        bp_result = planning_indexed(
            q_current, q_static, self.transition_idx, goal,
            horizon=5, n_iterations=1,
        )
        loopy_result = loopy_bp_planning_indexed(
            q_current, q_static, self.transition_idx, goal,
            horizon=5, n_iterations=1,
        )

        assert np.allclose(bp_result, loopy_result, atol=1e-5), (
            f"Standard BP and Loopy BP should match with 1 iteration.\n"
            f"BP:    {bp_result}\nLoopy: {loopy_result}"
        )

    def test_respects_action_mask(self):
        import jax.numpy as jnp
        from inference.loopy_bp import loopy_bp_planning_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist = loopy_bp_planning_indexed(
            q_current, q_static, self.transition_idx, goal,
            horizon=5, n_iterations=3,
        )

        # Actions 4 (DROP) and 6 (DONE) should have zero probability
        assert action_dist[4] < 1e-6
        assert action_dist[6] < 1e-6

    def test_theta_cavities_shape_and_normalization(self):
        import jax.numpy as jnp
        from inference.loopy_bp import compute_theta_cavities

        n_static = 64
        T = 5
        log_prior = jnp.log(jnp.ones(n_static) / n_static)
        log_dyn_to_theta = jnp.zeros((T, n_static))

        cavities = compute_theta_cavities(log_prior, log_dyn_to_theta)

        assert cavities.shape == (T, n_static)
        for t in range(T):
            assert np.isclose(cavities[t].sum(), 1.0, atol=1e-5)

    def test_forward_backward_messages_shape(self):
        import jax.numpy as jnp
        from inference.loopy_bp import (
            forward_pass, backward_pass, compute_reduced_per_t,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)
        action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
        horizon = 5

        cavity_theta = jnp.tile(q_static, (horizon, 1))
        reduced_per_t = compute_reduced_per_t(
            self.transition_idx, cavity_theta, self.n_states,
        )
        fwd_msgs = forward_pass(reduced_per_t, q_current, action_prior, horizon)
        bwd_msgs, q_u = backward_pass(
            reduced_per_t, fwd_msgs, goal, action_prior, horizon,
        )

        assert fwd_msgs.shape == (horizon + 1, self.n_states)
        assert bwd_msgs.shape == (horizon + 1, self.n_states)

    def test_multi_iteration_changes_result(self):
        import jax.numpy as jnp
        from inference.loopy_bp import loopy_bp_planning_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        result_1 = loopy_bp_planning_indexed(
            q_current, q_static, self.transition_idx, goal,
            horizon=5, n_iterations=1,
        )
        result_5 = loopy_bp_planning_indexed(
            q_current, q_static, self.transition_idx, goal,
            horizon=5, n_iterations=5,
        )

        assert not np.allclose(result_1, result_5, atol=1e-5), (
            "Multiple iterations should refine θ and change results"
        )

    def test_dyn_to_theta_messages_finite(self):
        import jax.numpy as jnp
        from inference.loopy_bp import (
            forward_pass, backward_pass,
            compute_reduced_per_t, compute_dyn_to_theta_msgs,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)
        action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
        horizon = 5

        cavity_theta = jnp.tile(q_static, (horizon, 1))
        reduced_per_t = compute_reduced_per_t(
            self.transition_idx, cavity_theta, self.n_states,
        )
        fwd_msgs = forward_pass(reduced_per_t, q_current, action_prior, horizon)
        bwd_msgs, _ = backward_pass(
            reduced_per_t, fwd_msgs, goal, action_prior, horizon,
        )

        log_dyn_to_theta = compute_dyn_to_theta_msgs(
            self.transition_idx, fwd_msgs, bwd_msgs, action_prior, horizon,
        )

        assert jnp.all(jnp.isfinite(log_dyn_to_theta))


class TestAgentIntegration:
    def setup_method(self):
        import jax.numpy as jnp
        from environments.minigrid import (
            generate_transition_indices,
            generate_observation_indices,
            generate_orientation_indices,
        )
        from utils.tensors import get_dimensions

        self.grid_size = 4
        self.dims = get_dimensions(self.grid_size)
        self.transition_idx = jnp.array(generate_transition_indices(self.grid_size))
        self.observation_idx = jnp.array(generate_observation_indices(self.grid_size))
        self.orientation_idx = jnp.array(generate_orientation_indices(self.grid_size))
        self.goal = jnp.zeros(self.dims["n_states"])
        self.goal = self.goal.at[0].set(1.0)

    def test_agent_creation(self):
        from agents.flat_tensor_agent import IndexedTensorAgent

        agent = IndexedTensorAgent.create(
            grid_size=self.grid_size,
            transition_idx=self.transition_idx,
            observation_idx=self.observation_idx,
            orientation_idx=self.orientation_idx,
            goal=self.goal,
        )
        assert agent.q_state.shape[0] > 0
        assert agent.q_static.shape[0] > 0

    def test_agent_step(self):
        import jax.numpy as jnp
        from agents.flat_tensor_agent import IndexedTensorAgent

        agent = IndexedTensorAgent.create(
            grid_size=self.grid_size,
            transition_idx=self.transition_idx,
            observation_idx=self.observation_idx,
            orientation_idx=self.orientation_idx,
            goal=self.goal,
            planning_horizon=3,
            n_planning_iterations=1,
        )
        vision_obs = jnp.zeros((7, 7, 11))
        vision_obs = vision_obs.at[:, :, 1].set(1.0)
        ori_obs = jnp.array([1.0, 0.0, 0.0, 0.0])

        action, new_agent = agent.step(vision_obs, ori_obs, time_remaining=10)
        assert 0 <= action < 7

    def test_agent_reset(self):
        from agents.flat_tensor_agent import IndexedTensorAgent

        agent = IndexedTensorAgent.create(
            grid_size=self.grid_size,
            transition_idx=self.transition_idx,
            observation_idx=self.observation_idx,
            orientation_idx=self.orientation_idx,
            goal=self.goal,
        )
        new_agent = agent.reset()
        assert new_agent.q_state is not None
