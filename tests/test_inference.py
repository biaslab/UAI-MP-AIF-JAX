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


class TestAIFPlanning:
    def setup_method(self):
        import jax.numpy as jnp
        from environments.minigrid import (
            generate_transition_indices,
            generate_observation_indices,
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
        self.observation_idx = jnp.array(generate_observation_indices(self.n))

    def test_aif_output_shape(self):
        import jax.numpy as jnp
        from inference.aif_planning import aif_planning_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist = aif_planning_indexed(
            q_current, q_static, self.transition_idx,
            self.observation_idx, goal, horizon=5, n_iterations=2
        )

        assert action_dist.shape == (self.n_actions,)
        assert np.isclose(action_dist.sum(), 1.0)

    def test_single_iter_matches_standard_bp(self):
        """AIF with 1 iteration and uniform channels should match standard BP exactly."""
        import jax.numpy as jnp
        from inference.planning import planning_indexed
        from inference.aif_planning import aif_planning_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        bp_result = planning_indexed(
            q_current, q_static, self.transition_idx, goal, horizon=5, n_iterations=1
        )
        aif_result = aif_planning_indexed(
            q_current, q_static, self.transition_idx,
            self.observation_idx, goal, horizon=5, n_iterations=1
        )

        assert np.allclose(bp_result, aif_result, atol=1e-5), (
            f"BP and AIF should match with 1 iteration.\n"
            f"BP:  {bp_result}\nAIF: {aif_result}"
        )

    def test_aif_respects_action_mask(self):
        import jax.numpy as jnp
        from inference.aif_planning import aif_planning_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist = aif_planning_indexed(
            q_current, q_static, self.transition_idx,
            self.observation_idx, goal, horizon=5, n_iterations=3
        )

        # Actions 4 (DROP) and 6 (DONE) should have zero probability
        assert action_dist[4] < 1e-6
        assert action_dist[6] < 1e-6

    def test_channel_is_normalized(self):
        import jax.numpy as jnp
        from inference.aif_planning import (
            channel_update_dynamics,
            compute_modified_kernel,
            compute_all_obs_msgs_to_x,
            aif_backward_pass_with_messages,
            aif_forward_pass,
            N_FOV,
            N_CELL_TYPES,
        )
        from inference.planning import marginalize_static_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        reduced_tensor = marginalize_static_indexed(
            self.transition_idx, q_static, self.n_states
        )
        action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
        horizon = 5

        q_state = jnp.concatenate([
            q_current[None, :],
            jnp.ones((horizon, self.n_states)) / self.n_states,
        ], axis=0)
        q_u = jnp.tile(action_prior, (horizon, 1))

        r_x = jnp.ones((horizon, self.n_states, self.n_states, self.n_actions)) / self.n_states
        r_y = jnp.ones((horizon + 1, N_FOV, N_CELL_TYPES, self.n_states, self.n_static)) / N_CELL_TYPES
        obs_idx_flat = self.observation_idx.reshape(N_FOV, self.n_states, self.n_static)

        reduced_per_t = jnp.tile(reduced_tensor[None, :, :, :], (horizon, 1, 1, 1))
        K_mod = compute_modified_kernel(reduced_per_t, r_x)

        q_theta_per_t = jnp.tile(q_static[None, :], (horizon + 1, 1))
        log_obs_msgs_to_x = compute_all_obs_msgs_to_x(obs_idx_flat, r_y, q_theta_per_t)

        q_state = aif_forward_pass(K_mod, q_state, action_prior, log_obs_msgs_to_x, horizon)
        q_u, backward_msgs = aif_backward_pass_with_messages(
            K_mod, q_state, goal, action_prior, log_obs_msgs_to_x, horizon
        )

        r_x = channel_update_dynamics(
            K_mod, q_state, q_u, backward_msgs, log_obs_msgs_to_x, horizon
        )

        # r_x[t] should sum to 1 over axis 0 (x_t) for each (x_{t-1}, u_t) at each t
        for t in range(horizon):
            sums = r_x[t].sum(axis=0)
            assert np.allclose(sums, 1.0, atol=1e-4), (
                f"Channel at t={t} not normalized. Min sum: {sums.min()}, Max sum: {sums.max()}"
            )

    def test_backward_pass_with_messages_shape(self):
        import jax.numpy as jnp
        from inference.aif_planning import (
            aif_backward_pass_with_messages,
            N_FOV,
            N_CELL_TYPES,
        )
        from inference.planning import marginalize_static_indexed

        q_static = jnp.ones(self.n_static) / self.n_static
        reduced_tensor = marginalize_static_indexed(
            self.transition_idx, q_static, self.n_states
        )
        action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
        horizon = 5

        q_state = jnp.ones((horizon + 1, self.n_states)) / self.n_states
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        # Use uniform obs messages (log-space, per-timestep)
        log_obs_msgs = jnp.zeros((horizon + 1, self.n_states))

        # K_mod needs per-timestep shape: tile reduced_tensor
        K_mod = jnp.tile(reduced_tensor[None, :, :, :], (horizon, 1, 1, 1))

        q_u, backward_msgs = aif_backward_pass_with_messages(
            K_mod, q_state, goal, action_prior, log_obs_msgs, horizon
        )

        assert q_u.shape == (horizon, self.n_actions)
        assert backward_msgs.shape == (horizon + 1, self.n_states)

        # Each backward message should be a valid distribution (or close)
        for t in range(horizon + 1):
            msg_sum = backward_msgs[t].sum()
            assert msg_sum > 0, f"Backward message at t={t} is all zeros"

    def test_theta_updates_across_iterations(self):
        """Verify q_theta changes when n_iterations > 1."""
        import jax.numpy as jnp
        from inference.aif_planning import aif_planning_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        result_1 = aif_planning_indexed(
            q_current, q_static, self.transition_idx,
            self.observation_idx, goal, horizon=5, n_iterations=1
        )
        result_5 = aif_planning_indexed(
            q_current, q_static, self.transition_idx,
            self.observation_idx, goal, horizon=5, n_iterations=5
        )

        # Multi-iteration should differ from single iteration
        # (due to θ updates and channel refinement)
        assert not np.allclose(result_1, result_5, atol=1e-6), (
            "Multi-iteration AIF should differ from single iteration"
        )

    def test_obs_channel_normalized(self):
        """Verify r_y[t] sums to 1 over y (axis 1) for each (fov, x, θ) at each t."""
        import jax.numpy as jnp
        from inference.aif_planning import channel_update_obs, N_FOV, N_CELL_TYPES

        obs_idx_flat = self.observation_idx.reshape(N_FOV, self.n_states, self.n_static)
        horizon = 5

        r_y = channel_update_obs(obs_idx_flat, self.n_states, self.n_static, horizon)

        # r_y[t] should sum to 1 over axis 1 (y) for each (fov, x, θ) at each t
        for t in range(horizon + 1):
            sums = r_y[t].sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-6), (
                f"Obs channel at t={t} not normalized. Min: {sums.min()}, Max: {sums.max()}"
            )

    def test_obs_messages_shape(self):
        """Verify obs message helper output shapes."""
        import jax.numpy as jnp
        from inference.aif_planning import compute_all_obs_msgs_to_x, N_FOV, N_CELL_TYPES

        horizon = 5
        obs_idx_flat = self.observation_idx.reshape(N_FOV, self.n_states, self.n_static)
        r_y = jnp.ones((horizon + 1, N_FOV, N_CELL_TYPES, self.n_states, self.n_static)) / N_CELL_TYPES
        q_theta_per_t = jnp.ones((horizon + 1, self.n_static)) / self.n_static

        log_msgs_to_x = compute_all_obs_msgs_to_x(obs_idx_flat, r_y, q_theta_per_t)

        assert log_msgs_to_x.shape == (horizon + 1, self.n_states)
        # Messages should be finite
        assert jnp.all(jnp.isfinite(log_msgs_to_x))

    def test_theta_message_computation(self):
        """Verify theta message shape and non-degeneracy."""
        import jax.numpy as jnp
        from inference.aif_planning import (
            compute_theta_messages_from_dynamics,
            compute_modified_kernel,
            aif_forward_pass,
            aif_backward_pass_with_messages,
        )
        from inference.planning import marginalize_static_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)
        action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
        horizon = 5

        reduced = marginalize_static_indexed(self.transition_idx, q_static, self.n_states)
        r_x = jnp.ones((horizon, self.n_states, self.n_states, self.n_actions)) / self.n_states
        reduced_per_t = jnp.tile(reduced[None, :, :, :], (horizon, 1, 1, 1))
        K_mod = compute_modified_kernel(reduced_per_t, r_x)

        q_state = jnp.concatenate([
            q_current[None, :],
            jnp.ones((horizon, self.n_states)) / self.n_states,
        ], axis=0)
        q_u = jnp.tile(action_prior, (horizon, 1))
        log_obs_msgs = jnp.zeros((horizon + 1, self.n_states))

        q_state = aif_forward_pass(K_mod, q_state, action_prior, log_obs_msgs, horizon)
        _, backward_msgs = aif_backward_pass_with_messages(
            K_mod, q_state, goal, action_prior, log_obs_msgs, horizon
        )

        log_dyn_msgs_per_t = compute_theta_messages_from_dynamics(
            self.transition_idx, q_state, backward_msgs,
            action_prior, r_x, log_obs_msgs, horizon
        )

        assert log_dyn_msgs_per_t.shape == (horizon, self.n_static)
        assert jnp.all(jnp.isfinite(log_dyn_msgs_per_t))

    def test_dyn_channel_normalized(self):
        """Verify r_x[t] sums to 1 over x_t (axis 0) for each (x_{t-1}, u) at each t."""
        import jax.numpy as jnp
        from inference.aif_planning import (
            channel_update_dynamics,
            compute_modified_kernel,
            aif_forward_pass,
            aif_backward_pass_with_messages,
        )
        from inference.planning import marginalize_static_indexed

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)
        action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
        horizon = 5

        reduced = marginalize_static_indexed(self.transition_idx, q_static, self.n_states)
        r_x = jnp.ones((horizon, self.n_states, self.n_states, self.n_actions)) / self.n_states
        reduced_per_t = jnp.tile(reduced[None, :, :, :], (horizon, 1, 1, 1))
        K_mod = compute_modified_kernel(reduced_per_t, r_x)

        q_state = jnp.concatenate([
            q_current[None, :],
            jnp.ones((horizon, self.n_states)) / self.n_states,
        ], axis=0)
        q_u = jnp.tile(action_prior, (horizon, 1))
        log_obs_msgs = jnp.zeros((horizon + 1, self.n_states))

        q_state = aif_forward_pass(K_mod, q_state, action_prior, log_obs_msgs, horizon)
        q_u, backward_msgs = aif_backward_pass_with_messages(
            K_mod, q_state, goal, action_prior, log_obs_msgs, horizon
        )

        r_x_new = channel_update_dynamics(
            K_mod, q_state, q_u, backward_msgs, log_obs_msgs, horizon
        )

        for t in range(horizon):
            sums = r_x_new[t].sum(axis=0)
            assert np.allclose(sums, 1.0, atol=1e-4), (
                f"Dyn channel at t={t} not normalized. Min: {sums.min()}, Max: {sums.max()}"
            )


class TestAgentIntegration:
    def test_agent_creation(self):
        import jax.numpy as jnp
        from agents.flat_tensor_agent import FlatTensorAgent
        from environments.minigrid import (
            generate_observation_tensor,
            generate_orientation_observation_tensor,
            generate_transition_tensor,
        )

        n = 4
        transition = jnp.array(generate_transition_tensor(n))
        observation = jnp.array(generate_observation_tensor(n))
        orientation = jnp.array(generate_orientation_observation_tensor(n))
        goal = jnp.ones(transition.shape[0]) / transition.shape[0]

        agent = FlatTensorAgent.create(
            grid_size=n,
            transition_tensor=transition,
            observation_tensors=observation,
            orientation_tensor=orientation,
            goal=goal,
            planning_horizon=5,
            n_inference_iterations=3,
            n_planning_iterations=1,
        )

        assert agent.grid_size == n
        assert agent.q_state.shape[0] == transition.shape[0]
        assert np.isclose(agent.q_state.sum(), 1.0)

    def test_agent_step(self):
        import jax.numpy as jnp
        from agents.flat_tensor_agent import FlatTensorAgent
        from environments.minigrid import (
            generate_observation_tensor,
            generate_orientation_observation_tensor,
            generate_transition_tensor,
            N_CELL_TYPES,
            N_ORIENTATIONS,
        )

        n = 4
        transition = jnp.array(generate_transition_tensor(n))
        observation = jnp.array(generate_observation_tensor(n))
        orientation = jnp.array(generate_orientation_observation_tensor(n))
        goal = jnp.zeros(transition.shape[0])
        goal = goal.at[0].set(1.0)

        agent = FlatTensorAgent.create(
            grid_size=n,
            transition_tensor=transition,
            observation_tensors=observation,
            orientation_tensor=orientation,
            goal=goal,
            planning_horizon=3,
            n_inference_iterations=2,
            n_planning_iterations=1,
        )

        vision_obs = jnp.zeros((7, 7, N_CELL_TYPES))
        vision_obs = vision_obs.at[:, :, 1].set(1.0)  # All EMPTY
        ori_obs = jnp.zeros(N_ORIENTATIONS)
        ori_obs = ori_obs.at[0].set(1.0)  # Facing RIGHT

        action, new_agent = agent.step(vision_obs, ori_obs, time_remaining=10)

        assert 0 <= action < 7
        assert action != 4  # Not DROP
        assert action != 6  # Not DONE
        assert new_agent.last_action == action
        assert np.isclose(new_agent.q_state.sum(), 1.0)

    def test_agent_reset(self):
        import jax.numpy as jnp
        from agents.flat_tensor_agent import FlatTensorAgent
        from environments.minigrid import (
            generate_observation_tensor,
            generate_orientation_observation_tensor,
            generate_transition_tensor,
        )

        n = 4
        transition = jnp.array(generate_transition_tensor(n))
        observation = jnp.array(generate_observation_tensor(n))
        orientation = jnp.array(generate_orientation_observation_tensor(n))
        goal = jnp.ones(transition.shape[0]) / transition.shape[0]

        agent = FlatTensorAgent.create(
            grid_size=n,
            transition_tensor=transition,
            observation_tensors=observation,
            orientation_tensor=orientation,
            goal=goal,
        )

        original_q_state = agent.q_state.copy()

        # Modify state
        agent = FlatTensorAgent(
            grid_size=agent.grid_size,
            dims=agent.dims,
            transition_tensor=agent.transition_tensor,
            observation_tensors=agent.observation_tensors,
            orientation_tensor=agent.orientation_tensor,
            q_state=jnp.ones_like(agent.q_state) / agent.q_state.shape[0],
            q_static=agent.q_static,
            goal=agent.goal,
            planning_horizon=agent.planning_horizon,
            n_inference_iterations=agent.n_inference_iterations,
            n_planning_iterations=agent.n_planning_iterations,
            last_action=3,
        )

        reset_agent = agent.reset()
        assert reset_agent.last_action == 0
        assert np.allclose(reset_agent.q_state, original_q_state)
