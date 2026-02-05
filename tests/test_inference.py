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
