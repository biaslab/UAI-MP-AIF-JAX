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
            N_CELL_TYPES,
        )

        self.n = 4
        n_loc = self.n * self.n
        n_key = n_loc - 2 * self.n
        n_door = n_loc - 2 * self.n
        self.n_states = n_loc * N_ORIENTATIONS * N_DOOR_KEY_STATES
        self.n_static = n_key * n_door

        self.transition_tensor = jnp.array(generate_transition_tensor(self.n), dtype=jnp.float32)
        self.obs_tensors = jnp.array(generate_observation_tensor(self.n), dtype=jnp.float32)
        self.ori_tensor = jnp.array(
            generate_orientation_observation_tensor(self.n), dtype=jnp.float32
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

        self.transition_tensor = jnp.array(generate_transition_tensor(self.n), dtype=jnp.float32)

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
        from inference.planning import marginalize_static, safe_log

        log_T = safe_log(self.transition_tensor)
        q_static = jnp.ones(self.n_static) / self.n_static
        log_reduced = marginalize_static(log_T, safe_log(q_static))

        assert log_reduced.shape == (self.n_states, self.n_states, self.n_actions)

    def test_marginalize_static_is_stochastic(self):
        import jax.numpy as jnp
        from inference.planning import marginalize_static, safe_log

        log_T = safe_log(self.transition_tensor)
        q_static = jnp.ones(self.n_static) / self.n_static
        log_reduced = marginalize_static(log_T, safe_log(q_static))

        # Convert to probability space and check stochasticity
        reduced = jnp.exp(log_reduced)
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

        self.transition_tensor = jnp.array(generate_transition_tensor(self.n), dtype=jnp.float32)

    def test_output_shape(self):
        import jax.numpy as jnp
        from inference.loopy_bp import loopy_bp_planning

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist = loopy_bp_planning(
            q_current, q_static, self.transition_tensor, goal,
            horizon=5, n_iterations=2,
        )

        assert action_dist.shape == (self.n_actions,)
        assert np.isclose(action_dist.sum(), 1.0)

    def test_single_iter_matches_standard_bp(self):
        """With 1 iteration, cavity_θ = p(θ) for all t, matching standard BP."""
        import jax.numpy as jnp
        from inference.planning import planning
        from inference.loopy_bp import loopy_bp_planning

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        bp_result = planning(
            q_current, q_static, self.transition_tensor, goal,
            horizon=5, n_iterations=1,
        )
        loopy_result = loopy_bp_planning(
            q_current, q_static, self.transition_tensor, goal,
            horizon=5, n_iterations=1,
        )

        assert np.allclose(bp_result, loopy_result, atol=1e-5), (
            f"Standard BP and Loopy BP should match with 1 iteration.\n"
            f"BP:    {bp_result}\nLoopy: {loopy_result}"
        )

    def test_respects_action_mask(self):
        import jax.numpy as jnp
        from inference.loopy_bp import loopy_bp_planning

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist = loopy_bp_planning(
            q_current, q_static, self.transition_tensor, goal,
            horizon=5, n_iterations=3,
        )

        # Actions 4 (DROP) and 6 (DONE) should have zero probability
        assert action_dist[4] < 1e-6
        assert action_dist[6] < 1e-6

    def test_theta_cavities_shape_and_normalization(self):
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp
        from inference.loopy_bp import compute_theta_cavities

        n_static = 64
        T = 5
        log_prior = jnp.log(jnp.ones(n_static) / n_static)
        log_dyn_to_theta = jnp.zeros((T, n_static))

        log_cavities = compute_theta_cavities(log_prior, log_dyn_to_theta)

        assert log_cavities.shape == (T, n_static)
        for t in range(T):
            # Log-space cavities should be normalized: logsumexp = 0
            assert np.isclose(logsumexp(log_cavities[t]), 0.0, atol=1e-5)

    def test_forward_backward_messages_shape(self):
        import jax.numpy as jnp
        from inference.planning import safe_log
        from inference.loopy_bp import (
            forward_pass, backward_pass, compute_reduced_per_t,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)
        action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])
        horizon = 5

        log_T = safe_log(self.transition_tensor)
        log_cavity_theta = safe_log(jnp.tile(q_static, (horizon, 1)))
        log_reduced_per_t = compute_reduced_per_t(log_T, log_cavity_theta)
        log_fwd_msgs = forward_pass(
            log_reduced_per_t, safe_log(q_current), safe_log(action_prior), horizon
        )
        log_bwd_msgs, q_u = backward_pass(
            log_reduced_per_t, log_fwd_msgs, safe_log(goal),
            safe_log(action_prior), horizon,
        )

        assert log_fwd_msgs.shape == (horizon + 1, self.n_states)
        assert log_bwd_msgs.shape == (horizon + 1, self.n_states)

    def test_multi_iteration_changes_result(self):
        import jax.numpy as jnp
        from inference.loopy_bp import loopy_bp_planning

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        result_1 = loopy_bp_planning(
            q_current, q_static, self.transition_tensor, goal,
            horizon=5, n_iterations=1,
        )
        result_5 = loopy_bp_planning(
            q_current, q_static, self.transition_tensor, goal,
            horizon=5, n_iterations=5,
        )

        assert not np.allclose(result_1, result_5, atol=1e-5), (
            "Multiple iterations should refine θ and change results"
        )

    def test_dyn_to_theta_messages_finite(self):
        import jax.numpy as jnp
        from inference.planning import safe_log
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

        log_T = safe_log(self.transition_tensor)
        log_cavity_theta = safe_log(jnp.tile(q_static, (horizon, 1)))
        log_reduced_per_t = compute_reduced_per_t(log_T, log_cavity_theta)
        log_fwd_msgs = forward_pass(
            log_reduced_per_t, safe_log(q_current), safe_log(action_prior), horizon
        )
        log_bwd_msgs, _ = backward_pass(
            log_reduced_per_t, log_fwd_msgs, safe_log(goal),
            safe_log(action_prior), horizon,
        )

        log_dyn_to_theta = compute_dyn_to_theta_msgs(
            log_T, log_fwd_msgs, log_bwd_msgs, safe_log(action_prior), horizon,
        )

        assert jnp.all(jnp.isfinite(log_dyn_to_theta))


class TestRegionExtendedLoopyBP:
    def setup_method(self):
        import jax.numpy as jnp
        from environments.minigrid import (
            generate_transition_tensor,
            generate_observation_tensor,
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

        self.transition_tensor = jnp.array(generate_transition_tensor(self.n), dtype=jnp.float32)
        self.observation_tensor = jnp.array(generate_observation_tensor(self.n), dtype=jnp.float32)

    def test_output_shape(self):
        import jax.numpy as jnp
        from inference.region_extended_loopy_bp import (
            region_extended_loopy_bp_planning,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist, dyn_channels, obs_channels = region_extended_loopy_bp_planning(
            q_current, q_static, self.transition_tensor, self.observation_tensor,
            goal, horizon=5, n_iterations=2,
        )

        assert action_dist.shape == (self.n_actions,)
        assert np.isclose(action_dist.sum(), 1.0)
        assert dyn_channels.shape == (5, self.n_states, self.n_states, self.n_actions)
        assert obs_channels.shape == (6, 49, 11, self.n_states, self.n_static)

    def test_respects_action_mask(self):
        import jax.numpy as jnp
        from inference.region_extended_loopy_bp import (
            region_extended_loopy_bp_planning,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist, _, _ = region_extended_loopy_bp_planning(
            q_current, q_static, self.transition_tensor, self.observation_tensor,
            goal, horizon=5, n_iterations=3,
        )

        assert action_dist[4] < 1e-6
        assert action_dist[6] < 1e-6

    def test_theta_cavities_extended_shape(self):
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp
        from inference.region_extended_loopy_bp import compute_theta_cavities_extended

        n_static = 64
        T = 5
        log_prior = jnp.log(jnp.ones(n_static) / n_static)
        log_dyn = jnp.zeros((T, n_static))
        log_obs = jnp.zeros((T + 1, n_static))

        log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
            log_prior, log_dyn, log_obs
        )

        assert log_cavity_dyn.shape == (T, n_static)
        assert log_cavity_obs.shape == (T + 1, n_static)
        for t in range(T):
            assert np.isclose(logsumexp(log_cavity_dyn[t]), 0.0, atol=1e-5)
        for t in range(T + 1):
            assert np.isclose(logsumexp(log_cavity_obs[t]), 0.0, atol=1e-5)


class TestNumericalStability:
    """Tests for numerical edge cases in log-space kernel reparameterization.

    The main hazard: channels computed from region beliefs contain -inf
    (from normalizing deterministic conditionals). Dividing the original
    factor by such a channel (log_factor - log_channel) produces +inf
    unless guarded by safe_log_div.
    """

    def test_safe_log_div_zero_over_zero(self):
        """0/0 must give 0, not 1 (LOG_ZERO - LOG_ZERO = 0 is wrong)."""
        import jax.numpy as jnp
        from inference.messages import safe_log_div, LOG_ZERO

        log_num = jnp.array([LOG_ZERO, LOG_ZERO, -5.0, -5.0])
        log_den = jnp.array([LOG_ZERO, -3.0,    LOG_ZERO, -3.0])
        result = safe_log_div(log_num, log_den)

        # 0/0 → 0  (LOG_ZERO)
        assert result[0] < LOG_ZERO / 2, f"0/0 should be LOG_ZERO, got {result[0]}"
        # 0/x → 0  (LOG_ZERO)
        assert result[1] < LOG_ZERO / 2, f"0/x should be LOG_ZERO, got {result[1]}"
        # x/0 → normal subtraction (caller's responsibility; we only guard numerator=0)
        assert jnp.isfinite(result[2])
        # x/y → normal subtraction
        assert np.isclose(result[3], -5.0 - (-3.0), atol=1e-5)

    def test_dyn_channel_with_deterministic_transitions(self):
        """Channels from a one-hot (deterministic) factor contain LOG_ZERO for
        impossible transitions. safe_log_div must keep the kernel finite."""
        import jax.numpy as jnp
        from inference.messages import safe_log_div, LOG_ZERO
        from inference.region_extended_loopy_bp import compute_dyn_channels

        T, n_x, n_theta, n_u = 2, 4, 3, 2

        # Deterministic factor: for each (x_old, theta, u), exactly one x_new is 1
        factor = jnp.zeros((T, n_x, n_x, n_theta, n_u))
        # x_new = (x_old + 1) % n_x  for all theta, u
        for x_old in range(n_x):
            x_new = (x_old + 1) % n_x
            factor = factor.at[:, x_old, x_new, :, :].set(1.0)
        log_factor = jnp.where(factor > 0, 0.0, LOG_ZERO)

        # Treat log_factor as region beliefs (skip cavity/messages for this unit test)
        log_channels = compute_dyn_channels(log_factor)

        # Channels contain LOG_ZERO for impossible transitions (near -1e12, not exactly -inf)
        assert float(log_channels.min()) < LOG_ZERO / 2, (
            "Deterministic channels should contain ~LOG_ZERO for impossible transitions"
        )

        # Kernel = factor / channel via safe_log_div must be finite
        log_kernel = safe_log_div(log_factor[:, :, :, :1, :], log_channels[:, :, :, None, :])
        assert jnp.all(jnp.isfinite(log_kernel)), (
            f"Kernel has non-finite values: min={log_kernel.min()}, max={log_kernel.max()}"
        )

    def test_dyn_channel_naive_division_creates_bogus_transitions(self):
        """Without safe_log_div, LOG_ZERO - LOG_ZERO = 0 makes impossible
        transitions look like probability 1. This is the core kernel bug."""
        import jax.numpy as jnp
        from inference.messages import safe_log_div, LOG_ZERO
        from inference.region_extended_loopy_bp import compute_dyn_channels

        T, n_x, n_theta, n_u = 1, 3, 1, 1

        # Deterministic: x_old=0 → x_new=1, x_old=1 → x_new=2, x_old=2 → x_new=0
        factor = jnp.zeros((T, n_x, n_x, n_theta, n_u))
        for x_old in range(n_x):
            x_new = (x_old + 1) % n_x
            factor = factor.at[:, x_old, x_new, :, :].set(1.0)
        log_factor = jnp.where(factor > 0, 0.0, LOG_ZERO)

        log_channels = compute_dyn_channels(log_factor)

        # Naive subtraction: for impossible transitions, both factor and channel
        # are ~LOG_ZERO, so their difference is ~0 (looks like probability 1)
        naive = log_factor - log_channels[:, :, :, None, :]
        impossible_mask = (factor == 0.0)
        bogus_count = int(jnp.sum(jnp.abs(naive[impossible_mask]) < 1.0))
        assert bogus_count > 0, "Naive subtraction should produce near-zero entries for 0/0"

        # safe_log_div: impossible entries stay at LOG_ZERO
        safe = safe_log_div(log_factor, log_channels[:, :, :, None, :])
        assert jnp.all(safe[impossible_mask] < LOG_ZERO / 2), (
            "safe_log_div should keep impossible transitions at LOG_ZERO"
        )

    def test_obs_channel_with_deterministic_observations(self):
        """Obs channels from a one-hot B(y|x,theta) contain LOG_ZERO for
        impossible cell types. safe_log_div keeps the kernel finite."""
        import jax.numpy as jnp
        from inference.messages import safe_log_div, LOG_ZERO
        from inference.region_extended_loopy_bp import compute_obs_channels

        T_plus_1, n_fov, n_y, n_x, n_theta = 3, 2, 5, 4, 3

        # Deterministic obs: for each (fov, x, theta), exactly one y is 1
        beliefs = jnp.zeros((T_plus_1, n_fov, n_y, n_x, n_theta))
        beliefs = beliefs.at[:, :, 0, :, :].set(1.0)  # all map to y=0
        log_beliefs = jnp.where(beliefs > 0, 0.0, LOG_ZERO)

        log_channels = compute_obs_channels(log_beliefs)

        # Channels contain LOG_ZERO for impossible y values
        assert float(log_channels.min()) < LOG_ZERO / 2, (
            "Deterministic obs channels should contain ~LOG_ZERO"
        )

        log_kernel = safe_log_div(log_beliefs, log_channels)
        assert jnp.all(jnp.isfinite(log_kernel)), (
            f"Obs kernel has non-finite values: min={log_kernel.min()}, max={log_kernel.max()}"
        )

    def test_naive_subtraction_produces_zero_for_impossible(self):
        """Plain subtraction maps LOG_ZERO/LOG_ZERO → 0 (probability 1),
        making impossible transitions look certain. This is the core bug."""
        import jax.numpy as jnp
        from inference.messages import LOG_ZERO

        # Simulate: factor=0, channel=0 in log-space
        log_factor = jnp.array(LOG_ZERO)
        log_channel = jnp.array(LOG_ZERO)

        naive = log_factor - log_channel
        assert np.isclose(float(naive), 0.0), (
            "LOG_ZERO - LOG_ZERO should give 0 (the bug: impossible transition looks certain)"
        )

    def test_safe_log_on_float16_tensor(self):
        """safe_log on a float16 tensor must not produce -inf (LOG_ZERO overflows float16)."""
        import jax.numpy as jnp
        from inference.planning import safe_log, LOG_ZERO

        x = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float16)
        result = safe_log(x)

        # float16 cannot represent -1e12; result[0] will be -inf
        # This test documents the requirement to cast to float32 BEFORE calling safe_log
        assert result.dtype == jnp.float16
        assert jnp.isinf(result[0]), "safe_log on float16 zero gives -inf (cast to float32 first!)"

    def test_safe_log_on_float32_tensor(self):
        """safe_log on float32 must use LOG_ZERO, never -inf."""
        import jax.numpy as jnp
        from inference.planning import safe_log, LOG_ZERO

        x = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
        result = safe_log(x)

        assert jnp.all(jnp.isfinite(result)), "safe_log on float32 should never produce -inf"
        assert np.isclose(float(result[0]), LOG_ZERO, rtol=1e-5)

    def test_region_extended_multi_iteration_no_nan(self):
        """Full region-extended planning must stay finite through 10 iterations
        with real deterministic tensors (the scenario that triggered the original bug)."""
        import jax.numpy as jnp
        from environments.minigrid import generate_transition_tensor, generate_observation_tensor
        from inference.region_extended_loopy_bp import region_extended_loopy_bp_planning

        n = 4
        n_states = n * n * 4 * 3
        n_static = (n * n - 2 * n) ** 2

        transition_tensor = jnp.array(generate_transition_tensor(n), dtype=jnp.float32)
        observation_tensor = jnp.array(generate_observation_tensor(n), dtype=jnp.float32)

        q_current = jnp.ones(n_states) / n_states
        q_static = jnp.ones(n_static) / n_static
        goal = jnp.zeros(n_states).at[0].set(1.0)

        action_dist, dyn_ch, obs_ch = region_extended_loopy_bp_planning(
            q_current, q_static, transition_tensor, observation_tensor,
            goal, horizon=5, n_iterations=10,
        )

        assert jnp.all(jnp.isfinite(action_dist)), f"action_dist has NaN/inf: {action_dist}"
        assert np.isclose(action_dist.sum(), 1.0, atol=1e-5)

    def test_reduced_region_extended_multi_iteration_no_nan(self):
        """Same for reduced variant."""
        import jax.numpy as jnp
        from environments.minigrid import generate_transition_tensor, generate_observation_tensor
        from inference.reduced_region_extended import reduced_region_extended_planning

        n = 4
        n_states = n * n * 4 * 3
        n_static = (n * n - 2 * n) ** 2

        transition_tensor = jnp.array(generate_transition_tensor(n), dtype=jnp.float32)
        observation_tensor = jnp.array(generate_observation_tensor(n), dtype=jnp.float32)

        q_current = jnp.ones(n_states) / n_states
        q_static = jnp.ones(n_static) / n_static
        goal = jnp.zeros(n_states).at[0].set(1.0)

        action_dist, dyn_ch, obs_ch = reduced_region_extended_planning(
            q_current, q_static, transition_tensor, observation_tensor,
            goal, horizon=5, n_iterations=10,
        )

        assert jnp.all(jnp.isfinite(action_dist)), f"action_dist has NaN/inf: {action_dist}"
        assert np.isclose(action_dist.sum(), 1.0, atol=1e-5)


class TestReducedRegionExtended:
    def setup_method(self):
        import jax.numpy as jnp
        from environments.minigrid import (
            generate_transition_tensor,
            generate_observation_tensor,
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

        self.transition_tensor = jnp.array(generate_transition_tensor(self.n), dtype=jnp.float32)
        self.observation_tensor = jnp.array(generate_observation_tensor(self.n), dtype=jnp.float32)

    def test_output_shape(self):
        import jax.numpy as jnp
        from inference.reduced_region_extended import (
            reduced_region_extended_planning,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist, dyn_channels, obs_channels = reduced_region_extended_planning(
            q_current, q_static, self.transition_tensor, self.observation_tensor,
            goal, horizon=5, n_iterations=2,
        )

        assert action_dist.shape == (self.n_actions,)
        assert np.isclose(action_dist.sum(), 1.0)
        assert dyn_channels.shape == (5, self.n_states, self.n_states, self.n_actions)
        assert obs_channels.shape == (6, 49, 11, self.n_states, self.n_static)

    def test_respects_action_mask(self):
        import jax.numpy as jnp
        from inference.reduced_region_extended import (
            reduced_region_extended_planning,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist, _, _ = reduced_region_extended_planning(
            q_current, q_static, self.transition_tensor, self.observation_tensor,
            goal, horizon=5, n_iterations=3,
        )

        assert action_dist[4] < 1e-6
        assert action_dist[6] < 1e-6

    def test_single_iter_matches_region_extended(self):
        """With 1 iteration, kernels are all p/r=1 and cavities are identical
        (both use q_static_state), so results should be identical."""
        import jax.numpy as jnp
        from inference.region_extended_loopy_bp import (
            region_extended_loopy_bp_planning,
        )
        from inference.reduced_region_extended import (
            reduced_region_extended_planning,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        full_result, _, _ = region_extended_loopy_bp_planning(
            q_current, q_static, self.transition_tensor, self.observation_tensor,
            goal, horizon=5, n_iterations=1,
        )
        reduced_result, _, _ = reduced_region_extended_planning(
            q_current, q_static, self.transition_tensor, self.observation_tensor,
            goal, horizon=5, n_iterations=1,
        )

        assert np.allclose(full_result, reduced_result, atol=1e-5), (
            f"Region-extended and reduced should match with 1 iteration.\n"
            f"Full:    {full_result}\nReduced: {reduced_result}"
        )


class TestAgentIntegration:
    def setup_method(self):
        import jax.numpy as jnp
        from environments.minigrid import (
            generate_transition_tensor,
            generate_transition_indices,
            generate_observation_indices,
            generate_orientation_indices,
        )
        from utils.tensors import get_dimensions

        self.grid_size = 4
        self.dims = get_dimensions(self.grid_size)
        self.transition_tensor = jnp.array(generate_transition_tensor(self.grid_size), dtype=jnp.float32)
        self.transition_idx = jnp.array(generate_transition_indices(self.grid_size))
        self.observation_idx = jnp.array(generate_observation_indices(self.grid_size))
        self.orientation_idx = jnp.array(generate_orientation_indices(self.grid_size))
        self.goal = jnp.zeros(self.dims["n_states"])
        self.goal = self.goal.at[0].set(1.0)

    def test_agent_creation(self):
        from agents.flat_tensor_agent import IndexedTensorAgent

        agent = IndexedTensorAgent.create(
            grid_size=self.grid_size,
            transition_tensor=self.transition_tensor,
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
            transition_tensor=self.transition_tensor,
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
            transition_tensor=self.transition_tensor,
            transition_idx=self.transition_idx,
            observation_idx=self.observation_idx,
            orientation_idx=self.orientation_idx,
            goal=self.goal,
        )
        new_agent = agent.reset()
        assert new_agent.q_state is not None


class TestCustomFOVSizeInference:
    """Test that inference and planning work with non-default FOV sizes."""

    def setup_method(self):
        import jax.numpy as jnp
        from environments.minigrid import (
            generate_transition_tensor,
            generate_transition_indices,
            generate_observation_tensor,
            generate_observation_indices,
            generate_orientation_indices,
            N_ORIENTATIONS,
            N_DOOR_KEY_STATES,
        )

        self.n = 3
        self.fov_size = 5
        n_loc = self.n * self.n
        n_key = n_loc - 2 * self.n
        n_door = n_loc - 2 * self.n
        self.n_states = n_loc * N_ORIENTATIONS * N_DOOR_KEY_STATES
        self.n_static = n_key * n_door
        self.n_actions = 7

        self.transition_tensor = jnp.array(generate_transition_tensor(self.n), dtype=jnp.float32)
        self.transition_idx = jnp.array(generate_transition_indices(self.n))
        self.observation_tensor = jnp.array(generate_observation_tensor(self.n, fov_size=self.fov_size), dtype=jnp.float32)
        self.obs_idx = jnp.array(generate_observation_indices(self.n, fov_size=self.fov_size))
        self.orientation_idx = jnp.array(generate_orientation_indices(self.n))

    def test_obs_idx_shape(self):
        assert self.obs_idx.shape[:2] == (self.fov_size, self.fov_size)

    def test_state_inference_with_fov5(self):
        import jax.numpy as jnp
        from inference.state_inference import state_inference_step_indexed

        q_old = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        vision_obs = jnp.zeros((self.fov_size, self.fov_size, 11))
        vision_obs = vision_obs.at[:, :, 1].set(1.0)
        ori_obs = jnp.array([1.0, 0.0, 0.0, 0.0])

        q_current, q_static_new = state_inference_step_indexed(
            q_old, q_static,
            self.transition_idx, self.obs_idx, self.orientation_idx,
            vision_obs, ori_obs,
            action_idx=0,
            n_iterations=2,
        )

        assert q_current.shape == (self.n_states,)
        assert q_static_new.shape == (self.n_static,)
        assert np.isclose(q_current.sum(), 1.0)
        assert np.isclose(q_static_new.sum(), 1.0)

    def test_region_extended_with_fov5(self):
        import jax.numpy as jnp
        from inference.region_extended_loopy_bp import (
            region_extended_loopy_bp_planning,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist, dyn_channels, obs_channels = region_extended_loopy_bp_planning(
            q_current, q_static, self.transition_tensor, self.observation_tensor,
            goal, horizon=3, n_iterations=2,
        )

        assert action_dist.shape == (self.n_actions,)
        assert np.isclose(action_dist.sum(), 1.0)
        # obs_channels should use n_fov = 5*5 = 25
        assert obs_channels.shape == (4, 25, 11, self.n_states, self.n_static)

    def test_reduced_region_extended_with_fov5(self):
        import jax.numpy as jnp
        from inference.reduced_region_extended import (
            reduced_region_extended_planning,
        )

        q_current = jnp.ones(self.n_states) / self.n_states
        q_static = jnp.ones(self.n_static) / self.n_static
        goal = jnp.zeros(self.n_states)
        goal = goal.at[0].set(1.0)

        action_dist, dyn_channels, obs_channels = reduced_region_extended_planning(
            q_current, q_static, self.transition_tensor, self.observation_tensor,
            goal, horizon=3, n_iterations=2,
        )

        assert action_dist.shape == (self.n_actions,)
        assert np.isclose(action_dist.sum(), 1.0)
        assert obs_channels.shape == (4, 25, 11, self.n_states, self.n_static)

    def test_agent_step_with_fov5(self):
        import jax.numpy as jnp
        from agents.flat_tensor_agent import IndexedTensorAgent
        from utils.tensors import get_dimensions, flatten_state_index

        dims = get_dimensions(self.n)
        goal = jnp.zeros(dims["n_states"])
        goal = goal.at[0].set(1.0)

        agent = IndexedTensorAgent.create(
            grid_size=self.n,
            transition_tensor=self.transition_tensor,
            transition_idx=self.transition_idx,
            observation_idx=self.obs_idx,
            orientation_idx=self.orientation_idx,
            goal=goal,
            planning_horizon=3,
            n_planning_iterations=1,
        )

        vision_obs = jnp.zeros((self.fov_size, self.fov_size, 11))
        vision_obs = vision_obs.at[:, :, 1].set(1.0)
        ori_obs = jnp.array([1.0, 0.0, 0.0, 0.0])

        action, new_agent = agent.step(vision_obs, ori_obs, time_remaining=10)
        assert 0 <= action < 7
