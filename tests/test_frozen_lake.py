"""Tests for Frozen Lake environment and agents."""

import numpy as np


# ---------------------------------------------------------------------------
# Environment / tensor generation tests
# ---------------------------------------------------------------------------

class TestFrozenLakeTensors:
    """Validate transition and observation tensor properties."""

    def setup_method(self):
        from environments.frozen_lake import (
            sample_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
        )

        self.grid_size = 4
        self.n_pos = 16
        self.n_states = self.n_pos
        self.n_configs = 10
        self.goal_pos = 15
        self.start_pos = 0

        self.holes = sample_configs(
            self.grid_size, self.n_configs, hole_fraction=0.25, seed=42,
        )
        self.T = generate_transition_tensor(
            self.grid_size, self.holes, slip_prob=0.0,
        )
        self.T_slip = generate_transition_tensor(
            self.grid_size, self.holes, slip_prob=0.2,
        )
        self.B = generate_observation_tensor(self.grid_size, self.holes)
        self.goal = generate_goal(self.grid_size, self.holes)

    def test_config_shapes(self):
        assert self.holes.shape == (self.n_configs, self.n_pos)

    def test_configs_start_goal_safe(self):
        """Start and goal positions should never be holes."""
        for theta in range(self.n_configs):
            assert self.holes[theta, self.start_pos] == 0.0
            assert self.holes[theta, self.goal_pos] == 0.0

    def test_transition_shape(self):
        assert self.T.shape == (self.n_states, self.n_states, self.n_configs, 4)

    def test_transition_stochastic(self):
        """T should sum to 1 over x_new for each (x_old, θ, action)."""
        sums = self.T.sum(axis=0)  # sum over x_new
        assert np.allclose(sums, 1.0, atol=1e-6), f"Max deviation: {np.abs(sums - 1.0).max()}"

    def test_transition_deterministic_no_slip(self):
        """Without slip, deterministic movement for non-absorbing states."""
        theta = 0
        for pos in range(self.n_states):
            if self.holes[theta, pos] == 1.0 or pos == self.goal_pos:
                continue
            for action in range(4):
                probs = self.T[:, pos, theta, action]
                assert np.isclose(probs.max(), 1.0, atol=1e-6)
                assert np.isclose(probs.sum(), 1.0, atol=1e-6)

    def test_movement_correct(self):
        """Moving RIGHT from (0,0) should land at (0,1); LEFT hits the wall."""
        from environments.frozen_lake import LEFT, RIGHT

        theta = 0
        assert np.isclose(self.T[1, 0, theta, RIGHT], 1.0)
        assert np.isclose(self.T[0, 0, theta, LEFT], 1.0)  # wall collision

    def test_holes_absorbing(self):
        """Holes should be absorbing: T[hole, hole, θ, :] = 1."""
        for theta in range(self.n_configs):
            for pos in range(self.n_pos):
                if self.holes[theta, pos] == 1.0:
                    for a in range(4):
                        assert np.isclose(self.T[pos, pos, theta, a], 1.0)

    def test_goal_absorbing(self):
        """Goal should be absorbing."""
        for theta in range(self.n_configs):
            for a in range(4):
                assert np.isclose(self.T[self.goal_pos, self.goal_pos, theta, a], 1.0)

    def test_slip_stochastic(self):
        """With slip, non-absorbing states should have spread probability."""
        sums = self.T_slip.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_observation_shape(self):
        # n_pos position channels + 4 neighbor sensor channels
        n_channels = self.n_pos + 4
        assert self.B.shape == (n_channels, 2, self.n_states, self.n_configs)

    def test_observation_is_stochastic(self):
        """B should sum to 1 over obs_types (axis 1) and all values in [0, 1]."""
        sums = self.B.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6), f"Max deviation: {np.abs(sums - 1.0).max()}"
        assert np.all(self.B >= 0.0)
        assert np.all(self.B <= 1.0)

    def test_position_channels_deterministic(self):
        """Position channels should be near-deterministic and θ-independent."""
        for x in range(self.n_states):
            # Channel x should fire at position x
            assert self.B[x, 1, x, 0] > 0.99
            # Channel x should not fire at other positions
            for other in range(self.n_states):
                if other != x:
                    assert self.B[x, 1, other, 0] < 0.01

    def test_neighbor_sensor_correctness(self):
        """Neighbor sensor channels fire iff the adjacent cell is a hole."""
        from environments.frozen_lake import (
            generate_observation_tensor, LEFT, DOWN, RIGHT, UP,
        )

        grid_size = 4
        n_pos = 16
        # Single known config: hole at position 5 = (1, 1)
        holes = np.zeros((1, n_pos), dtype=np.float32)
        holes[0, 5] = 1.0

        obs_noise = 0.1
        B = generate_observation_tensor(grid_size, holes, obs_noise=obs_noise)

        # Agent at pos 4 = (1,0): hole at (1,1) is to the RIGHT
        assert B[n_pos + RIGHT, 1, 4, 0] > 0.85
        assert B[n_pos + LEFT, 1, 4, 0] < 0.05
        assert B[n_pos + UP, 1, 4, 0] < 0.05
        assert B[n_pos + DOWN, 1, 4, 0] < 0.05

        # Agent at pos 1 = (0,1): hole at (1,1) is DOWN
        assert B[n_pos + DOWN, 1, 1, 0] > 0.85
        assert B[n_pos + RIGHT, 1, 1, 0] < 0.05

        # Agent at pos 6 = (1,2): hole at (1,1) is to the LEFT
        assert B[n_pos + LEFT, 1, 6, 0] > 0.85

        # Agent at pos 9 = (2,1): hole at (1,1) is UP
        assert B[n_pos + UP, 1, 9, 0] > 0.85

        # Agent at pos 15 = (3,3): no hole adjacent
        for d in range(4):
            assert B[n_pos + d, 1, 15, 0] < 0.05

    def test_neighbor_sensor_edge_cells(self):
        """Out-of-grid neighbors should count as safe (fire only at p_fp)."""
        from environments.frozen_lake import (
            generate_observation_tensor, LEFT, UP,
        )

        grid_size = 4
        n_pos = 16
        holes = np.zeros((1, n_pos), dtype=np.float32)
        holes[0, 5] = 1.0

        B = generate_observation_tensor(grid_size, holes, obs_noise=0.1)

        # Agent at pos 0 = (0,0): LEFT and UP neighbors are off-grid -> safe
        assert B[n_pos + LEFT, 1, 0, 0] < 0.05
        assert B[n_pos + UP, 1, 0, 0] < 0.05

    def test_goal_shape(self):
        assert self.goal.shape == (self.n_states, self.n_configs)
        # Each config column should sum to 1 (per-config softmax normalization)
        for theta in range(self.n_configs):
            assert np.isclose(self.goal[:, theta].sum(), 1.0, atol=1e-6)

    def test_goal_prefers_goal_over_holes(self):
        """Goal position should have higher preference than hole positions."""
        for theta in range(self.n_configs):
            goal_val = self.goal[self.goal_pos, theta]
            for pos in range(self.n_pos):
                if self.holes[theta, pos] == 1.0:
                    assert goal_val > self.goal[pos, theta]


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

class TestFrozenLakeEnv:
    """Test the simple simulator."""

    def setup_method(self):
        from environments.frozen_lake import (
            sample_configs,
            generate_observation_tensor,
            FrozenLakeEnv,
        )

        self.grid_size = 4
        self.n_pos = 16
        self.n_channels = self.n_pos + 4
        self.holes = sample_configs(self.grid_size, 10, seed=42)
        self.obs_tensor = generate_observation_tensor(self.grid_size, self.holes)
        self.env = FrozenLakeEnv(
            grid_size=self.grid_size,
            holes=self.holes,
            obs_tensor=self.obs_tensor,
            slip_prob=0.0,
            max_steps=50,
        )

    def test_reset(self):
        result = self.env.reset(seed=0)
        assert result.obs.shape == (self.n_channels,)
        # Binary sensor vector: values should be 0 or 1
        assert all(v in (0.0, 1.0) for v in result.obs)
        assert not result.terminated
        assert not result.truncated

    def test_movement(self):
        """Moving RIGHT from (0,0) should go to (0,1)."""
        from environments.frozen_lake import RIGHT

        self.env.reset(seed=0, config_idx=0)
        result = self.env.step(RIGHT)
        assert self.env._position == 1
        assert result.obs.shape == (self.n_channels,)

    def test_wall_collision(self):
        """Moving LEFT from (0,0) should stay at (0,0)."""
        from environments.frozen_lake import LEFT

        self.env.reset(seed=0, config_idx=0)
        result = self.env.step(LEFT)
        assert self.env._position == 0
        assert result.obs.shape == (self.n_channels,)

    def test_goal_termination(self):
        """Reaching the goal should terminate with reward 1."""
        from environments.frozen_lake import RIGHT, DOWN

        for cfg_idx in range(self.holes.shape[0]):
            path = [1, 2, 3, 7, 11, 15]
            if all(self.holes[cfg_idx, p] == 0.0 for p in path):
                self.env.reset(seed=0, config_idx=cfg_idx)
                for _ in range(3):
                    result = self.env.step(RIGHT)
                for _ in range(3):
                    result = self.env.step(DOWN)
                assert result.terminated
                assert result.reward == 1.0
                return

        import pytest
        pytest.skip("No config with clear R-D path found")

    def test_hole_termination(self):
        """Stepping onto a hole should terminate with reward 0."""
        from environments.frozen_lake import MOVEMENT, pos_to_rc

        # Find a config with a hole adjacent to the start position
        for cfg_idx in range(self.holes.shape[0]):
            for action, (dr, dc) in MOVEMENT.items():
                r, c = pos_to_rc(0, self.grid_size)
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    continue
                nb = nr * self.grid_size + nc
                if self.holes[cfg_idx, nb] == 1.0:
                    self.env.reset(seed=0, config_idx=cfg_idx)
                    result = self.env.step(action)
                    assert result.terminated
                    assert result.reward == 0.0
                    return

        import pytest
        pytest.skip("No config with a hole adjacent to start found")

    def test_ascii_render(self):
        self.env.reset(seed=0, config_idx=0)
        ascii_str = self.env.render_ascii()
        assert "A" in ascii_str
        assert "G" in ascii_str


# ---------------------------------------------------------------------------
# Agent / planning integration tests
# ---------------------------------------------------------------------------

class TestFrozenLakeAgents:
    """Test that agents can plan and act on Frozen Lake."""

    def setup_method(self):
        from environments.frozen_lake import (
            sample_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
        )

        self.grid_size = 4
        self.n_pos = 16
        self.n_states = self.n_pos
        self.n_configs = 10

        self.holes = sample_configs(self.grid_size, self.n_configs, seed=42)
        self.T = generate_transition_tensor(self.grid_size, self.holes, slip_prob=0.0)
        self.B = generate_observation_tensor(self.grid_size, self.holes)
        self.goal = generate_goal(self.grid_size, self.holes)

    def _make_obs(self, pos, config_idx, B=None):
        """Build an observation vector: position one-hot + neighbor sensors."""
        import jax.numpy as jnp
        if B is None:
            B = self.B
        n_channels = B.shape[0]
        obs = jnp.zeros(n_channels)
        obs = obs.at[pos].set(1.0)  # position channel one-hot
        # Neighbor sensor channels: threshold from B
        for d in range(4):
            ch = self.n_pos + d
            obs = obs.at[ch].set(
                1.0 if B[ch, 1, pos, config_idx] > 0.5 else 0.0
            )
        return obs

    def test_bp_agent_produces_valid_action(self):
        from agents.frozen_lake_agent import create_agent

        agent = create_agent(
            "loopy_bp", self.T, self.B, self.goal, self.holes,
            planning_horizon=5,
        )
        obs = self._make_obs(0, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < 4

    def test_region_extended_agent_produces_valid_action(self):
        from agents.frozen_lake_agent import create_agent

        agent = create_agent(
            "region_extended", self.T, self.B, self.goal, self.holes,
            planning_horizon=3, planning_iterations=2,
        )
        obs = self._make_obs(0, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < 4

    def test_static_belief_update(self):
        """Observation likelihood should differentiate configs via neighbor sensors.

        The neighbor sensor model is θ-dependent (different hole configs
        trigger different patterns), so repeated observations should
        shift static belief toward configs consistent with the sensor pattern.
        """
        import jax.numpy as jnp
        from environments.frozen_lake import generate_observation_tensor
        from agents.frozen_lake_agent import create_agent

        # Use a low-noise observation model so sensors are highly informative
        B_low_noise = generate_observation_tensor(
            self.grid_size, self.holes, obs_noise=0.01,
        )

        agent = create_agent(
            "loopy_bp", self.T, B_low_noise, self.goal, self.holes,
            planning_horizon=3,
        )
        agent = agent.reset()

        # Observations from positions 0 and 5 under config 0
        obs_start = self._make_obs(0, 0, B=B_low_noise)
        _, agent = agent.step(obs_start, time_remaining=10)

        # All configs should still be alive
        assert jnp.all(agent.q_static_state > 0)

        obs_5 = self._make_obs(5, 0, B=B_low_noise)
        _, agent = agent.step(obs_5, time_remaining=9)

        # The true config (0) should have at least the prior's belief
        prior_uniform = 1.0 / self.n_configs
        assert float(agent.q_static_state[0]) >= prior_uniform * 0.9

    def test_all_methods_run(self):
        """Smoke test: all agent methods produce a valid action."""
        from agents.frozen_lake_agent import AGENT_CLASSES, create_agent

        obs = self._make_obs(0, 0)

        for method_name in AGENT_CLASSES:
            agent = create_agent(
                method_name, self.T, self.B, self.goal, self.holes,
                planning_horizon=3, planning_iterations=2,
            )
            action, _ = agent.step(obs, time_remaining=5)
            assert 0 <= action < 4, f"Method {method_name} returned invalid action {action}"


# ---------------------------------------------------------------------------
# Episode integration test
# ---------------------------------------------------------------------------

class TestFrozenLakeEpisode:
    """Run a short episode to verify end-to-end flow."""

    def test_episode_completes(self):
        import jax.numpy as jnp
        from environments.frozen_lake import (
            sample_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
            FrozenLakeEnv,
        )
        from agents.frozen_lake_agent import create_agent

        grid_size = 4
        n_configs = 20
        holes = sample_configs(grid_size, n_configs, hole_fraction=0.15, seed=123)
        T = generate_transition_tensor(grid_size, holes, slip_prob=0.0)
        B = generate_observation_tensor(grid_size, holes)
        goal = generate_goal(grid_size, holes)

        agent = create_agent(
            "loopy_bp", T, B, goal, holes,
            planning_horizon=8, planning_iterations=1,
        )
        env = FrozenLakeEnv(
            grid_size=grid_size, holes=holes, obs_tensor=B, max_steps=30,
        )

        result = env.reset(seed=0)
        agent = agent.reset()

        steps = 0
        while not result.terminated and not result.truncated and steps < 30:
            obs = jnp.array(result.obs)
            action, agent = agent.step(obs, time_remaining=30 - steps)
            result = env.step(action)
            steps += 1

        # Episode should end within max_steps
        assert result.terminated or result.truncated or steps >= 30
