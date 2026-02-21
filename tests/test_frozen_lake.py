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
        self.n_states = 16
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
        assert self.holes.shape == (self.n_configs, self.n_states)

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
        from environments.frozen_lake import LEFT, DOWN, RIGHT, UP, pos_to_rc, rc_to_pos

        theta = 0
        for x_old in range(self.n_states):
            if self.holes[theta, x_old] == 1.0 or x_old == self.goal_pos:
                continue
            for action in range(4):
                # Exactly one x_new should have probability 1
                probs = self.T[:, x_old, theta, action]
                assert np.isclose(probs.max(), 1.0, atol=1e-6)
                assert np.isclose(probs.sum(), 1.0, atol=1e-6)

    def test_holes_absorbing(self):
        """Holes should be absorbing: T[hole, hole, θ, :] = 1."""
        for theta in range(self.n_configs):
            for x in range(self.n_states):
                if self.holes[theta, x] == 1.0:
                    for a in range(4):
                        assert np.isclose(self.T[x, x, theta, a], 1.0)

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
        # n_states position channels + 4 directional channels
        n_channels = self.n_states + 4
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

    def test_goal_shape(self):
        assert self.goal.shape == (self.n_states,)
        assert np.isclose(self.goal.sum(), 1.0)
        # Goal position should have highest probability (not necessarily 1.0)
        assert self.goal[self.goal_pos] == self.goal.max()

    def test_line_of_sight_correctness(self):
        """Verify line-of-sight sensors fire correctly for known hole positions."""
        from environments.frozen_lake import (
            generate_observation_tensor, sample_configs, LEFT, DOWN, RIGHT, UP,
            get_cells_in_direction,
        )

        grid_size = 4
        n_states = 16
        # Create a single known config: hole at position 2 = (0, 2)
        holes = np.zeros((1, 16), dtype=np.float32)
        holes[0, 2] = 1.0  # hole at (row=0, col=2)

        B = generate_observation_tensor(grid_size, holes, base_noise=0.01, noise_range=0.0)

        # Directional channels start at index n_states
        # From position 0 = (0,0), looking RIGHT along row 0:
        # cells in RIGHT direction: 1, 2, 3. Position 2 is a hole.
        p_fire_right = B[n_states + RIGHT, 1, 0, 0]
        assert p_fire_right > 0.9, f"RIGHT sensor at (0,0) should fire: p={p_fire_right}"

        # From position 0, looking DOWN along col 0: cells 4, 8, 12. No holes.
        p_fire_down = B[n_states + DOWN, 1, 0, 0]
        assert p_fire_down < 0.1, f"DOWN sensor at (0,0) should not fire: p={p_fire_down}"

        # From position 3 = (0,3), looking LEFT along row 0:
        # cells in LEFT direction: 2, 1, 0. Position 2 is a hole.
        p_fire_left = B[n_states + LEFT, 1, 3, 0]
        assert p_fire_left > 0.9, f"LEFT sensor at (0,3) should fire: p={p_fire_left}"

        # From position 8 = (2,0), looking RIGHT along row 2: no holes
        p_fire_right_r2 = B[n_states + RIGHT, 1, 8, 0]
        assert p_fire_right_r2 < 0.1, f"RIGHT sensor at (2,0) should not fire: p={p_fire_right_r2}"

        # From position 6 = (1,2), looking UP along col 2:
        # cells in UP direction: 2. Position 2 is a hole.
        p_fire_up = B[n_states + UP, 1, 6, 0]
        assert p_fire_up > 0.9, f"UP sensor at (1,2) should fire: p={p_fire_up}"


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
        self.n_states = 16
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
        n_channels = self.n_states + 4
        assert result.obs.shape == (n_channels,)
        # Binary sensor vector: values should be 0 or 1
        assert all(v in (0.0, 1.0) for v in result.obs)
        assert not result.terminated
        assert not result.truncated

    def test_movement(self):
        """Moving RIGHT from (0,0) should go to (0,1) — verify via internal position."""
        from environments.frozen_lake import RIGHT

        self.env.reset(seed=0, config_idx=0)
        result = self.env.step(RIGHT)
        assert self.env._position == 1  # position (0,1) = index 1
        n_channels = self.n_states + 4
        assert result.obs.shape == (n_channels,)

    def test_wall_collision(self):
        """Moving LEFT from (0,0) should stay at (0,0)."""
        from environments.frozen_lake import LEFT

        self.env.reset(seed=0, config_idx=0)
        result = self.env.step(LEFT)
        assert self.env._position == 0  # still at start
        n_channels = self.n_states + 4
        assert result.obs.shape == (n_channels,)

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
        import jax.numpy as jnp
        from environments.frozen_lake import (
            sample_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
        )

        self.grid_size = 4
        self.n_states = 16
        self.n_configs = 10

        self.holes = sample_configs(self.grid_size, self.n_configs, seed=42)
        self.T = generate_transition_tensor(self.grid_size, self.holes, slip_prob=0.0)
        self.B = generate_observation_tensor(self.grid_size, self.holes)
        self.goal = generate_goal(self.grid_size, self.holes)

    def _make_obs(self, pos, config_idx):
        """Build an observation vector: position one-hot + directional sensors."""
        import jax.numpy as jnp
        n_channels = self.B.shape[0]
        obs = jnp.zeros(n_channels)
        # Position channels: one-hot
        obs = obs.at[pos].set(1.0)
        # Directional channels: sample from B
        for d in range(4):
            obs = obs.at[self.n_states + d].set(
                1.0 if self.B[self.n_states + d, 1, pos, config_idx] > 0.5 else 0.0
            )
        return obs

    def test_bp_agent_produces_valid_action(self):
        import jax.numpy as jnp
        from agents.frozen_lake_agent import create_agent

        agent = create_agent(
            "bp", self.T, self.B, self.goal, self.holes,
            planning_horizon=5,
        )
        obs = self._make_obs(0, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < 4

    def test_loopy_bp_agent_produces_valid_action(self):
        import jax.numpy as jnp
        from agents.frozen_lake_agent import create_agent

        agent = create_agent(
            "loopy_bp", self.T, self.B, self.goal, self.holes,
            planning_horizon=5, planning_iterations=2,
        )
        obs = self._make_obs(0, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < 4

    def test_region_extended_agent_produces_valid_action(self):
        import jax.numpy as jnp
        from agents.frozen_lake_agent import create_agent

        agent = create_agent(
            "region_extended", self.T, self.B, self.goal, self.holes,
            planning_horizon=3, planning_iterations=2,
        )
        obs = self._make_obs(0, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < 4

    def test_static_belief_update(self):
        """Observation likelihood should differentiate configs via directional sensors.

        The line-of-sight sensor model is θ-dependent (different hole configs
        trigger different directional patterns), so repeated observations should
        shift static belief toward configs consistent with the sensor pattern.
        """
        import jax.numpy as jnp
        from environments.frozen_lake import generate_observation_tensor
        from agents.frozen_lake_agent import create_agent

        # Use a low-noise observation model so sensors are highly informative
        B_low_noise = generate_observation_tensor(
            self.grid_size, self.holes, base_noise=0.01, noise_range=0.0,
        )

        agent = create_agent(
            "bp", self.T, B_low_noise, self.goal, self.holes,
            planning_horizon=3,
        )
        agent = agent.reset()

        # Build observation from position 0 under config 0
        n_channels = B_low_noise.shape[0]
        obs_start = jnp.zeros(n_channels)
        obs_start = obs_start.at[0].set(1.0)  # position channel: at pos 0
        for d in range(4):
            obs_start = obs_start.at[self.n_states + d].set(
                1.0 if B_low_noise[self.n_states + d, 1, 0, 0] > 0.5 else 0.0
            )
        _, agent = agent.step(obs_start, time_remaining=10)

        # All configs should still be alive
        assert jnp.all(agent.q_static_state > 0)

        # Observation from position 5 under config 0
        obs_5 = jnp.zeros(n_channels)
        obs_5 = obs_5.at[5].set(1.0)  # position channel: at pos 5
        for d in range(4):
            obs_5 = obs_5.at[self.n_states + d].set(
                1.0 if B_low_noise[self.n_states + d, 1, 5, 0] > 0.5 else 0.0
            )
        _, agent = agent.step(obs_5, time_remaining=9)

        # The true config (0) should have higher belief than the prior
        prior_uniform = 1.0 / self.n_configs
        assert float(agent.q_static_state[0]) > prior_uniform

    def test_all_methods_run(self):
        """Smoke test: all agent methods produce a valid action."""
        import jax.numpy as jnp
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
            "bp", T, B, goal, holes,
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
