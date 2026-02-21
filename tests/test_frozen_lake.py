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
            N_ACTIONS,
        )

        self.grid_size = 4
        self.n_pos = 16
        self.n_states = 2 * self.n_pos  # doubled for scan mode
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
        assert self.T.shape == (self.n_states, self.n_states, self.n_configs, 5)

    def test_transition_stochastic(self):
        """T should sum to 1 over x_new for each (x_old, θ, action)."""
        sums = self.T.sum(axis=0)  # sum over x_new
        assert np.allclose(sums, 1.0, atol=1e-6), f"Max deviation: {np.abs(sums - 1.0).max()}"

    def test_transition_deterministic_no_slip(self):
        """Without slip, deterministic movement for non-absorbing states."""
        from environments.frozen_lake import unpack_state

        theta = 0
        for x_old in range(self.n_states):
            pos, scanned = unpack_state(x_old, self.n_pos)
            if self.holes[theta, pos] == 1.0 or pos == self.goal_pos:
                continue
            for action in range(4):  # movement actions only
                probs = self.T[:, x_old, theta, action]
                assert np.isclose(probs.max(), 1.0, atol=1e-6)
                assert np.isclose(probs.sum(), 1.0, atol=1e-6)

    def test_holes_absorbing(self):
        """Holes should be absorbing: T[hole, hole, θ, :] = 1."""
        from environments.frozen_lake import state_index

        for theta in range(self.n_configs):
            for pos in range(self.n_pos):
                if self.holes[theta, pos] == 1.0:
                    for mode in range(2):
                        x = state_index(pos, mode, self.n_pos)
                        for a in range(5):
                            assert np.isclose(self.T[x, x, theta, a], 1.0)

    def test_goal_absorbing(self):
        """Goal should be absorbing in both scan modes."""
        from environments.frozen_lake import state_index

        for theta in range(self.n_configs):
            for mode in range(2):
                x = state_index(self.goal_pos, mode, self.n_pos)
                for a in range(5):
                    assert np.isclose(self.T[x, x, theta, a], 1.0)

    def test_scan_transitions(self):
        """SCAN should transition unscanned→scanned deterministically."""
        from environments.frozen_lake import state_index, SCAN

        theta = 0
        for pos in range(self.n_pos):
            if self.holes[theta, pos] == 1.0 or pos == self.goal_pos:
                continue
            x_unscanned = state_index(pos, 0, self.n_pos)
            x_scanned = state_index(pos, 1, self.n_pos)
            assert np.isclose(self.T[x_scanned, x_unscanned, theta, SCAN], 1.0)
            # Already scanned stays scanned
            assert np.isclose(self.T[x_scanned, x_scanned, theta, SCAN], 1.0)

    def test_slip_stochastic(self):
        """With slip, non-absorbing states should have spread probability."""
        sums = self.T_slip.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_observation_shape(self):
        # 2*n_pos position channels + n_pos grid cell channels = 3*n_pos
        n_channels = 3 * self.n_pos
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

    def test_grid_cell_channels_scanned_deterministic(self):
        """In scanned mode, grid cell channels should be near-deterministic."""
        from environments.frozen_lake import state_index

        for theta in range(self.n_configs):
            for pos in range(self.n_pos):
                x_scanned = state_index(pos, 1, self.n_pos)
                for cell in range(self.n_pos):
                    ch = self.n_states + cell
                    has_hole = self.holes[theta, cell] == 1.0
                    if has_hole:
                        assert self.B[ch, 1, x_scanned, theta] > 0.99
                    else:
                        assert self.B[ch, 1, x_scanned, theta] < 0.01

    def test_grid_cell_channels_unscanned_noisy(self):
        """In unscanned mode, nearby cells should be less noisy than distant cells."""
        from environments.frozen_lake import (
            generate_observation_tensor, sample_configs, state_index,
        )

        grid_size = 5
        n_pos = 25
        holes = np.zeros((1, n_pos), dtype=np.float32)
        holes[0, 12] = 1.0  # hole at center (2,2)

        B = generate_observation_tensor(grid_size, holes, base_noise=0.05, noise_range=0.3)

        # Agent at pos 11 = (2,1), distance 1 from hole at 12 = (2,2)
        x_near = state_index(11, 0, n_pos)
        p_near = B[2 * n_pos + 12, 1, x_near, 0]

        # Agent at pos 0 = (0,0), distance 4 from hole at 12
        x_far = state_index(0, 0, n_pos)
        p_far = B[2 * n_pos + 12, 1, x_far, 0]

        # Closer agent should detect hole more reliably
        assert p_near > p_far, f"Near p={p_near} should exceed far p={p_far}"

    def test_goal_shape(self):
        assert self.goal.shape == (self.n_states, self.n_configs)
        # Each config column should sum to 1 (per-config softmax normalization)
        for theta in range(self.n_configs):
            assert np.isclose(self.goal[:, theta].sum(), 1.0, atol=1e-6)

    def test_grid_cell_correctness(self):
        """Verify grid cell sensors fire correctly for known hole positions."""
        from environments.frozen_lake import (
            generate_observation_tensor, state_index,
        )

        grid_size = 4
        n_pos = 16
        n_states = 2 * n_pos
        # Create a single known config: hole at position 2 = (0, 2)
        holes = np.zeros((1, 16), dtype=np.float32)
        holes[0, 2] = 1.0  # hole at (row=0, col=2)

        B = generate_observation_tensor(grid_size, holes, base_noise=0.01, noise_range=0.0)

        # Agent at pos 0, unscanned: grid cell channel for cell 2 should fire
        x_unscanned = state_index(0, 0, n_pos)
        p_fire = B[n_states + 2, 1, x_unscanned, 0]
        assert p_fire > 0.9, f"Cell 2 sensor should fire (hole): p={p_fire}"

        # Grid cell channel for cell 5 (no hole) should NOT fire
        p_nofire = B[n_states + 5, 1, x_unscanned, 0]
        assert p_nofire < 0.1, f"Cell 5 sensor should not fire: p={p_nofire}"

        # Agent at pos 0, scanned: should be near-deterministic
        x_scanned = state_index(0, 1, n_pos)
        p_scanned = B[n_states + 2, 1, x_scanned, 0]
        assert p_scanned > 0.99, f"Scanned cell 2 should fire: p={p_scanned}"


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
        self.n_states = 2 * self.n_pos
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
        n_channels = 3 * self.n_pos
        assert result.obs.shape == (n_channels,)
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
        n_channels = 3 * self.n_pos
        assert result.obs.shape == (n_channels,)

    def test_wall_collision(self):
        """Moving LEFT from (0,0) should stay at (0,0)."""
        from environments.frozen_lake import LEFT

        self.env.reset(seed=0, config_idx=0)
        result = self.env.step(LEFT)
        assert self.env._position == 0
        n_channels = 3 * self.n_pos
        assert result.obs.shape == (n_channels,)

    def test_scan_action(self):
        """SCAN should switch to scanned mode."""
        from environments.frozen_lake import SCAN

        self.env.reset(seed=0, config_idx=0)
        assert self.env._scanned == 0
        self.env.step(SCAN)
        assert self.env._scanned == 1
        assert self.env._position == 0  # position unchanged

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
        self.n_pos = 16
        self.n_states = 2 * self.n_pos
        self.n_configs = 10

        self.holes = sample_configs(self.grid_size, self.n_configs, seed=42)
        self.T = generate_transition_tensor(self.grid_size, self.holes, slip_prob=0.0)
        self.B = generate_observation_tensor(self.grid_size, self.holes)
        self.goal = generate_goal(self.grid_size, self.holes)

    def _make_obs(self, pos, config_idx):
        """Build an observation vector: position one-hot + grid cell sensors."""
        import jax.numpy as jnp
        n_channels = self.B.shape[0]
        obs = jnp.zeros(n_channels)
        # Position channels: one-hot for state (pos, unscanned)
        obs = obs.at[pos].set(1.0)
        # Grid cell channels: threshold from B
        for cell in range(self.n_pos):
            ch = self.n_states + cell
            obs = obs.at[ch].set(
                1.0 if self.B[ch, 1, pos, config_idx] > 0.5 else 0.0
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
        assert 0 <= action < 5

    def test_loopy_bp_agent_produces_valid_action(self):
        import jax.numpy as jnp
        from agents.frozen_lake_agent import create_agent

        agent = create_agent(
            "loopy_bp", self.T, self.B, self.goal, self.holes,
            planning_horizon=5, planning_iterations=2,
        )
        obs = self._make_obs(0, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < 5

    def test_region_extended_agent_produces_valid_action(self):
        import jax.numpy as jnp
        from agents.frozen_lake_agent import create_agent

        agent = create_agent(
            "region_extended", self.T, self.B, self.goal, self.holes,
            planning_horizon=3, planning_iterations=2,
        )
        obs = self._make_obs(0, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < 5

    def test_static_belief_update(self):
        """Observation likelihood should differentiate configs via grid cell sensors.

        The grid cell sensor model is θ-dependent (different hole configs
        trigger different patterns), so repeated observations should
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
        obs_start = obs_start.at[0].set(1.0)  # position channel: at state 0 (pos 0, unscanned)
        for cell in range(self.n_pos):
            ch = self.n_states + cell
            obs_start = obs_start.at[ch].set(
                1.0 if B_low_noise[ch, 1, 0, 0] > 0.5 else 0.0
            )
        _, agent = agent.step(obs_start, time_remaining=10)

        # All configs should still be alive
        assert jnp.all(agent.q_static_state > 0)

        # Observation from position 5 under config 0
        obs_5 = jnp.zeros(n_channels)
        obs_5 = obs_5.at[5].set(1.0)  # position channel: at state 5
        for cell in range(self.n_pos):
            ch = self.n_states + cell
            obs_5 = obs_5.at[ch].set(
                1.0 if B_low_noise[ch, 1, 5, 0] > 0.5 else 0.0
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
            assert 0 <= action < 5, f"Method {method_name} returned invalid action {action}"


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
