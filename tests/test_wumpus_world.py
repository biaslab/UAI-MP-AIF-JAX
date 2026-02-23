"""Tests for Wumpus World environment and agents."""

import numpy as np


# ---------------------------------------------------------------------------
# Environment / tensor generation tests
# ---------------------------------------------------------------------------

class TestWumpusWorldTensors:
    """Validate transition and observation tensor properties."""

    def setup_method(self):
        from environments.wumpus_world import (
            sample_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
            N_ACTIONS,
        )

        self.grid_size = 4
        self.n_pos = self.grid_size * self.grid_size
        self.n_states = 2 * self.n_pos  # doubled for scan mode
        self.n_configs = 10
        self.n_actions = N_ACTIONS
        self.start_pos = 0

        self.pits, self.wumpus, self.gold = sample_configs(
            self.grid_size, self.n_configs, n_pits=2, seed=42,
        )
        self.T = generate_transition_tensor(
            self.grid_size, self.pits, self.wumpus, slip_prob=0.0,
        )
        self.obs_noise = 0.1
        self.B = generate_observation_tensor(
            self.grid_size, self.pits, self.wumpus, self.gold,
            obs_noise=self.obs_noise,
        )
        self.goal = generate_goal(self.grid_size, self.pits, self.wumpus, self.gold)

    def test_config_shapes(self):
        assert self.pits.shape == (self.n_configs, self.n_pos)
        assert self.wumpus.shape == (self.n_configs, self.n_pos)
        assert self.gold.shape == (self.n_configs, self.n_pos)

    def test_start_always_safe(self):
        """Start position should never have pit, wumpus, or gold."""
        for theta in range(self.n_configs):
            assert self.pits[theta, self.start_pos] == 0.0
            assert self.wumpus[theta, self.start_pos] == 0.0
            assert self.gold[theta, self.start_pos] == 0.0

    def test_no_overlap(self):
        """Pits, wumpus, and gold should not overlap within a config."""
        for theta in range(self.n_configs):
            for pos in range(self.n_pos):
                occupied = (
                    self.pits[theta, pos]
                    + self.wumpus[theta, pos]
                    + self.gold[theta, pos]
                )
                assert occupied <= 1.0, f"Overlap at θ={theta}, pos={pos}"

    def test_transition_shape(self):
        assert self.T.shape == (self.n_states, self.n_states, self.n_configs, self.n_actions)

    def test_transition_stochastic(self):
        sums = self.T.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_pits_absorbing(self):
        from environments.wumpus_world import state_index
        for theta in range(self.n_configs):
            for pos in range(self.n_pos):
                if self.pits[theta, pos] == 1.0:
                    for scanned in range(2):
                        x = state_index(pos, scanned, self.n_pos)
                        for a in range(self.n_actions):
                            assert np.isclose(self.T[x, x, theta, a], 1.0)

    def test_wumpus_absorbing(self):
        from environments.wumpus_world import state_index
        for theta in range(self.n_configs):
            for pos in range(self.n_pos):
                if self.wumpus[theta, pos] == 1.0:
                    for scanned in range(2):
                        x = state_index(pos, scanned, self.n_pos)
                        for a in range(self.n_actions):
                            assert np.isclose(self.T[x, x, theta, a], 1.0)

    def test_observation_shape(self):
        n_pos = self.grid_size * self.grid_size
        n_doubled = 2 * n_pos
        n_channels = 3 + n_pos
        assert self.B.shape == (n_channels, 2, n_doubled, self.n_configs)

    def test_observation_is_stochastic(self):
        """Each channel should sum to 1 over obs_types but NOT be one-hot."""
        n_channels = self.B.shape[0]
        n_states = self.B.shape[2]
        has_non_onehot = False
        for c in range(n_channels):
            for x in range(n_states):
                for theta in range(self.n_configs):
                    probs = self.B[c, :, x, theta]
                    # Must sum to 1
                    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
                    # Values must be valid probabilities
                    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
                    # Check if this entry is NOT one-hot (i.e. neither value is exactly 0 or 1)
                    if not np.isclose(probs.max(), 1.0, atol=1e-6):
                        has_non_onehot = True
        assert has_non_onehot, "Observation tensor should not be fully deterministic (one-hot)"

    def test_breeze_correctness(self):
        """Breeze probability should be high when adjacent to a pit, low otherwise."""
        from environments.wumpus_world import get_neighbors, state_index, BREEZE

        for theta in range(self.n_configs):
            for pos in range(self.n_pos):
                neighbors = get_neighbors(pos, self.grid_size)
                has_pit_neighbor = any(self.pits[theta, n] == 1.0 for n in neighbors)
                for scanned in range(2):
                    x = state_index(pos, scanned, self.n_pos)
                    if has_pit_neighbor:
                        assert self.B[BREEZE, 1, x, theta] > 0.5, (
                            f"P(breeze=1) should be high near pit at pos={pos}, θ={theta}"
                        )
                    else:
                        assert self.B[BREEZE, 0, x, theta] > 0.5, (
                            f"P(breeze=0) should be high away from pit at pos={pos}, θ={theta}"
                        )

    def test_stench_correctness(self):
        """Stench probability should be high when adjacent to wumpus, low otherwise."""
        from environments.wumpus_world import get_neighbors, state_index, STENCH

        for theta in range(self.n_configs):
            for pos in range(self.n_pos):
                neighbors = get_neighbors(pos, self.grid_size)
                has_wumpus_neighbor = any(self.wumpus[theta, n] == 1.0 for n in neighbors)
                for scanned in range(2):
                    x = state_index(pos, scanned, self.n_pos)
                    if has_wumpus_neighbor:
                        assert self.B[STENCH, 1, x, theta] > 0.5, (
                            f"P(stench=1) should be high near wumpus at pos={pos}, θ={theta}"
                        )
                    else:
                        assert self.B[STENCH, 0, x, theta] > 0.5, (
                            f"P(stench=0) should be high away from wumpus at pos={pos}, θ={theta}"
                        )

    def test_glitter_correctness(self):
        """Glitter probability should be high on gold cell, low otherwise."""
        from environments.wumpus_world import state_index, GLITTER

        for theta in range(self.n_configs):
            for pos in range(self.n_pos):
                for scanned in range(2):
                    x = state_index(pos, scanned, self.n_pos)
                    if self.gold[theta, pos] == 1.0:
                        assert self.B[GLITTER, 1, x, theta] > 0.5, (
                            f"P(glitter=1) should be high on gold at pos={pos}, θ={theta}"
                        )
                    else:
                        assert self.B[GLITTER, 0, x, theta] > 0.5, (
                            f"P(glitter=0) should be high off gold at pos={pos}, θ={theta}"
                        )

    def test_position_channel_correctness(self):
        """Position channel c should fire with high prob at position c, low elsewhere."""
        from environments.wumpus_world import state_index, N_FEATURE_CHANNELS

        for target_pos in range(self.n_pos):
            ch = N_FEATURE_CHANNELS + target_pos
            for pos in range(self.n_pos):
                for scanned in range(2):
                    x = state_index(pos, scanned, self.n_pos)
                    # θ-independent: check first config
                    if pos == target_pos:
                        assert self.B[ch, 1, x, 0] > 0.5, (
                            f"Position channel {target_pos} should fire at pos={pos}"
                        )
                    else:
                        assert self.B[ch, 0, x, 0] > 0.5, (
                            f"Position channel {target_pos} should NOT fire at pos={pos}"
                        )
            # θ-independent: all configs should be identical
            for theta in range(1, self.n_configs):
                assert np.allclose(self.B[ch, :, :, 0], self.B[ch, :, :, theta])

    def test_goal_shape(self):
        assert self.goal.shape == (self.n_states, self.n_configs)
        # Each config column should sum to 1 (softmax)
        for theta in range(self.n_configs):
            assert np.isclose(self.goal[:, theta].sum(), 1.0)


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

class TestWumpusWorldEnv:

    def setup_method(self):
        from environments.wumpus_world import (
            sample_configs,
            generate_observation_tensor,
            WumpusWorldEnv,
        )

        self.grid_size = 4
        self.pits, self.wumpus, self.gold = sample_configs(
            self.grid_size, 10, seed=42,
        )
        self.obs_tensor = generate_observation_tensor(
            self.grid_size, self.pits, self.wumpus, self.gold, obs_noise=0.1,
        )
        self.env = WumpusWorldEnv(
            grid_size=self.grid_size,
            pits=self.pits,
            wumpus=self.wumpus,
            gold=self.gold,
            obs_tensor=self.obs_tensor,
            slip_prob=0.0,
            max_steps=50,
        )

    def test_reset(self):
        result = self.env.reset(seed=0)
        n_pos = self.grid_size * self.grid_size
        assert result.obs.shape == (3 + n_pos,)
        assert not result.terminated
        assert not result.truncated

    def test_obs_binary(self):
        """Sampled observations should still be binary (0 or 1) even though the model is stochastic."""
        result = self.env.reset(seed=0)
        for val in result.obs:
            assert val == 0.0 or val == 1.0

    def test_ascii_render(self):
        self.env.reset(seed=0, config_idx=0)
        ascii_str = self.env.render_ascii()
        assert "A" in ascii_str


# ---------------------------------------------------------------------------
# Agent / planning integration tests
# ---------------------------------------------------------------------------

class TestWumpusAgents:
    """Test that agents can plan and act on Wumpus World."""

    def setup_method(self):
        import jax.numpy as jnp
        from environments.wumpus_world import (
            sample_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
            N_ACTIONS,
        )

        self.grid_size = 4
        self.n_pos = self.grid_size * self.grid_size
        self.n_configs = 10
        self.n_actions = N_ACTIONS

        self.pits, self.wumpus, self.gold = sample_configs(
            self.grid_size, self.n_configs, seed=42,
        )
        self.T = generate_transition_tensor(
            self.grid_size, self.pits, self.wumpus,
        )
        self.B = generate_observation_tensor(
            self.grid_size, self.pits, self.wumpus, self.gold,
            obs_noise=0.1,
        )
        self.goal = generate_goal(self.grid_size, self.pits, self.wumpus, self.gold)
        self.n_channels = self.B.shape[0]

    def _zero_obs(self):
        import jax.numpy as jnp
        return jnp.zeros(self.n_channels)

    def test_bp_agent_produces_valid_action(self):
        from agents.wumpus_agent import create_agent

        agent = create_agent(
            "bp", self.T, self.B, self.goal,
            planning_horizon=3,
        )
        action, agent = agent.step(self._zero_obs(), time_remaining=10)
        assert 0 <= action < self.n_actions

    def test_loopy_bp_agent_produces_valid_action(self):
        from agents.wumpus_agent import create_agent

        agent = create_agent(
            "loopy_bp", self.T, self.B, self.goal,
            planning_horizon=3, planning_iterations=2,
        )
        action, agent = agent.step(self._zero_obs(), time_remaining=10)
        assert 0 <= action < self.n_actions

    def test_region_extended_agent_produces_valid_action(self):
        from agents.wumpus_agent import create_agent

        agent = create_agent(
            "region_extended", self.T, self.B, self.goal,
            planning_horizon=3, planning_iterations=2,
        )
        action, agent = agent.step(self._zero_obs(), time_remaining=10)
        assert 0 <= action < self.n_actions

    def test_all_methods_run(self):
        """Smoke test: all agent methods produce a valid action."""
        from agents.wumpus_agent import AGENT_CLASSES, create_agent

        obs = self._zero_obs()

        for method_name in AGENT_CLASSES:
            agent = create_agent(
                method_name, self.T, self.B, self.goal,
                planning_horizon=3, planning_iterations=2,
            )
            action, _ = agent.step(obs, time_remaining=5)
            assert 0 <= action < self.n_actions, f"Method {method_name} returned invalid action {action}"


# ---------------------------------------------------------------------------
# Episode integration test
# ---------------------------------------------------------------------------

class TestWumpusEpisode:

    def test_episode_completes(self):
        import jax.numpy as jnp
        from environments.wumpus_world import (
            sample_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
            WumpusWorldEnv,
        )
        from agents.wumpus_agent import create_agent

        grid_size = 4
        n_configs = 20
        pits, wumpus, gold = sample_configs(grid_size, n_configs, n_pits=1, seed=123)
        T = generate_transition_tensor(grid_size, pits, wumpus)
        B = generate_observation_tensor(grid_size, pits, wumpus, gold, obs_noise=0.1)
        goal = generate_goal(grid_size, pits, wumpus, gold)

        agent = create_agent(
            "bp", T, B, goal,
            planning_horizon=5, planning_iterations=1,
        )
        env = WumpusWorldEnv(
            grid_size=grid_size, pits=pits, wumpus=wumpus, gold=gold,
            obs_tensor=B, max_steps=30,
        )

        result = env.reset(seed=0)
        agent = agent.reset()

        steps = 0
        while not result.terminated and not result.truncated and steps < 30:
            obs = jnp.array(result.obs)
            action, agent = agent.step(obs, time_remaining=30 - steps)
            result = env.step(action)
            steps += 1

        assert result.terminated or result.truncated or steps >= 30
