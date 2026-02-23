"""Tests for RockSample environment and agents."""

import numpy as np


# ---------------------------------------------------------------------------
# Environment / tensor generation tests
# ---------------------------------------------------------------------------

class TestRockSampleTensors:
    """Validate transition and observation tensor properties."""

    def setup_method(self):
        from environments.rocksample import (
            sample_rock_positions,
            all_quality_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
            N_ACTIONS,
        )

        self.grid_size = 4
        self.n_rocks = 2
        self.n_pos = self.grid_size * self.grid_size
        self.n_collect = 2 ** self.n_rocks
        self.n_scan = 2 ** self.n_rocks
        self.n_states = self.n_pos * self.n_collect * self.n_scan
        self.n_configs = 2 ** self.n_rocks  # exhaustive
        self.n_actions = N_ACTIONS

        # Start position: middle-left
        self.start_pos = (self.grid_size // 2) * self.grid_size  # row=2, col=0

        self.rock_positions = sample_rock_positions(
            self.grid_size, self.n_rocks, seed=42,
        )
        self.qualities = all_quality_configs(self.n_rocks)
        self.T = generate_transition_tensor(
            self.grid_size, self.rock_positions, self.n_rocks,
            slip_prob=0.0,
        )
        self.T_slip = generate_transition_tensor(
            self.grid_size, self.rock_positions, self.n_rocks,
            slip_prob=0.2,
        )
        self.B = generate_observation_tensor(
            self.grid_size, self.rock_positions, self.qualities,
            self.n_rocks, half_eff_dist=2.0, pos_noise=0.1,
        )
        self.goal = generate_goal(
            self.grid_size, self.rock_positions, self.qualities,
            self.n_rocks,
        )

    def test_config_shapes(self):
        assert self.rock_positions.shape == (self.n_rocks,)
        assert self.qualities.shape == (self.n_configs, self.n_rocks)

    def test_exhaustive_configs(self):
        """All 2^k quality configurations should be present."""
        assert self.qualities.shape[0] == 2 ** self.n_rocks
        # Each row should be unique
        rows = set(tuple(r) for r in self.qualities)
        assert len(rows) == 2 ** self.n_rocks

    def test_rock_quality_independence(self):
        """Each rock should be good in exactly half of all configs."""
        for j in range(self.n_rocks):
            n_good = int(self.qualities[:, j].sum())
            assert n_good == self.n_configs // 2, (
                f"Rock {j}: {n_good} good configs, expected {self.n_configs // 2}"
            )

    def test_rocks_not_at_start_or_exit(self):
        """Rocks should not be at start or exit positions."""
        from environments.rocksample import is_exit

        for rp in self.rock_positions:
            assert int(rp) != self.start_pos
            assert not is_exit(int(rp), self.grid_size)

    def test_state_indexing_roundtrip(self):
        """state_index -> unpack_state should be a roundtrip."""
        from environments.rocksample import state_index, unpack_state

        for pos in range(self.n_pos):
            for coll in range(self.n_collect):
                for scanned in range(self.n_scan):
                    x = state_index(pos, coll, scanned, self.n_pos, self.n_collect, self.n_scan)
                    p2, c2, s2 = unpack_state(x, self.n_pos, self.n_collect, self.n_scan)
                    assert (p2, c2, s2) == (pos, coll, scanned)

    def test_transition_shape(self):
        assert self.T.shape == (self.n_states, self.n_states, self.n_configs, self.n_actions)

    def test_transition_stochastic(self):
        """T should sum to 1 over x_new for each (x_old, theta, action)."""
        sums = self.T.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-6), f"Max deviation: {np.abs(sums - 1.0).max()}"

    def test_exit_absorbing(self):
        """Exit positions should be absorbing for all actions and all theta."""
        from environments.rocksample import state_index, is_exit

        for pos in range(self.n_pos):
            if not is_exit(pos, self.grid_size):
                continue
            for coll in range(self.n_collect):
                for scanned in range(self.n_scan):
                    x = state_index(pos, coll, scanned, self.n_pos, self.n_collect, self.n_scan)
                    for theta in range(self.n_configs):
                        for a in range(self.n_actions):
                            assert np.isclose(self.T[x, x, theta, a], 1.0), (
                                f"Exit pos={pos}, coll={coll}, scanned={scanned}, "
                                f"theta={theta}, a={a}: T[x,x]={self.T[x, x, theta, a]}"
                            )

    def test_scan_transitions(self):
        """SCAN should set the nearest unscanned rock's bit."""
        from environments.rocksample import (
            state_index, is_exit, nearest_unscanned_rock, SCAN,
        )

        for pos in range(self.n_pos):
            if is_exit(pos, self.grid_size):
                continue
            for coll in range(self.n_collect):
                for scanned in range(self.n_scan):
                    x_old = state_index(pos, coll, scanned, self.n_pos, self.n_collect, self.n_scan)
                    j = nearest_unscanned_rock(pos, scanned, self.rock_positions, self.grid_size)
                    if j >= 0:
                        scanned_new = scanned | (1 << j)
                        x_new = state_index(pos, coll, scanned_new, self.n_pos, self.n_collect, self.n_scan)
                    else:
                        x_new = x_old  # all scanned: self-loop
                    assert np.isclose(self.T[x_new, x_old, 0, SCAN], 1.0), (
                        f"SCAN at pos={pos}, coll={coll}, scanned={scanned}: "
                        f"expected T[{x_new},{x_old}]=1.0, got {self.T[x_new, x_old, 0, SCAN]}"
                    )

    def test_scan_nearest_rock(self):
        """SCAN should target the nearest unscanned rock."""
        from environments.rocksample import (
            nearest_unscanned_rock, euclidean_distance,
        )

        # Place agent at a position and verify nearest-unscanned logic
        pos = self.start_pos
        scanned_mask = 0  # nothing scanned

        j = nearest_unscanned_rock(pos, scanned_mask, self.rock_positions, self.grid_size)
        assert j >= 0, "Should find an unscanned rock"

        # Verify it's actually the nearest
        d_chosen = euclidean_distance(pos, int(self.rock_positions[j]), self.grid_size)
        for k in range(self.n_rocks):
            if k == j:
                continue
            d_other = euclidean_distance(pos, int(self.rock_positions[k]), self.grid_size)
            assert d_chosen <= d_other, (
                f"Rock {j} (d={d_chosen}) should be nearest, but rock {k} (d={d_other}) is closer"
            )

        # After scanning rock j, next scan should target a different rock
        scanned_mask = 1 << j
        j2 = nearest_unscanned_rock(pos, scanned_mask, self.rock_positions, self.grid_size)
        assert j2 >= 0 and j2 != j, "Should find a different unscanned rock"

    def test_scan_all_scanned_selfloop(self):
        """SCAN when all rocks are scanned should self-loop."""
        from environments.rocksample import (
            state_index, nearest_unscanned_rock, SCAN,
        )

        pos = self.start_pos
        coll = 0
        all_scanned = self.n_scan - 1  # all bits set

        j = nearest_unscanned_rock(pos, all_scanned, self.rock_positions, self.grid_size)
        assert j == -1, "No unscanned rocks should remain"

        x = state_index(pos, coll, all_scanned, self.n_pos, self.n_collect, self.n_scan)
        assert np.isclose(self.T[x, x, 0, SCAN], 1.0), "SCAN should self-loop when all scanned"

    def test_sample_at_rock(self):
        """SAMPLE at an uncollected rock should set the collected bit."""
        from environments.rocksample import state_index, SAMPLE

        for j, rp in enumerate(self.rock_positions):
            rp = int(rp)
            coll_before = 0  # nothing collected
            x_old = state_index(rp, coll_before, 0, self.n_pos, self.n_collect, self.n_scan)
            coll_after = coll_before | (1 << j)
            x_new = state_index(rp, coll_after, 0, self.n_pos, self.n_collect, self.n_scan)
            assert np.isclose(self.T[x_new, x_old, 0, SAMPLE], 1.0), (
                f"SAMPLE at rock {j} (pos={rp}) should collect it"
            )

    def test_sample_no_rock(self):
        """SAMPLE not at a rock should self-loop."""
        from environments.rocksample import state_index, is_exit, SAMPLE

        rock_set = set(int(rp) for rp in self.rock_positions)
        for pos in range(self.n_pos):
            if pos in rock_set or is_exit(pos, self.grid_size):
                continue
            x = state_index(pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)
            assert np.isclose(self.T[x, x, 0, SAMPLE], 1.0), (
                f"SAMPLE at pos={pos} (no rock) should self-loop"
            )
            break  # one check is enough

    def test_sample_already_collected(self):
        """SAMPLE at an already-collected rock should self-loop."""
        from environments.rocksample import state_index, SAMPLE

        rp = int(self.rock_positions[0])
        coll = 1 << 0  # rock 0 already collected
        x = state_index(rp, coll, 0, self.n_pos, self.n_collect, self.n_scan)
        assert np.isclose(self.T[x, x, 0, SAMPLE], 1.0)

    def test_theta_independent_transitions(self):
        """Transitions should be identical across all theta."""
        for theta in range(1, self.n_configs):
            assert np.allclose(self.T[:, :, 0, :], self.T[:, :, theta, :])

    def test_observation_shape(self):
        n_channels = self.n_pos + self.n_rocks
        assert self.B.shape == (n_channels, 2, self.n_states, self.n_configs)

    def test_observation_is_stochastic(self):
        """B should sum to 1 over obs_types and all values in [0, 1]."""
        sums = self.B.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)
        assert np.all(self.B >= 0.0)
        assert np.all(self.B <= 1.0)

    def test_position_channels_correctness(self):
        """Position channel p should fire high at pos=p, low elsewhere.

        Position channels always use fixed noise regardless of scanned_mask.
        """
        from environments.rocksample import unpack_state

        for target_pos in range(self.n_pos):
            ch = target_pos
            for x in range(self.n_states):
                pos, _, _ = unpack_state(x, self.n_pos, self.n_collect, self.n_scan)
                if pos == target_pos:
                    assert self.B[ch, 1, x, 0] > 0.5
                else:
                    assert self.B[ch, 0, x, 0] > 0.5
            # theta-independent
            for theta in range(1, self.n_configs):
                assert np.allclose(self.B[ch, :, :, 0], self.B[ch, :, :, theta])

    def test_position_channels_scan_independent(self):
        """Position channels should be the same regardless of scanned_mask."""
        from environments.rocksample import state_index

        for target_pos in range(self.n_pos):
            ch = target_pos
            # Compare scanned_mask=0 vs scanned_mask=all
            x_unscanned = state_index(target_pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)
            x_scanned = state_index(target_pos, 0, self.n_scan - 1, self.n_pos, self.n_collect, self.n_scan)
            assert np.isclose(
                self.B[ch, 1, x_unscanned, 0],
                self.B[ch, 1, x_scanned, 0],
            ), "Position channels should not depend on scanned_mask"

    def test_rock_quality_distance_dependent(self):
        """Rock observation accuracy should decrease with distance."""
        from environments.rocksample import (
            state_index, euclidean_distance,
        )

        # Pick first rock that is good in config 0
        for j in range(self.n_rocks):
            if self.qualities[0, j] == 1.0:
                rock_pos = int(self.rock_positions[j])
                ch = self.n_pos + j

                # Find a near and far position
                near_pos = rock_pos  # distance 0
                far_pos = 0 if rock_pos != 0 else self.n_pos - 1

                d_near = euclidean_distance(near_pos, rock_pos, self.grid_size)
                d_far = euclidean_distance(far_pos, rock_pos, self.grid_size)
                if d_near >= d_far:
                    continue

                x_near = state_index(near_pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)
                x_far = state_index(far_pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)

                p_near = self.B[ch, 1, x_near, 0]
                p_far = self.B[ch, 1, x_far, 0]
                assert p_near > p_far, (
                    f"Near obs (d={d_near}) p={p_near} should exceed "
                    f"far obs (d={d_far}) p={p_far}"
                )
                return

        # If no good rock in config 0, check bad rock (reversed logic)
        for j in range(self.n_rocks):
            if self.qualities[0, j] == 0.0:
                rock_pos = int(self.rock_positions[j])
                ch = self.n_pos + j

                near_pos = rock_pos
                far_pos = 0 if rock_pos != 0 else self.n_pos - 1

                x_near = state_index(near_pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)
                x_far = state_index(far_pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)

                p_near = self.B[ch, 1, x_near, 0]
                p_far = self.B[ch, 1, x_far, 0]
                assert p_near < p_far, (
                    f"Bad rock: near p(good)={p_near} should be less than "
                    f"far p(good)={p_far}"
                )
                return

    def test_rock_quality_scanned_deterministic(self):
        """When a rock's bit is set in scanned_mask, its channel should be near-deterministic."""
        from environments.rocksample import state_index

        for theta in range(self.n_configs):
            for j in range(self.n_rocks):
                ch = self.n_pos + j
                rock_good = self.qualities[theta, j] == 1.0
                scanned_mask = 1 << j  # only this rock scanned
                for pos in range(self.n_pos):
                    x_scanned = state_index(pos, 0, scanned_mask, self.n_pos, self.n_collect, self.n_scan)
                    if rock_good:
                        assert self.B[ch, 1, x_scanned, theta] > 0.99
                    else:
                        assert self.B[ch, 0, x_scanned, theta] > 0.99

    def test_rock_quality_unscanned_not_deterministic(self):
        """When rock j is NOT scanned, its channel should be distance-dependent (not near-deterministic)."""
        from environments.rocksample import state_index

        # scanned_mask=0: no rocks scanned. For a rock far from the agent,
        # the observation should not be near-deterministic.
        for j in range(self.n_rocks):
            ch = self.n_pos + j
            rock_pos = int(self.rock_positions[j])
            # Pick a far position
            far_pos = 0 if rock_pos != 0 else self.n_pos - 1
            x = state_index(far_pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)
            p = self.B[ch, 1, x, 0]
            # Should be substantially less than near-deterministic
            assert p < 0.95, (
                f"Unscanned rock {j} at far pos should not be near-deterministic, got p={p}"
            )

    def test_goal_shape(self):
        assert self.goal.shape == (self.n_states, self.n_configs)
        for theta in range(self.n_configs):
            assert np.isclose(self.goal[:, theta].sum(), 1.0, atol=1e-6)

    def test_slip_stochastic(self):
        """With slip, transitions should still sum to 1."""
        sums = self.T_slip.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

class TestRockSampleEnv:

    def setup_method(self):
        from environments.rocksample import (
            sample_rock_positions,
            all_quality_configs,
            generate_observation_tensor,
            RockSampleEnv,
            rc_to_pos,
        )

        self.grid_size = 4
        self.n_rocks = 2
        self.n_pos = self.grid_size * self.grid_size
        self.n_collect = 2 ** self.n_rocks
        self.n_scan = 2 ** self.n_rocks
        self.start_pos = rc_to_pos(self.grid_size // 2, 0, self.grid_size)

        self.rock_positions = sample_rock_positions(
            self.grid_size, self.n_rocks, seed=42,
        )
        self.qualities = all_quality_configs(self.n_rocks)
        self.obs_tensor = generate_observation_tensor(
            self.grid_size, self.rock_positions, self.qualities,
            self.n_rocks, half_eff_dist=2.0, pos_noise=0.1,
        )
        self.env = RockSampleEnv(
            grid_size=self.grid_size,
            rock_positions=self.rock_positions,
            qualities=self.qualities,
            n_rocks=self.n_rocks,
            obs_tensor=self.obs_tensor,
            slip_prob=0.0,
            max_steps=50,
        )

    def test_reset(self):
        result = self.env.reset(seed=0)
        n_channels = self.n_pos + self.n_rocks
        assert result.obs.shape == (n_channels,)
        assert all(v in (0.0, 1.0) for v in result.obs)
        assert not result.terminated
        assert not result.truncated
        assert self.env._position == self.start_pos

    def test_movement_right(self):
        """Moving RIGHT from start should increase column."""
        from environments.rocksample import RIGHT, pos_to_rc

        self.env.reset(seed=0, config_idx=0)
        self.env.step(RIGHT)
        _, col = pos_to_rc(self.env._position, self.grid_size)
        assert col == 1

    def test_wall_collision(self):
        """Moving LEFT from start (col=0) should stay in place."""
        from environments.rocksample import LEFT

        self.env.reset(seed=0, config_idx=0)
        self.env.step(LEFT)
        assert self.env._position == self.start_pos

    def test_scan_action(self):
        """SCAN should set the nearest unscanned rock's bit."""
        from environments.rocksample import SCAN, nearest_unscanned_rock

        self.env.reset(seed=0, config_idx=0)
        assert self.env._scanned_mask == 0

        # Find which rock should be scanned first
        j = nearest_unscanned_rock(
            self.env._position, 0, self.rock_positions, self.grid_size,
        )
        self.env.step(SCAN)
        assert self.env._scanned_mask == (1 << j), (
            f"Expected scanned_mask={1 << j}, got {self.env._scanned_mask}"
        )
        assert self.env._position == self.start_pos  # position unchanged

    def test_scan_second_rock(self):
        """Second SCAN should target the next nearest unscanned rock."""
        from environments.rocksample import SCAN, nearest_unscanned_rock

        self.env.reset(seed=0, config_idx=0)

        # First scan
        j1 = nearest_unscanned_rock(
            self.env._position, 0, self.rock_positions, self.grid_size,
        )
        self.env.step(SCAN)
        assert self.env._scanned_mask == (1 << j1)

        # Second scan
        j2 = nearest_unscanned_rock(
            self.env._position, self.env._scanned_mask,
            self.rock_positions, self.grid_size,
        )
        self.env.step(SCAN)
        assert self.env._scanned_mask == (1 << j1) | (1 << j2)
        assert j1 != j2

    def test_scan_all_scanned_noop(self):
        """SCAN when all rocks scanned should be a no-op."""
        from environments.rocksample import SCAN

        self.env.reset(seed=0, config_idx=0)
        # Scan all rocks
        for _ in range(self.n_rocks):
            self.env.step(SCAN)

        all_scanned = self.env._scanned_mask
        assert all_scanned == self.n_scan - 1  # all bits set

        # Additional scan should be no-op
        self.env.step(SCAN)
        assert self.env._scanned_mask == all_scanned

    def test_sample_at_rock(self):
        """Navigate to a rock and SAMPLE it."""
        from environments.rocksample import (
            SAMPLE, LEFT, RIGHT, UP, DOWN, pos_to_rc,
        )

        self.env.reset(seed=0, config_idx=0)

        rock_pos = int(self.rock_positions[0])
        rock_r, rock_c = pos_to_rc(rock_pos, self.grid_size)
        start_r, start_c = pos_to_rc(self.start_pos, self.grid_size)

        dr = rock_r - start_r
        dc = rock_c - start_c
        for _ in range(abs(dc)):
            self.env.step(RIGHT if dc > 0 else LEFT)
        for _ in range(abs(dr)):
            self.env.step(DOWN if dr > 0 else UP)

        assert self.env._position == rock_pos
        assert self.env._collected == 0

        result = self.env.step(SAMPLE)
        assert self.env._collected & (1 << 0)  # rock 0 collected
        assert result.reward != 0  # should get reward or penalty

    def test_exit_termination(self):
        """Moving to exit column should terminate with exit_reward."""
        from environments.rocksample import RIGHT

        self.env.reset(seed=0, config_idx=0)
        for _ in range(self.grid_size):
            result = self.env.step(RIGHT)
            if result.terminated:
                break
        assert result.terminated
        assert result.reward >= self.env.exit_reward

    def test_ascii_render(self):
        self.env.reset(seed=0, config_idx=0)
        ascii_str = self.env.render_ascii()
        assert "A" in ascii_str
        assert "E" in ascii_str
        assert "scanned=" in ascii_str


# ---------------------------------------------------------------------------
# Agent / planning integration tests
# ---------------------------------------------------------------------------

class TestRockSampleAgents:
    """Test that agents can plan and act on RockSample."""

    def setup_method(self):
        import jax.numpy as jnp
        from environments.rocksample import (
            sample_rock_positions,
            all_quality_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
            state_index,
            rc_to_pos,
            N_ACTIONS,
        )

        self.grid_size = 3
        self.n_rocks = 2
        self.n_pos = self.grid_size * self.grid_size
        self.n_collect = 2 ** self.n_rocks
        self.n_scan = 2 ** self.n_rocks
        self.n_states = self.n_pos * self.n_collect * self.n_scan
        self.n_configs = 2 ** self.n_rocks
        self.n_actions = N_ACTIONS
        self.n_channels = self.n_pos + self.n_rocks
        self.start_pos = rc_to_pos(self.grid_size // 2, 0, self.grid_size)
        self.start_state_idx = state_index(self.start_pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)

        self.rock_positions = sample_rock_positions(
            self.grid_size, self.n_rocks, seed=42,
        )
        self.qualities = all_quality_configs(self.n_rocks)
        self.T = generate_transition_tensor(
            self.grid_size, self.rock_positions, self.n_rocks,
            slip_prob=0.0,
        )
        self.B = generate_observation_tensor(
            self.grid_size, self.rock_positions, self.qualities,
            self.n_rocks, half_eff_dist=2.0, pos_noise=0.1,
        )
        self.goal = generate_goal(
            self.grid_size, self.rock_positions, self.qualities,
            self.n_rocks,
        )

    def _make_obs(self, pos, config_idx):
        """Build an observation vector for position."""
        import jax.numpy as jnp
        from environments.rocksample import state_index

        obs = jnp.zeros(self.n_channels)
        x = state_index(pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)
        for ch in range(self.n_channels):
            obs = obs.at[ch].set(
                1.0 if self.B[ch, 1, x, config_idx] > 0.5 else 0.0
            )
        return obs

    def test_bp_agent_produces_valid_action(self):
        from agents.rocksample_agent import create_agent

        agent = create_agent(
            "bp", self.T, self.B, self.goal,
            self.rock_positions, self.qualities,
            self.n_pos, self.start_state_idx,
            planning_horizon=3,
        )
        obs = self._make_obs(self.start_pos, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < self.n_actions

    def test_loopy_bp_agent_produces_valid_action(self):
        from agents.rocksample_agent import create_agent

        agent = create_agent(
            "loopy_bp", self.T, self.B, self.goal,
            self.rock_positions, self.qualities,
            self.n_pos, self.start_state_idx,
            planning_horizon=3, planning_iterations=2,
        )
        obs = self._make_obs(self.start_pos, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < self.n_actions

    def test_region_extended_agent_produces_valid_action(self):
        from agents.rocksample_agent import create_agent

        agent = create_agent(
            "region_extended", self.T, self.B, self.goal,
            self.rock_positions, self.qualities,
            self.n_pos, self.start_state_idx,
            planning_horizon=3, planning_iterations=2,
        )
        obs = self._make_obs(self.start_pos, 0)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < self.n_actions

    def test_static_belief_update(self):
        """Observation likelihood should shift static belief toward true config."""
        import jax.numpy as jnp
        from agents.rocksample_agent import create_agent
        from environments.rocksample import (
            generate_observation_tensor, state_index,
        )

        B_low = generate_observation_tensor(
            self.grid_size, self.rock_positions, self.qualities,
            self.n_rocks, half_eff_dist=0.5, pos_noise=0.01,
        )

        agent = create_agent(
            "bp", self.T, B_low, self.goal,
            self.rock_positions, self.qualities,
            self.n_pos, self.start_state_idx,
            planning_horizon=3,
        )
        agent = agent.reset()

        n_channels = B_low.shape[0]
        x_start = state_index(self.start_pos, 0, 0, self.n_pos, self.n_collect, self.n_scan)
        obs = jnp.zeros(n_channels)
        for ch in range(n_channels):
            obs = obs.at[ch].set(
                1.0 if B_low[ch, 1, x_start, 0] > 0.5 else 0.0
            )

        _, agent = agent.step(obs, time_remaining=10)
        assert jnp.all(agent.q_static_state > 0)

        prior_uniform = 1.0 / self.n_configs
        assert float(agent.q_static_state[0]) > prior_uniform * 0.5

    def test_all_methods_run(self):
        """Smoke test: all agent methods produce a valid action."""
        from agents.rocksample_agent import AGENT_CLASSES, create_agent

        obs = self._make_obs(self.start_pos, 0)

        for method_name in AGENT_CLASSES:
            agent = create_agent(
                method_name, self.T, self.B, self.goal,
                self.rock_positions, self.qualities,
                self.n_pos, self.start_state_idx,
                planning_horizon=3, planning_iterations=2,
            )
            action, _ = agent.step(obs, time_remaining=5)
            assert 0 <= action < self.n_actions, (
                f"Method {method_name} returned invalid action {action}"
            )


# ---------------------------------------------------------------------------
# Episode integration test
# ---------------------------------------------------------------------------

class TestRockSampleEpisode:

    def test_episode_completes(self):
        import jax.numpy as jnp
        from environments.rocksample import (
            sample_rock_positions,
            all_quality_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
            state_index,
            rc_to_pos,
            RockSampleEnv,
        )
        from agents.rocksample_agent import create_agent

        grid_size = 3
        n_rocks = 2
        n_pos = grid_size * grid_size
        n_collect = 2 ** n_rocks
        n_scan = 2 ** n_rocks
        start_pos = rc_to_pos(grid_size // 2, 0, grid_size)
        start_state_idx = state_index(start_pos, 0, 0, n_pos, n_collect, n_scan)

        rock_positions = sample_rock_positions(grid_size, n_rocks, seed=123)
        qualities = all_quality_configs(n_rocks)
        T = generate_transition_tensor(grid_size, rock_positions, n_rocks)
        B = generate_observation_tensor(
            grid_size, rock_positions, qualities, n_rocks,
        )
        goal = generate_goal(grid_size, rock_positions, qualities, n_rocks)

        agent = create_agent(
            "bp", T, B, goal,
            rock_positions, qualities, n_pos, start_state_idx,
            planning_horizon=5, planning_iterations=1,
        )
        env = RockSampleEnv(
            grid_size=grid_size,
            rock_positions=rock_positions,
            qualities=qualities,
            n_rocks=n_rocks,
            obs_tensor=B,
            max_steps=20,
        )

        result = env.reset(seed=0)
        agent = agent.reset()

        steps = 0
        while not result.terminated and not result.truncated and steps < 20:
            obs = jnp.array(result.obs)
            action, agent = agent.step(obs, time_remaining=20 - steps)
            result = env.step(action)
            steps += 1

        assert result.terminated or result.truncated or steps >= 20
