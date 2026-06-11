"""Tests for the canonical RockSample environment and agents."""

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
            n_actions_for,
            n_events_for,
            is_exit,
        )

        self.grid_size = 4
        self.n_rocks = 2
        self.n_pos = self.grid_size * self.grid_size
        self.n_mask = 2 ** self.n_rocks
        self.n_events = n_events_for(self.n_rocks)
        self.n_states = self.n_pos * self.n_mask * self.n_events
        self.n_configs = 2 ** self.n_rocks
        self.n_actions = n_actions_for(self.n_rocks)

        self.rock_positions = sample_rock_positions(
            self.grid_size, self.n_rocks, seed=42,
        )
        self.qualities = all_quality_configs(self.n_rocks)
        self.T = generate_transition_tensor(
            self.grid_size, self.rock_positions, self.n_rocks, slip_prob=0.0,
        )
        self.T_slip = generate_transition_tensor(
            self.grid_size, self.rock_positions, self.n_rocks, slip_prob=0.2,
        )
        self.B = generate_observation_tensor(
            self.grid_size, self.rock_positions, self.qualities,
            self.n_rocks, half_eff_dist=2.0, pos_noise=0.1,
        )
        self.goal = generate_goal(
            self.grid_size, self.rock_positions, self.qualities, self.n_rocks,
        )
        self._is_exit = is_exit

    # --- configs / indexing -------------------------------------------------

    def test_config_shapes(self):
        assert self.qualities.shape == (self.n_configs, self.n_rocks)

    def test_exhaustive_configs(self):
        """All 2^k quality assignments should appear exactly once."""
        seen = set()
        for theta in range(self.n_configs):
            key = tuple(int(q) for q in self.qualities[theta])
            seen.add(key)
        assert len(seen) == self.n_configs

    def test_rocks_not_at_start_or_exit(self):
        from environments.rocksample import rc_to_pos, is_exit

        start_pos = rc_to_pos(self.grid_size // 2, 0, self.grid_size)
        for rp in self.rock_positions:
            assert int(rp) != start_pos
            assert not is_exit(int(rp), self.grid_size)

    def test_state_indexing_roundtrip(self):
        from environments.rocksample import state_index, unpack_state

        for pos in range(self.n_pos):
            for mask in range(self.n_mask):
                for event in range(self.n_events):
                    x = state_index(pos, mask, event,
                                    self.n_pos, self.n_mask, self.n_events)
                    p, m, e = unpack_state(x, self.n_pos, self.n_mask, self.n_events)
                    assert (p, m, e) == (pos, mask, event)

    def test_chebyshev_distance(self):
        from environments.rocksample import chebyshev_distance, rc_to_pos

        a = rc_to_pos(0, 0, self.grid_size)
        b = rc_to_pos(2, 3, self.grid_size)
        assert chebyshev_distance(a, b, self.grid_size) == 3
        assert chebyshev_distance(a, a, self.grid_size) == 0

    # --- transitions ----------------------------------------------------------

    def test_transition_shape(self):
        assert self.T.shape == (self.n_states, self.n_states, self.n_configs, self.n_actions)

    def test_transition_stochastic(self):
        """Every (x_old, θ, action) column must sum to 1 — no zero columns."""
        sums = self.T.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-6), f"Max deviation: {np.abs(sums - 1.0).max()}"

    def test_exit_absorbing(self):
        from environments.rocksample import state_index, is_exit

        for pos in range(self.n_pos):
            if not is_exit(pos, self.grid_size):
                continue
            for mask in range(self.n_mask):
                for event in range(self.n_events):
                    x = state_index(pos, mask, event,
                                    self.n_pos, self.n_mask, self.n_events)
                    for a in range(self.n_actions):
                        assert np.isclose(self.T[x, x, 0, a], 1.0)

    def test_movement_resets_event(self):
        """Movement from any non-exit state must land in event=OTHER."""
        from environments.rocksample import state_index, unpack_state, is_exit

        theta = 0
        for x_old in range(self.n_states):
            pos, mask, event = unpack_state(x_old, self.n_pos, self.n_mask, self.n_events)
            if is_exit(pos, self.grid_size):
                continue
            for a in range(4):
                dest = np.flatnonzero(self.T[:, x_old, theta, a])
                for x_new in dest:
                    _, _, e_new = unpack_state(int(x_new), self.n_pos, self.n_mask, self.n_events)
                    assert e_new == 0, f"Movement should reset event (got {e_new})"

    def test_sense_sets_event(self):
        """SENSE_r keeps pos/mask and sets event=1+r deterministically."""
        from environments.rocksample import (
            state_index, is_exit, sense_action, event_sense,
        )

        theta = 0
        for pos in range(self.n_pos):
            if is_exit(pos, self.grid_size):
                continue
            for mask in range(self.n_mask):
                for event in range(self.n_events):
                    x_old = state_index(pos, mask, event,
                                        self.n_pos, self.n_mask, self.n_events)
                    for r in range(self.n_rocks):
                        x_new = state_index(pos, mask, event_sense(r),
                                            self.n_pos, self.n_mask, self.n_events)
                        assert np.isclose(self.T[x_new, x_old, theta, sense_action(r)], 1.0)

    def test_sample_at_rock(self):
        """SAMPLE at an uncollected rock sets its mask bit and the SAMPLE event."""
        from environments.rocksample import (
            state_index, sample_action, event_sample, EVENT_OTHER,
        )

        theta = 0
        a_sample = sample_action(self.n_rocks)
        rock_pos = int(self.rock_positions[0])
        x_old = state_index(rock_pos, 0, EVENT_OTHER,
                            self.n_pos, self.n_mask, self.n_events)
        x_new = state_index(rock_pos, 1, event_sample(self.n_rocks),
                            self.n_pos, self.n_mask, self.n_events)
        assert np.isclose(self.T[x_new, x_old, theta, a_sample], 1.0)

    def test_sample_no_rock_resets_event(self):
        """SAMPLE away from rocks is a no-op that resets the event to OTHER."""
        from environments.rocksample import (
            state_index, sample_action, event_sense, EVENT_OTHER, is_exit,
        )

        theta = 0
        a_sample = sample_action(self.n_rocks)
        rock_set = {int(rp) for rp in self.rock_positions}
        for pos in range(self.n_pos):
            if is_exit(pos, self.grid_size) or pos in rock_set:
                continue
            x_old = state_index(pos, 0, event_sense(0),
                                self.n_pos, self.n_mask, self.n_events)
            x_new = state_index(pos, 0, EVENT_OTHER,
                                self.n_pos, self.n_mask, self.n_events)
            assert np.isclose(self.T[x_new, x_old, theta, a_sample], 1.0)
            break

    def test_sample_already_collected_noop(self):
        """SAMPLE on an already-collected rock is a no-op (event resets)."""
        from environments.rocksample import (
            state_index, sample_action, EVENT_OTHER,
        )

        theta = 0
        a_sample = sample_action(self.n_rocks)
        rock_pos = int(self.rock_positions[0])
        x_old = state_index(rock_pos, 1, EVENT_OTHER,
                            self.n_pos, self.n_mask, self.n_events)
        assert np.isclose(self.T[x_old, x_old, theta, a_sample], 1.0)

    def test_theta_independent_transitions(self):
        """Dynamics must not depend on rock quality."""
        for theta in range(1, self.n_configs):
            assert np.allclose(self.T[:, :, 0, :], self.T[:, :, theta, :])

    def test_slip_stochastic(self):
        sums = self.T_slip.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-6)

    # --- observations ---------------------------------------------------------

    def test_observation_shape(self):
        n_channels = self.n_pos + self.n_rocks
        assert self.B.shape == (n_channels, 3, self.n_states, self.n_configs)

    def test_observation_is_stochastic(self):
        """B should sum to 1 over the 3 outcomes and contain valid probabilities."""
        sums = self.B.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6), f"Max deviation: {np.abs(sums - 1.0).max()}"
        assert np.all(self.B >= 0.0)
        assert np.all(self.B <= 1.0)

    def test_position_channels_correctness(self):
        from environments.rocksample import state_index, EVENT_OTHER

        for target_pos in range(self.n_pos):
            for pos in range(self.n_pos):
                x = state_index(pos, 0, EVENT_OTHER,
                                self.n_pos, self.n_mask, self.n_events)
                if pos == target_pos:
                    assert self.B[target_pos, 1, x, 0] > 0.5
                else:
                    assert self.B[target_pos, 0, x, 0] > 0.5
                # Position channels never emit NO_INFO
                assert self.B[target_pos, 2, x, 0] == 0.0

    def test_rock_channels_no_info_by_default(self):
        """Rock channels emit NO_INFO unless their trigger event is active."""
        from environments.rocksample import (
            state_index, EVENT_OTHER, ROCK_NO_INFO,
        )

        for j in range(self.n_rocks):
            ch = self.n_pos + j
            for pos in range(self.n_pos):
                x = state_index(pos, 0, EVENT_OTHER,
                                self.n_pos, self.n_mask, self.n_events)
                for theta in range(self.n_configs):
                    assert np.isclose(self.B[ch, ROCK_NO_INFO, x, theta], 1.0)

    def test_sense_other_rock_no_info(self):
        """SENSE_r gives info only about rock r, not other rocks."""
        from environments.rocksample import (
            state_index, event_sense, ROCK_NO_INFO,
        )

        x = state_index(0, 0, event_sense(0),
                        self.n_pos, self.n_mask, self.n_events)
        ch_other = self.n_pos + 1  # rock 1's channel during SENSE_0
        assert np.isclose(self.B[ch_other, ROCK_NO_INFO, x, 0], 1.0)

    def test_sense_accuracy_at_rock_cell(self):
        """SENSE_r from the rock's own cell has maximum accuracy (α(0)=1, clipped)."""
        from environments.rocksample import state_index, event_sense

        j = 0
        ch = self.n_pos + j
        rock_pos = int(self.rock_positions[j])
        x = state_index(rock_pos, 0, event_sense(j),
                        self.n_pos, self.n_mask, self.n_events)
        for theta in range(self.n_configs):
            q = int(self.qualities[theta, j])
            assert self.B[ch, q, x, theta] > 0.95

    def test_sense_accuracy_decays_with_distance(self):
        """SENSE accuracy must decrease with Chebyshev distance."""
        from environments.rocksample import (
            state_index, event_sense, chebyshev_distance,
        )

        j = 0
        ch = self.n_pos + j
        rock_pos = int(self.rock_positions[j])
        theta_good = next(
            t for t in range(self.n_configs) if self.qualities[t, j] == 1.0
        )

        # Collect (distance, accuracy) pairs
        pairs = []
        for pos in range(self.n_pos):
            d = chebyshev_distance(pos, rock_pos, self.grid_size)
            x = state_index(pos, 0, event_sense(j),
                            self.n_pos, self.n_mask, self.n_events)
            pairs.append((d, float(self.B[ch, 1, x, theta_good])))

        d_min = min(p[0] for p in pairs)
        d_max = max(p[0] for p in pairs)
        acc_near = max(acc for d, acc in pairs if d == d_min)
        acc_far = min(acc for d, acc in pairs if d == d_max)
        assert acc_near > acc_far

    def test_sample_reveal(self):
        """SAMPLE event at rock r's cell with bit r set reveals rock r's quality."""
        from environments.rocksample import state_index, event_sample

        j = 0
        ch = self.n_pos + j
        rock_pos = int(self.rock_positions[j])
        x = state_index(rock_pos, 1 << j, event_sample(self.n_rocks),
                        self.n_pos, self.n_mask, self.n_events)
        for theta in range(self.n_configs):
            q = int(self.qualities[theta, j])
            assert self.B[ch, q, x, theta] > 0.99

    # --- goal -------------------------------------------------------------------

    def test_goal_shape(self):
        assert self.goal.shape == (self.n_states, self.n_configs)
        for theta in range(self.n_configs):
            assert np.isclose(self.goal[:, theta].sum(), 1.0, atol=1e-5)

    def test_goal_flat_over_event(self):
        """Goal must not depend on the event component."""
        from environments.rocksample import state_index

        for pos in range(0, self.n_pos, 3):
            for mask in range(self.n_mask):
                x0 = state_index(pos, mask, 0, self.n_pos, self.n_mask, self.n_events)
                for event in range(1, self.n_events):
                    x = state_index(pos, mask, event,
                                    self.n_pos, self.n_mask, self.n_events)
                    assert np.allclose(self.goal[x0, :], self.goal[x, :], atol=1e-7)

    def test_goal_asymmetry_penalizes_blind_sampling(self):
        """Expected logit of collecting a rock under uniform θ must be negative
        (bad_logit > good_logit guards against sampling-for-information)."""
        from environments.rocksample import generate_goal

        good_logit, bad_logit = 2.0, 4.0
        # Under uniform quality belief: E[logit] = 0.5*good - 0.5*bad < 0
        assert 0.5 * good_logit - 0.5 * bad_logit < 0

        # And in the generated tensor: marginal preference of collecting rock 0
        # (uniform over θ) is lower than not collecting it.
        from environments.rocksample import state_index, EVENT_OTHER
        pos = int(self.rock_positions[0])
        x_no = state_index(pos, 0, EVENT_OTHER, self.n_pos, self.n_mask, self.n_events)
        x_yes = state_index(pos, 1, EVENT_OTHER, self.n_pos, self.n_mask, self.n_events)
        pref_no = float(self.goal[x_no, :].mean())
        pref_yes = float(self.goal[x_yes, :].mean())
        assert pref_yes < pref_no


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
        )

        self.grid_size = 4
        self.n_rocks = 2
        self.n_pos = self.grid_size * self.grid_size
        self.rock_positions = sample_rock_positions(self.grid_size, self.n_rocks, seed=42)
        self.qualities = all_quality_configs(self.n_rocks)
        self.obs_tensor = generate_observation_tensor(
            self.grid_size, self.rock_positions, self.qualities, self.n_rocks,
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
        assert not result.terminated
        assert not result.truncated
        assert self.env._event == 0
        assert self.env._mask == 0

    def test_obs_categorical(self):
        """Observations are per-channel outcome indices in {0, 1, 2}."""
        from environments.rocksample import ROCK_NO_INFO

        result = self.env.reset(seed=0)
        for c, val in enumerate(result.obs):
            assert val in (0.0, 1.0, 2.0)
            if c < self.n_pos:
                # Position channels never emit NO_INFO
                assert val in (0.0, 1.0)
        # At reset (event=OTHER), rock channels read NO_INFO
        for j in range(self.n_rocks):
            assert result.obs[self.n_pos + j] == float(ROCK_NO_INFO)

    def test_sense_action(self):
        """SENSE_r sets the event and yields a reading for rock r."""
        from environments.rocksample import sense_action, event_sense

        self.env.reset(seed=0, config_idx=0)
        pos_before = self.env._position
        result = self.env.step(sense_action(1))
        assert self.env._event == event_sense(1)
        assert self.env._position == pos_before
        # Rock 1's channel now reads a quality bit, not NO_INFO
        assert result.obs[self.n_pos + 1] in (0.0, 1.0)
        # Rock 0's channel still NO_INFO
        assert result.obs[self.n_pos + 0] == 2.0

    def test_movement_resets_event(self):
        from environments.rocksample import sense_action, RIGHT, EVENT_OTHER

        self.env.reset(seed=0, config_idx=0)
        self.env.step(sense_action(0))
        assert self.env._event != EVENT_OTHER
        self.env.step(RIGHT)
        assert self.env._event == EVENT_OTHER

    def test_sample_at_rock(self):
        """Walking to a rock and sampling collects it with the right reward."""
        from environments.rocksample import (
            sample_action, event_sample, pos_to_rc, LEFT, DOWN, RIGHT, UP,
        )

        config_idx = 1  # rock 0 good in config 1 (LSB-first encoding)
        self.env.reset(seed=0, config_idx=config_idx)

        # Manually walk to rock 0
        target = int(self.rock_positions[0])
        tr, tc = pos_to_rc(target, self.grid_size)
        while self.env._position != target:
            r, c = pos_to_rc(self.env._position, self.grid_size)
            if c < tc:
                self.env.step(RIGHT)
            elif c > tc:
                self.env.step(LEFT)
            elif r < tr:
                self.env.step(DOWN)
            else:
                self.env.step(UP)
            if self.env._steps > 30:
                break

        if self.env._position == target:
            result = self.env.step(sample_action(self.n_rocks))
            assert self.env._mask & 1
            assert self.env._event == event_sample(self.n_rocks)
            rock_good = self.qualities[config_idx, 0] == 1.0
            if not result.terminated:  # exit reward would confound
                if rock_good:
                    assert result.reward > 0
                else:
                    assert result.reward < 0

    def test_exit_termination(self):
        from environments.rocksample import RIGHT

        self.env.reset(seed=0, config_idx=0)
        result = None
        for _ in range(self.grid_size):
            result = self.env.step(RIGHT)
            if result.terminated:
                break
        assert result.terminated
        assert result.reward > 0  # exit reward

    def test_ascii_render(self):
        self.env.reset(seed=0, config_idx=0)
        ascii_str = self.env.render_ascii()
        assert "A" in ascii_str
        assert "E" in ascii_str


# ---------------------------------------------------------------------------
# Agent / planning integration tests
# ---------------------------------------------------------------------------

class TestRockSampleAgents:
    """Test that agents can plan and act on canonical RockSample."""

    def setup_method(self):
        from environments.rocksample import (
            sample_rock_positions,
            all_quality_configs,
            generate_transition_tensor,
            generate_observation_tensor,
            generate_goal,
            state_index,
            rc_to_pos,
            n_actions_for,
            n_events_for,
            EVENT_OTHER,
        )

        self.grid_size = 3
        self.n_rocks = 2
        self.n_pos = self.grid_size * self.grid_size
        self.n_mask = 2 ** self.n_rocks
        self.n_events = n_events_for(self.n_rocks)
        self.n_states = self.n_pos * self.n_mask * self.n_events
        self.n_configs = 2 ** self.n_rocks
        self.n_actions = n_actions_for(self.n_rocks)
        self.n_channels = self.n_pos + self.n_rocks
        self.start_pos = rc_to_pos(self.grid_size // 2, 0, self.grid_size)
        self.start_state_idx = state_index(
            self.start_pos, 0, EVENT_OTHER, self.n_pos, self.n_mask, self.n_events,
        )

        self.rock_positions = sample_rock_positions(
            self.grid_size, self.n_rocks, seed=42,
        )
        self.qualities = all_quality_configs(self.n_rocks)
        self.T = generate_transition_tensor(
            self.grid_size, self.rock_positions, self.n_rocks, slip_prob=0.0,
        )
        self.B = generate_observation_tensor(
            self.grid_size, self.rock_positions, self.qualities,
            self.n_rocks, half_eff_dist=2.0, pos_noise=0.1,
        )
        self.goal = generate_goal(
            self.grid_size, self.rock_positions, self.qualities, self.n_rocks,
        )

    def _make_obs(self, pos):
        """Build a start-of-episode observation: position one-hot + NO_INFO rocks."""
        import jax.numpy as jnp
        from environments.rocksample import ROCK_NO_INFO

        obs = jnp.zeros(self.n_channels)
        obs = obs.at[pos].set(1.0)
        for j in range(self.n_rocks):
            obs = obs.at[self.n_pos + j].set(float(ROCK_NO_INFO))
        return obs

    def test_bp_agent_produces_valid_action(self):
        from agents.rocksample_agent import create_agent

        agent = create_agent(
            "loopy_bp", self.T, self.B, self.goal,
            self.rock_positions, self.qualities,
            self.n_pos, self.start_state_idx,
            planning_horizon=3,
        )
        obs = self._make_obs(self.start_pos)
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
        obs = self._make_obs(self.start_pos)
        action, agent = agent.step(obs, time_remaining=10)
        assert 0 <= action < self.n_actions

    def test_sense_updates_static_belief(self):
        """A SENSE_r reading must shift P(rock r good) in the right direction."""
        import jax.numpy as jnp
        from agents.rocksample_agent import create_agent
        from environments.rocksample import (
            state_index, sense_action, event_sense, ROCK_NO_INFO,
        )

        agent = create_agent(
            "loopy_bp", self.T, self.B, self.goal,
            self.rock_positions, self.qualities,
            self.n_pos, self.start_state_idx,
            planning_horizon=3,
        )
        agent = agent.reset()

        # First step: start observation (no rock info)
        _, agent = agent.step(self._make_obs(self.start_pos), time_remaining=10)

        p_good_before = sum(
            float(agent.q_static_state[t])
            for t in range(self.n_configs)
            if self.qualities[t, 0] == 1.0
        )

        # Pretend the agent executed SENSE_0 and observed "good" (outcome 1)
        import dataclasses
        agent = dataclasses.replace(agent, last_action=sense_action(0))
        obs = self._make_obs(self.start_pos)
        obs = obs.at[self.n_pos + 0].set(1.0)  # rock 0 reads good
        _, agent = agent.step(obs, time_remaining=9)

        p_good_after = sum(
            float(agent.q_static_state[t])
            for t in range(self.n_configs)
            if self.qualities[t, 0] == 1.0
        )
        assert p_good_after > p_good_before

    def test_all_methods_run(self):
        """Smoke test: all agent methods produce a valid action."""
        from agents.rocksample_agent import AGENT_CLASSES, create_agent

        obs = self._make_obs(self.start_pos)

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
            n_events_for,
            EVENT_OTHER,
            RockSampleEnv,
        )
        from agents.rocksample_agent import create_agent

        grid_size = 3
        n_rocks = 2
        n_pos = grid_size * grid_size
        n_mask = 2 ** n_rocks
        n_events = n_events_for(n_rocks)
        start_pos = rc_to_pos(grid_size // 2, 0, grid_size)
        start_state_idx = state_index(start_pos, 0, EVENT_OTHER, n_pos, n_mask, n_events)

        rock_positions = sample_rock_positions(grid_size, n_rocks, seed=123)
        qualities = all_quality_configs(n_rocks)
        T = generate_transition_tensor(grid_size, rock_positions, n_rocks)
        B = generate_observation_tensor(grid_size, rock_positions, qualities, n_rocks)
        goal = generate_goal(grid_size, rock_positions, qualities, n_rocks)

        agent = create_agent(
            "loopy_bp", T, B, goal,
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
