#!/usr/bin/env python
"""Single-episode diagnostic script for RockSample with full internal state output."""

import sys
from pathlib import Path
import argparse
import time
import random

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
from environments.rocksample import (
    sample_rock_positions,
    all_quality_configs,
    generate_transition_tensor,
    generate_observation_tensor,
    generate_goal,
    RockSampleEnv,
    pos_to_rc,
    rc_to_pos,
    state_index,
    unpack_state,
    nearest_unscanned_rock,
    euclidean_distance,
    is_exit,
    N_ACTIONS,
)
from agents.rocksample_agent import create_agent

ACTION_NAMES = ["left", "down", "right", "up", "scan", "sample"]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def entropy(p):
    """Shannon entropy in bits."""
    p = jnp.clip(p, 1e-12, 1.0)
    return -float(jnp.sum(p * jnp.log2(p)))


def print_position_grid(belief, grid_size, n_pos, n_collect, n_scan, label="Position belief"):
    """Print a 2D grid of position probabilities (marginalised over collected & scanned)."""
    pos_marginal = np.zeros(n_pos)
    n_states = n_pos * n_collect * n_scan
    for x in range(n_states):
        pos, _, _ = unpack_state(x, n_pos, n_collect, n_scan)
        pos_marginal[pos] += float(belief[x])

    print(f"    {label} (row\\col):")
    header = "        " + "  ".join(f"c={c:>2}" for c in range(grid_size))
    print(header)
    for r in range(grid_size):
        row = f"  r={r:>2}  "
        for c in range(grid_size):
            pos = r * grid_size + c
            p = pos_marginal[pos]
            if p < 0.0005:
                row += "   .  "
            else:
                row += f"{p:5.3f} "
        print(row)


def print_action_distribution(action_dist):
    """Print actions with bars."""
    bar_width = 40
    for i, name in enumerate(ACTION_NAMES):
        if i >= len(action_dist):
            break
        p = float(action_dist[i])
        n_bars = int(p * bar_width)
        bar = "#" * n_bars
        print(f"      {name:>6s}: {p:.4f}  {bar}")


def print_obs(obs, n_pos, n_rocks, rock_positions, grid_size):
    """Print position + rock quality sensor readings."""
    # Position modality: first n_pos channels
    pos_obs = obs[:n_pos]
    observed_pos = int(jnp.argmax(pos_obs))
    r, c = pos_to_rc(observed_pos, grid_size)
    print(f"    Position sensors → best match: ({r},{c})")

    # Rock quality channels
    for j in range(n_rocks):
        rp = int(rock_positions[j])
        rr, rc_ = pos_to_rc(rp, grid_size)
        val = float(obs[n_pos + j])
        quality_str = "GOOD" if val > 0.5 else "BAD"
        print(f"    Rock {j} at ({rr},{rc_}): sensor={val:.0f} ({quality_str})")


def print_rock_quality_beliefs(q_static, qualities, n_rocks):
    """Print marginal P(rock j is good) under current θ belief."""
    n_configs = q_static.shape[0]
    print("    Rock quality beliefs (marginal over θ):")
    for j in range(n_rocks):
        p_good = 0.0
        for theta in range(n_configs):
            if qualities[theta, j] == 1.0:
                p_good += float(q_static[theta])
        bar_width = 30
        n_bars = int(p_good * bar_width)
        bar = "#" * n_bars + "." * (bar_width - n_bars)
        print(f"      Rock {j}: P(good)={p_good:.4f}  [{bar}]")


def print_scanned_collected_belief(belief, n_pos, n_collect, n_scan, n_rocks):
    """Print marginal belief over scanned_mask and collected_mask."""
    n_states = n_pos * n_collect * n_scan
    scanned_marginal = np.zeros(n_scan)
    collected_marginal = np.zeros(n_collect)
    for x in range(n_states):
        _, coll, scanned = unpack_state(x, n_pos, n_collect, n_scan)
        scanned_marginal[scanned] += float(belief[x])
        collected_marginal[coll] += float(belief[x])

    # Top scanned masks
    top_scanned = np.argsort(-scanned_marginal)[:5]
    print("    Top scanned_mask beliefs:")
    for sm in top_scanned:
        p = scanned_marginal[sm]
        if p < 0.001:
            break
        bits = bin(sm)[2:].zfill(n_rocks)
        print(f"      scanned={bits}: {p:.4f}")

    # Top collected masks
    top_collected = np.argsort(-collected_marginal)[:5]
    print("    Top collected_mask beliefs:")
    for cm in top_collected:
        p = collected_marginal[cm]
        if p < 0.001:
            break
        bits = bin(cm)[2:].zfill(n_rocks)
        print(f"      collected={bits}: {p:.4f}")


def print_static_summary(q_static, qualities, n_rocks, top_k=5):
    """Print top-k most likely quality configurations."""
    n_static = q_static.shape[0]
    order = jnp.argsort(-q_static)
    print(f"    Top-{min(top_k, n_static)} quality configs:")
    for rank in range(min(top_k, n_static)):
        idx = int(order[rank])
        p = float(q_static[idx])
        if p < 0.001:
            break
        rock_quals = []
        for j in range(n_rocks):
            rock_quals.append("G" if qualities[idx, j] == 1.0 else "B")
        qual_str = " ".join(rock_quals)
        print(f"      θ={idx:>3d}: p={p:.4f}  rocks=[{qual_str}]")


def print_goal_diagnostic(goal, grid_size, rock_positions, qualities, n_rocks):
    """Print goal vector values for key positions."""
    n_pos = grid_size * grid_size
    n_collect = 2 ** n_rocks
    n_scan = 2 ** n_rocks
    n_states = n_pos * n_collect * n_scan
    n_static = goal.shape[1]

    goal_avg = np.array(goal).mean(axis=1)

    print("  [GOAL VECTOR DIAGNOSTIC]")

    # Exit positions (rightmost column)
    print("    Exit column goal values (averaged over θ, collected=0, scanned=0):")
    for r in range(grid_size):
        pos = rc_to_pos(r, grid_size - 1, grid_size)
        x = state_index(pos, 0, 0, n_pos, n_collect, n_scan)
        val = float(goal_avg[x])
        print(f"      Exit ({r},{grid_size-1}): avg_goal={val:.6f}")

    # Exit with all good rocks collected
    print("    Exit with all-good collected (θ=all_good, scanned=0):")
    all_good_theta = n_collect - 1  # all bits set = all good
    all_collected = n_collect - 1
    for r in range(grid_size):
        pos = rc_to_pos(r, grid_size - 1, grid_size)
        x = state_index(pos, all_collected, 0, n_pos, n_collect, n_scan)
        val = float(goal[x, all_good_theta])
        print(f"      Exit ({r},{grid_size-1}) all_collected: goal={val:.6f}")

    # Rock positions (no collection)
    print("    Rock positions (collected=0, scanned=0, avg over θ):")
    for j in range(n_rocks):
        rp = int(rock_positions[j])
        rr, rc_ = pos_to_rc(rp, grid_size)
        x = state_index(rp, 0, 0, n_pos, n_collect, n_scan)
        val = float(goal_avg[x])
        print(f"      Rock {j} ({rr},{rc_}): avg_goal={val:.6f}")

    print()


def print_distance_table(agent_pos, rock_positions, grid_size, n_rocks, scanned_mask):
    """Print distance to each rock and scan status."""
    print("    Rock distances & status:")
    for j in range(n_rocks):
        rp = int(rock_positions[j])
        rr, rc_ = pos_to_rc(rp, grid_size)
        d = euclidean_distance(agent_pos, rp, grid_size)
        scanned = "SCANNED" if scanned_mask & (1 << j) else "unscanned"
        p_correct = 0.5 + 0.5 * (2.0 ** (-d / 2.0))  # default half_eff_dist
        print(f"      Rock {j} at ({rr},{rc_}): dist={d:.2f}  P(correct)={p_correct:.3f}  {scanned}")


# ---------------------------------------------------------------------------
# Diagnostic episode
# ---------------------------------------------------------------------------


def run_diagnostic_episode(agent, env, args, rock_positions, qualities, compare_bp_agent=None):
    grid_size = args.grid_size
    n_pos = grid_size * grid_size
    n_rocks = args.n_rocks
    n_collect = 2 ** n_rocks
    n_scan = 2 ** n_rocks
    n_states = n_pos * n_collect * n_scan
    n_static = qualities.shape[0]
    max_steps = env.max_steps

    max_entropy_x = jnp.log2(float(n_states))
    max_entropy_static = jnp.log2(float(n_static))

    result = env.reset(seed=args.seed)
    agent = agent.reset()
    if compare_bp_agent is not None:
        compare_bp_agent = compare_bp_agent.reset()
    theta = env.config_idx

    print("=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    print()
    print(f"  True config index: {theta}")
    print(f"  True rock qualities: {['G' if qualities[theta, j] == 1.0 else 'B' for j in range(n_rocks)]}")
    print(f"  True world:")
    for line in env.render_ascii().split("\n"):
        print(f"    {line}")
    print()
    print(f"  State entropy: {entropy(agent.q_current_state):.2f} bits (max={max_entropy_x:.2f})")
    print(f"  Static entropy: {entropy(agent.q_static_state):.2f} bits (max={max_entropy_static:.2f})")
    print()

    total_reward = 0.0
    step_num = 0

    while True:
        print("=" * 70)
        print(f"STEP {step_num}")
        print("=" * 70)
        print()

        # --- TRUE STATE ---
        true_r, true_c = pos_to_rc(env._position, grid_size)
        scan_str = bin(env._scanned_mask)[2:].zfill(n_rocks)
        coll_str = bin(env._collected)[2:].zfill(n_rocks)
        print(f"  [TRUE STATE] pos=({true_r},{true_c})  scanned={scan_str}  collected={coll_str}  config={theta}")
        print(f"  True world:")
        for line in env.render_ascii().split("\n"):
            print(f"    {line}")
        print()

        # --- DISTANCES ---
        print("  [ROCK DISTANCES]")
        print_distance_table(env._position, rock_positions, grid_size, n_rocks, env._scanned_mask)
        nearest_j = nearest_unscanned_rock(env._position, env._scanned_mask, rock_positions, grid_size)
        if nearest_j >= 0:
            nr, nc = pos_to_rc(int(rock_positions[nearest_j]), grid_size)
            print(f"    SCAN would target: Rock {nearest_j} at ({nr},{nc})")
        else:
            print(f"    SCAN would target: none (all scanned)")
        print()

        # --- OBSERVATION ---
        obs = jnp.array(result.obs)
        print("  [OBSERVATION]")
        print_obs(obs, n_pos, n_rocks, rock_positions, grid_size)
        print()

        # --- STATE INFERENCE + PLANNING ---
        print("  [INFERENCE + PLANNING]")
        if args.receding_horizon:
            time_remaining = max_steps - step_num
        else:
            time_remaining = agent.planning_horizon

        t0 = time.time()
        action, agent = agent.step(obs, time_remaining)
        elapsed_ms = (time.time() - t0) * 1000
        print(f"    Step time: {elapsed_ms:.1f}ms")
        print()

        # --- POSITION BELIEF ---
        print("  [POSITION BELIEF]")
        q_pos = agent.q_current_state
        print_position_grid(q_pos, grid_size, n_pos, n_collect, n_scan)

        # MAP over position marginal
        pos_marginal = np.zeros(n_pos)
        for x in range(n_states):
            pos, _, _ = unpack_state(x, n_pos, n_collect, n_scan)
            pos_marginal[pos] += float(q_pos[x])
        map_pos = int(np.argmax(pos_marginal))
        map_r, map_c = pos_to_rc(map_pos, grid_size)
        map_p = float(pos_marginal[map_pos])
        correct = (map_pos == env._position)
        print(f"    MAP position: ({map_r},{map_c}) p={map_p:.4f} {'CORRECT' if correct else 'WRONG'}")
        print(f"    State entropy: {entropy(q_pos):.2f} bits (max={max_entropy_x:.2f})")
        print()

        # --- SCANNED / COLLECTED BELIEF ---
        print("  [SCANNED & COLLECTED BELIEF]")
        print_scanned_collected_belief(q_pos, n_pos, n_collect, n_scan, n_rocks)
        print()

        # --- STATIC BELIEF ---
        print("  [STATIC BELIEF (θ = rock quality config)]")
        q_static = agent.q_static_state
        print_static_summary(q_static, qualities, n_rocks)

        true_config_p = float(q_static[theta])
        true_config_rank = int((q_static > q_static[theta]).sum())
        print(f"    True config θ={theta}: p={true_config_p:.4f} (rank {true_config_rank + 1}/{n_static})")
        print(f"    Static entropy: {entropy(q_static):.2f} bits (max={max_entropy_static:.2f})")
        print()

        # --- ROCK QUALITY BELIEFS ---
        print("  [ROCK QUALITY BELIEFS]")
        print_rock_quality_beliefs(q_static, qualities, n_rocks)
        print()

        # --- PLANNING DEBUG ---
        print("  [PLANNING DEBUG]")
        horizon = min(time_remaining, agent.planning_horizon)
        action_dist = agent._plan(agent.q_current_state, agent.q_static_state, horizon)
        print(f"    Action distribution (horizon={horizon}):")
        print_action_distribution(action_dist)
        print()

        # --- COMPARE BP ---
        if compare_bp_agent is not None:
            print("  [COMPARE BP]")
            _, compare_bp_agent = compare_bp_agent.step(obs, time_remaining)
            bp_action_dist = compare_bp_agent._plan(
                compare_bp_agent.q_current_state,
                compare_bp_agent.q_static_state,
                horizon,
            )
            bp_action = int(jnp.argmax(bp_action_dist))
            print(f"    BP action distribution:")
            print_action_distribution(bp_action_dist)
            print(f"    BP chosen action: {ACTION_NAMES[bp_action]}")
            print(f"    Primary chosen action: {ACTION_NAMES[action]}")
            if bp_action != action:
                print(f"    >>> MISMATCH: BP says {ACTION_NAMES[bp_action]}, primary says {ACTION_NAMES[action]}")
            print()

        # --- ACTION ---
        print(f"  [ACTION] {ACTION_NAMES[action]}")
        print()

        # --- EXECUTE ---
        result = env.step(action)
        total_reward += result.reward

        reward_str = f"reward={result.reward:+.1f}" if result.reward != 0 else "reward=0"
        print(f"  [RESULT] {reward_str}  terminated={result.terminated}  truncated={result.truncated}")
        if result.terminated:
            if result.reward > 0:
                print("  >>> REACHED EXIT!")
            else:
                print("  >>> EXIT (no extra reward)")
        if action == 5 and result.reward != 0:  # SAMPLE
            if result.reward > 0:
                print(f"  >>> SAMPLED GOOD ROCK! (+{result.reward:.1f})")
            else:
                print(f"  >>> SAMPLED BAD ROCK! ({result.reward:.1f})")
        print()

        step_num += 1
        if result.terminated or result.truncated:
            break

    # --- EPISODE SUMMARY ---
    print("=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)
    print(f"  Success: {result.terminated and total_reward > 0}")
    print(f"  Steps: {step_num}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Final state entropy: {entropy(agent.q_current_state):.2f} bits")
    print(f"  Final static entropy: {entropy(agent.q_static_state):.2f} bits")
    print(f"  Final true config rank: {int((agent.q_static_state > agent.q_static_state[theta]).sum()) + 1}/{n_static}")

    # Final rock quality beliefs
    print("  Final rock quality beliefs:")
    for j in range(n_rocks):
        p_good = sum(
            float(agent.q_static_state[t])
            for t in range(n_static)
            if qualities[t, j] == 1.0
        )
        true_q = "GOOD" if qualities[theta, j] == 1.0 else "BAD"
        print(f"    Rock {j}: P(good)={p_good:.4f}  (true: {true_q})")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="RockSample single-episode diagnostic")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--n-rocks", type=int, default=3)
    parser.add_argument("--half-eff-dist", type=float, default=2.0)
    parser.add_argument("--pos-noise", type=float, default=0.1)
    parser.add_argument("--slip-prob", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--planning-horizon", type=int, default=10)
    parser.add_argument("--planning-iterations", type=int, default=3)
    parser.add_argument("--planning-method", type=str, default="loopy",
                        choices=["loopy-vbp", "loopy", "region-extended",
                                 "dyn-channel", "nuijten", "vbp-channel",
                                 "precise-info-seeking"])
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--good-reward", type=float, default=10.0)
    parser.add_argument("--bad-penalty", type=float, default=10.0)
    parser.add_argument("--exit-reward", type=float, default=10.0)
    parser.add_argument("--goal-temperature", type=float, default=1.0)
    parser.add_argument("--scan-cost", type=float, default=0.5)
    parser.add_argument("--sample-cost", type=float, default=0.5)
    parser.add_argument("--receding-horizon", action="store_true")
    parser.add_argument("--terminal-goal-only", action="store_true",
                        help="Apply goal only at final planning step")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    grid_size = args.grid_size
    n_rocks = args.n_rocks
    n_pos = grid_size * grid_size
    n_collect = 2 ** n_rocks
    n_scan = 2 ** n_rocks
    n_states = n_pos * n_collect * n_scan
    n_configs = n_collect

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print()
    print(f"RockSample[{grid_size},{n_rocks}]")
    print(f"  States: {n_states} = {n_pos} pos x {n_collect} collected x {n_scan} scanned")
    print(f"  Configs (θ): {n_configs}")
    print(f"  Half-eff dist: {args.half_eff_dist}, pos noise: {args.pos_noise}")
    print(f"  Slip prob: {args.slip_prob}")
    print(f"  Rewards: good={args.good_reward}, bad_penalty={args.bad_penalty}, exit={args.exit_reward}")
    print(f"  Goal temperature: {args.goal_temperature}")
    print(f"  Action costs: scan={args.scan_cost}, sample={args.sample_cost}")
    print(f"  Method: {args.planning_method}")
    print(f"  Horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    print(f"  Iterations: {args.planning_iterations}")
    if args.damping < 1.0:
        print(f"  Damping: {args.damping}")
    if args.terminal_goal_only:
        print(f"  Terminal goal only: enabled")
    print(f"  Seed: {args.seed}")
    print()

    print("Generating tensors...")
    t0 = time.time()

    start_pos = rc_to_pos(grid_size // 2, 0, grid_size)
    start_state_idx = state_index(start_pos, 0, 0, n_pos, n_collect, n_scan)

    rock_positions = sample_rock_positions(grid_size, n_rocks, seed=args.seed)
    qualities = all_quality_configs(n_rocks)
    T = generate_transition_tensor(grid_size, rock_positions, n_rocks, slip_prob=args.slip_prob)
    B = generate_observation_tensor(
        grid_size, rock_positions, qualities, n_rocks,
        half_eff_dist=args.half_eff_dist, pos_noise=args.pos_noise,
    )
    goal = generate_goal(
        grid_size, rock_positions, qualities, n_rocks,
        exit_reward=args.exit_reward, good_reward=args.good_reward,
        bad_penalty=args.bad_penalty, temperature=args.goal_temperature,
    )

    print(f"  Rock positions: {rock_positions.tolist()}")
    for j in range(n_rocks):
        rr, rc_ = pos_to_rc(int(rock_positions[j]), grid_size)
        print(f"    Rock {j}: ({rr},{rc_})")
    print(f"  T: {T.shape}  B: {B.shape}  goal: {goal.shape}")
    print(f"  Generated in {time.time() - t0:.2f}s")
    print()

    # --- Goal diagnostic ---
    print_goal_diagnostic(goal, grid_size, rock_positions, qualities, n_rocks)

    METHOD_MAP = {
        "loopy-vbp": "loopy_vbp",
        "loopy": "loopy_bp",
        "region-extended": "region_extended",
        "dyn-channel": "dyn_channel",
        "nuijten": "nuijten",
        "vbp-channel": "vbp_channel",
        "precise-info-seeking": "precise_info_seeking",
    }
    method_key = METHOD_MAP[args.planning_method]

    # Action prior: [1, 1, 1, 1, scan_cost, sample_cost] normalized
    action_prior = np.array(
        [1.0, 1.0, 1.0, 1.0, args.scan_cost, args.sample_cost],
        dtype=np.float32,
    )
    action_prior = action_prior / action_prior.sum()

    agent = create_agent(
        method_key, T, B, goal,
        rock_positions, qualities, n_pos, start_state_idx,
        planning_horizon=args.planning_horizon,
        planning_iterations=args.planning_iterations,
        action_prior=action_prior,
        damping=args.damping,
        terminal_goal_only=args.terminal_goal_only,
    )

    compare_bp_agent = None

    env = RockSampleEnv(
        grid_size=grid_size,
        rock_positions=rock_positions,
        qualities=qualities,
        n_rocks=n_rocks,
        obs_tensor=B,
        slip_prob=args.slip_prob,
        max_steps=args.max_steps,
        good_reward=args.good_reward,
        bad_penalty=args.bad_penalty,
        exit_reward=args.exit_reward,
    )

    run_diagnostic_episode(agent, env, args, rock_positions, qualities,
                           compare_bp_agent=compare_bp_agent)


if __name__ == "__main__":
    main()
