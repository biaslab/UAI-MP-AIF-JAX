#!/usr/bin/env python
"""Single-episode diagnostic script for Frozen Lake with full internal state output."""

import sys
from pathlib import Path
import argparse
import time
import random

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
from environments.frozen_lake import (
    sample_configs,
    generate_transition_tensor,
    generate_observation_tensor,
    generate_goal,
    FrozenLakeEnv,
    pos_to_rc,
    N_ACTIONS,
    N_SENSOR_CHANNELS,
)
from agents.frozen_lake_agent import create_agent

ACTION_NAMES = ["left", "down", "right", "up"]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def entropy(p):
    """Shannon entropy in bits."""
    p = jnp.clip(p, 1e-12, 1.0)
    return -float(jnp.sum(p * jnp.log2(p)))


def print_position_grid(belief, grid_size, label="Position belief"):
    """Print a 2D grid of position probabilities."""
    pos_marginal = np.array(belief)
    print(f"    {label} (row\\col):")
    header = "        " + "  ".join(f"c={c:>2}" for c in range(grid_size))
    print(header)
    for r in range(grid_size):
        row = f"  r={r:>2}  "
        for c in range(grid_size):
            pos = r * grid_size + c
            p = float(pos_marginal[pos])
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
        print(f"      {name:>5s}: {p:.4f}  {bar}")


def print_obs(obs, grid_size):
    """Print position + neighbor sensor readings."""
    n_pos = grid_size * grid_size

    # Position modality: first n_pos channels
    pos_obs = obs[:n_pos]
    observed_pos = int(jnp.argmax(pos_obs))
    r, c = pos_to_rc(observed_pos, grid_size)
    print(f"    Position: ({r},{c})")

    # Neighbor sensor modality: last 4 channels (LEFT, DOWN, RIGHT, UP)
    sensor_obs = obs[n_pos:]
    readings = "  ".join(
        f"{name}={float(sensor_obs[d]):.0f}"
        for d, name in enumerate(["left", "down", "right", "up"])
    )
    print(f"    Neighbor hole sensors (1=hole detected): {readings}")


def print_hole_heatmap(q_static, holes, grid_size):
    """Print expected hole probability per cell (marginal over configs)."""
    n_pos = grid_size * grid_size
    hole_prob = np.zeros(n_pos)
    for pos in range(n_pos):
        for theta in range(q_static.shape[0]):
            if holes[theta, pos] == 1.0:
                hole_prob[pos] += float(q_static[theta])

    print("    Expected hole heatmap (row\\col):")
    header = "        " + "  ".join(f"c={c:>2}" for c in range(grid_size))
    print(header)
    for r in range(grid_size):
        row = f"  r={r:>2}  "
        for c in range(grid_size):
            pos = r * grid_size + c
            d = hole_prob[pos]
            if d < 0.0005:
                row += "   .  "
            else:
                row += f"{d:5.3f} "
        print(row)


def print_static_summary(q_static, holes, grid_size, top_k=5):
    """Print top-k most likely configurations with their hole positions."""
    n_static = q_static.shape[0]
    order = jnp.argsort(-q_static)
    print(f"    Top-{min(top_k, n_static)} static configs:")
    for rank in range(min(top_k, n_static)):
        idx = int(order[rank])
        p = float(q_static[idx])
        if p < 0.001:
            break
        hole_positions = []
        for pos in range(grid_size * grid_size):
            if holes[idx, pos] == 1.0:
                r, c = pos_to_rc(pos, grid_size)
                hole_positions.append(f"({r},{c})")
        holes_str = " ".join(hole_positions) if hole_positions else "none"
        print(f"      #{idx:>3d}: p={p:.4f}  holes={holes_str}")


def print_goal_diagnostic(goal, grid_size, holes):
    """Print goal vector values for key positions (goal, holes, normal).

    goal is (n_states, n_static) — per-config preference.
    We show the config-averaged values for the initial diagnostic.
    """
    n_pos = grid_size * grid_size
    n_static = goal.shape[1]
    goal_pos = n_pos - 1
    hole_marginal = np.array(holes).mean(axis=0)

    # Average over configs for summary display
    goal_avg = np.array(goal).mean(axis=1)  # (n_states,)

    print("  [GOAL VECTOR DIAGNOSTIC]")

    goal_r, goal_c = pos_to_rc(goal_pos, grid_size)
    print(f"    Goal position ({goal_r},{goal_c}): avg={float(goal_avg[goal_pos]):.6f}")

    # Show per-config variation at goal position
    goal_vals = np.array(goal[goal_pos, :])
    print(f"    Goal pos per-config: min={goal_vals.min():.6f}, max={goal_vals.max():.6f}, std={goal_vals.std():.6f}")

    # Find positions that are holes in most configs
    hole_positions = []
    normal_positions = []
    for pos in range(n_pos):
        if pos == 0 or pos == goal_pos:
            continue
        if hole_marginal[pos] > 0.3:
            hole_positions.append(pos)
        elif hole_marginal[pos] < 0.1:
            normal_positions.append(pos)

    if hole_positions:
        print(f"    Likely hole positions (P(hole) > 0.3):")
        for pos in hole_positions[:5]:
            r, c = pos_to_rc(pos, grid_size)
            val = float(goal_avg[pos])
            print(f"      ({r},{c}): avg_goal={val:.6f}  P(hole)={hole_marginal[pos]:.3f}")

    if normal_positions:
        print(f"    Safe positions (P(hole) < 0.1):")
        for pos in normal_positions[:5]:
            r, c = pos_to_rc(pos, grid_size)
            val = float(goal_avg[pos])
            print(f"      ({r},{c}): avg_goal={val:.6f}  P(hole)={hole_marginal[pos]:.3f}")

    # Summary: ratio between goal, safe, and hole values
    goal_val = float(goal_avg[goal_pos])
    if hole_positions:
        avg_hole = np.mean([float(goal_avg[p]) for p in hole_positions])
    else:
        avg_hole = 0.0
    if normal_positions:
        avg_safe = np.mean([float(goal_avg[p]) for p in normal_positions])
    else:
        avg_safe = 0.0

    print(f"    Summary (config-avg): goal={goal_val:.6f}, avg_safe={avg_safe:.6f}, avg_hole={avg_hole:.6f}")
    if avg_safe > 0:
        print(f"    Ratios: goal/safe={goal_val/avg_safe:.2f}x, safe/hole={avg_safe/max(avg_hole, 1e-12):.2f}x")
    print()


# ---------------------------------------------------------------------------
# Diagnostic episode
# ---------------------------------------------------------------------------


def run_diagnostic_episode(agent, env, args, holes, compare_bp_agent=None):
    grid_size = args.grid_size
    n_pos = grid_size * grid_size
    n_states = n_pos
    n_static = holes.shape[0]
    max_steps = env.max_steps

    max_entropy_pos = jnp.log2(float(n_states))
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
    print(f"  True world:")
    for line in env.render_ascii().split("\n"):
        print(f"    {line}")
    print()
    print(f"  Position entropy: {entropy(agent.q_current_state):.2f} bits (max={max_entropy_pos:.2f})")
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
        print(f"  [TRUE STATE] position=({true_r},{true_c})  config={theta}")
        print(f"  True world:")
        for line in env.render_ascii().split("\n"):
            print(f"    {line}")
        print()

        # --- OBSERVATION ---
        obs = jnp.array(result.obs)
        print("  [OBSERVATION]")
        print_obs(obs, grid_size)
        print()

        # --- STATE INFERENCE (happens inside agent.step) ---
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
        print_position_grid(q_pos, grid_size)

        # MAP position
        pos_marginal = np.array(q_pos)
        map_pos = int(np.argmax(pos_marginal))
        map_r, map_c = pos_to_rc(map_pos, grid_size)
        map_p = float(pos_marginal[map_pos])
        correct = (map_pos == env._position)
        print(f"    MAP position: ({map_r},{map_c}) p={map_p:.4f} {'CORRECT' if correct else 'WRONG'}")
        print(f"    Position entropy: {entropy(q_pos):.2f} bits (max={max_entropy_pos:.2f})")
        print()

        # --- STATIC BELIEF ---
        print("  [STATIC BELIEF]")
        q_static = agent.q_static_state
        print_static_summary(q_static, holes, grid_size)

        true_config_p = float(q_static[theta])
        true_config_rank = int((q_static > q_static[theta]).sum())
        print(f"    True config #{theta}: p={true_config_p:.4f} (rank {true_config_rank + 1}/{n_static})")
        print(f"    Static entropy: {entropy(q_static):.2f} bits (max={max_entropy_static:.2f})")
        print()

        # --- HOLE HEATMAP ---
        print("  [HOLE HEATMAP]")
        print_hole_heatmap(q_static, holes, grid_size)
        print()

        # --- PLANNING DEBUG: full action distribution ---
        print("  [PLANNING DEBUG]")
        horizon = min(time_remaining, agent.planning_horizon)
        action_dist = agent._plan(agent.q_current_state, agent.q_static_state, horizon)
        print(f"    Action distribution (horizon={horizon}):")
        print_action_distribution(action_dist)

        # Goal values at current position, goal, and hole positions
        # goal is (n_states, n_static) — show values for true config theta
        print(f"    Goal value at current pos ({true_r},{true_c}) [config {theta}]: "
              f"{float(agent.goal[env._position, theta]):.6f}")
        print(f"    Goal value at goal pos [config {theta}]: "
              f"{float(agent.goal[n_pos - 1, theta]):.6f}")

        # Show goal values at known hole positions
        hole_config = holes[theta]
        hole_positions = [p for p in range(n_pos) if hole_config[p] == 1.0]
        if hole_positions:
            for hp in hole_positions[:4]:
                hr, hc = pos_to_rc(hp, grid_size)
                hval = float(agent.goal[hp, theta])
                print(f"    Goal value at hole ({hr},{hc}) [config {theta}]: {hval:.6f}")
        print()

        # --- COMPARE BP (if requested) ---
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

        print(f"  [RESULT] reward={result.reward}  terminated={result.terminated}  truncated={result.truncated}")
        if result.terminated:
            if result.reward > 0:
                print("  >>> REACHED GOAL!")
            else:
                print("  >>> FELL IN HOLE!")
        print()

        step_num += 1
        if result.terminated or result.truncated:
            break

    # --- EPISODE SUMMARY ---
    print("=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)
    print(f"  Success: {result.reward > 0}")
    print(f"  Steps: {step_num}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final position entropy: {entropy(agent.q_current_state):.2f} bits")
    print(f"  Final static entropy: {entropy(agent.q_static_state):.2f} bits")
    print(f"  Final true config rank: {int((agent.q_static_state > agent.q_static_state[theta]).sum()) + 1}/{n_static}")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Frozen Lake single-episode diagnostic")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--hole-fraction", type=float, default=0.2)
    parser.add_argument("--min-hamming", type=int, default=0)
    parser.add_argument("--obs-noise", type=float, default=0.15)
    parser.add_argument("--slip-prob", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--planning-horizon", type=int, default=15)
    parser.add_argument("--planning-iterations", type=int, default=3)
    parser.add_argument("--planning-method", type=str, default="loopy",
                        choices=["loopy-vbp", "loopy", "region-extended",
                                 "dyn-channel", "nuijten", "vbp-channel",
                                 "precise-info-seeking", "active-inference"])
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--hole-penalty", type=float, default=1.0)
    parser.add_argument("--goal-temperature", type=float, default=1.0)
    parser.add_argument("--receding-horizon", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    grid_size = args.grid_size
    n_pos = grid_size * grid_size
    n_states = n_pos

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print()
    print(f"Frozen Lake {grid_size}x{grid_size}")
    print(f"  Configs: {args.n_configs}, hole fraction: {args.hole_fraction}")
    print(f"  Obs noise: {args.obs_noise}")
    print(f"  Slip prob: {args.slip_prob}")
    print(f"  Method: {args.planning_method}")
    print(f"  Horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    print(f"  Iterations: {args.planning_iterations}")
    if args.damping < 1.0:
        print(f"  Damping: {args.damping}")
    print(f"  Hole penalty: {args.hole_penalty}, goal temperature: {args.goal_temperature}")
    print(f"  Seed: {args.seed}")
    print(f"  State space: {n_states} states ({n_pos} positions)")
    print()

    print("Generating tensors...")
    t0 = time.time()

    holes = sample_configs(
        grid_size, args.n_configs,
        hole_fraction=args.hole_fraction, seed=args.seed,
        min_hamming=args.min_hamming,
    )
    T = generate_transition_tensor(grid_size, holes, slip_prob=args.slip_prob)
    B = generate_observation_tensor(grid_size, holes, obs_noise=args.obs_noise)
    goal = generate_goal(grid_size, holes, hole_penalty=args.hole_penalty,
                         temperature=args.goal_temperature)

    print(f"  T: {T.shape}  B: {B.shape}  goal: {goal.shape}")
    print(f"  Generated in {time.time() - t0:.2f}s")
    print()

    # --- Goal diagnostic at initialization ---
    print_goal_diagnostic(goal, grid_size, holes)

    METHOD_MAP = {
        "loopy-vbp": "loopy_vbp",
        "loopy": "loopy_bp",
        "region-extended": "region_extended",
        "dyn-channel": "dyn_channel",
        "nuijten": "nuijten",
        "vbp-channel": "vbp_channel",
        "precise-info-seeking": "precise_info_seeking",
        "active-inference": "active_inference",
    }
    method_key = METHOD_MAP[args.planning_method]

    # Uniform prior over the 4 movement actions
    action_prior = None

    agent = create_agent(
        method_key, T, B, goal, holes,
        planning_horizon=args.planning_horizon,
        planning_iterations=args.planning_iterations,
        action_prior=action_prior,
        damping=args.damping,
    )

    compare_bp_agent = None

    env = FrozenLakeEnv(
        grid_size=grid_size, holes=holes, obs_tensor=B,
        slip_prob=args.slip_prob, max_steps=args.max_steps,
    )

    run_diagnostic_episode(agent, env, args, holes, compare_bp_agent=compare_bp_agent)


if __name__ == "__main__":
    main()
