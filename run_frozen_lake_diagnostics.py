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
    state_index,
    unpack_state,
    N_ACTIONS,
)
from agents.frozen_lake_agent import create_agent

ACTION_NAMES = ["left", "down", "right", "up", "scan"]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def entropy(p):
    """Shannon entropy in bits."""
    p = jnp.clip(p, 1e-12, 1.0)
    return -float(jnp.sum(p * jnp.log2(p)))


def print_position_grid(belief, grid_size, label="Position belief"):
    """Print a 2D grid of position probabilities (marginalised over scan mode)."""
    n_pos = grid_size * grid_size
    # Sum over scan modes: belief has 2*n_pos states
    pos_marginal = np.array(belief[:n_pos]) + np.array(belief[n_pos:])
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
    """Print position + grid cell sensor readings."""
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos

    # Position modality: first 2*n_pos channels
    pos_obs = obs[:n_states]
    observed_state = int(jnp.argmax(pos_obs))
    pos, scanned = unpack_state(observed_state, n_pos)
    r, c = pos_to_rc(pos, grid_size)
    mode_str = "SCANNED" if scanned else "UNSCANNED"
    print(f"    Position: ({r},{c})  mode={mode_str}")

    # Grid cell modality: last n_pos channels (hole sensors)
    cell_obs = obs[n_states:]
    print(f"    Grid cell sensors (1=hole detected, row\\col):")
    header = "        " + "  ".join(f"c={c:>2}" for c in range(grid_size))
    print(header)
    for rr in range(grid_size):
        row_str = f"  r={rr:>2}  "
        for cc in range(grid_size):
            idx = rr * grid_size + cc
            if idx < len(cell_obs):
                val = float(cell_obs[idx])
                row_str += f"  {val:.0f}   "
            else:
                row_str += "   ?  "
        print(row_str)


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

    # Show goal values for key states (unscanned mode)
    goal_r, goal_c = pos_to_rc(goal_pos, grid_size)
    goal_val_unscanned = float(goal_avg[state_index(goal_pos, 0, n_pos)])
    goal_val_scanned = float(goal_avg[state_index(goal_pos, 1, n_pos)])
    print(f"    Goal position ({goal_r},{goal_c}): avg unscanned={goal_val_unscanned:.6f}, avg scanned={goal_val_scanned:.6f}")

    # Show per-config variation at goal position
    goal_vals = np.array(goal[state_index(goal_pos, 0, n_pos), :])
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
            val = float(goal_avg[state_index(pos, 0, n_pos)])
            print(f"      ({r},{c}): avg_goal={val:.6f}  P(hole)={hole_marginal[pos]:.3f}")

    if normal_positions:
        print(f"    Safe positions (P(hole) < 0.1):")
        for pos in normal_positions[:5]:
            r, c = pos_to_rc(pos, grid_size)
            val = float(goal_avg[state_index(pos, 0, n_pos)])
            print(f"      ({r},{c}): avg_goal={val:.6f}  P(hole)={hole_marginal[pos]:.3f}")

    # Summary: ratio between goal, safe, and hole values
    goal_val = float(goal_avg[state_index(goal_pos, 0, n_pos)])
    if hole_positions:
        avg_hole = np.mean([float(goal_avg[state_index(p, 0, n_pos)]) for p in hole_positions])
    else:
        avg_hole = 0.0
    if normal_positions:
        avg_safe = np.mean([float(goal_avg[state_index(p, 0, n_pos)]) for p in normal_positions])
    else:
        avg_safe = 0.0

    print(f"    Summary (config-avg): goal={goal_val:.6f}, avg_safe={avg_safe:.6f}, avg_hole={avg_hole:.6f}")
    if avg_safe > 0:
        print(f"    Ratios: goal/safe={goal_val/avg_safe:.2f}x, safe/hole={avg_safe/max(avg_hole, 1e-12):.2f}x")
    print()


def print_scan_mode_belief(belief, grid_size):
    """Print belief mass in unscanned vs scanned mode."""
    n_pos = grid_size * grid_size
    unscanned_mass = float(jnp.sum(belief[:n_pos]))
    scanned_mass = float(jnp.sum(belief[n_pos:]))
    print(f"    Scan mode: unscanned={unscanned_mass:.4f}, scanned={scanned_mass:.4f}")


# ---------------------------------------------------------------------------
# Diagnostic episode
# ---------------------------------------------------------------------------


def run_diagnostic_episode(agent, env, args, holes, compare_bp_agent=None):
    grid_size = args.grid_size
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos
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
        print(f"  [TRUE STATE] position=({true_r},{true_c})  scanned={env._scanned}  config={theta}")
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

        # --- SCAN MODE BELIEF ---
        print("  [SCAN MODE BELIEF]")
        print_scan_mode_belief(agent.q_current_state, grid_size)
        print()

        # --- POSITION BELIEF ---
        print("  [POSITION BELIEF]")
        q_pos = agent.q_current_state
        print_position_grid(q_pos, grid_size)

        # MAP over position marginal (sum over scan modes)
        pos_marginal = np.array(q_pos[:n_pos]) + np.array(q_pos[n_pos:])
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
        true_state_unscanned = state_index(env._position, 0, n_pos)
        true_state_scanned = state_index(env._position, 1, n_pos)
        goal_state_unscanned = state_index(n_pos - 1, 0, n_pos)
        goal_state_scanned = state_index(n_pos - 1, 1, n_pos)
        print(f"    Goal value at current pos ({true_r},{true_c}) [config {theta}]: "
              f"unscanned={float(agent.goal[true_state_unscanned, theta]):.6f}, "
              f"scanned={float(agent.goal[true_state_scanned, theta]):.6f}")
        print(f"    Goal value at goal pos [config {theta}]: "
              f"unscanned={float(agent.goal[goal_state_unscanned, theta]):.6f}, "
              f"scanned={float(agent.goal[goal_state_scanned, theta]):.6f}")

        # Show goal values at known hole positions
        hole_config = holes[theta]
        hole_positions = [p for p in range(n_pos) if hole_config[p] == 1.0]
        if hole_positions:
            for hp in hole_positions[:4]:
                hr, hc = pos_to_rc(hp, grid_size)
                hval = float(agent.goal[state_index(hp, 0, n_pos), theta])
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
    parser.add_argument("--base-noise", type=float, default=0.05)
    parser.add_argument("--noise-range", type=float, default=0.15)
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
    parser.add_argument("--scan-cost", type=float, default=0.5,
                        help="SCAN action prior weight (lower = more costly)")
    parser.add_argument("--receding-horizon", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    grid_size = args.grid_size
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print()
    print(f"Frozen Lake {grid_size}x{grid_size}")
    print(f"  Configs: {args.n_configs}, hole fraction: {args.hole_fraction}")
    print(f"  Base noise: {args.base_noise}, noise range: {args.noise_range}")
    print(f"  Slip prob: {args.slip_prob}")
    print(f"  Method: {args.planning_method}")
    print(f"  Horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    print(f"  Iterations: {args.planning_iterations}")
    if args.damping < 1.0:
        print(f"  Damping: {args.damping}")
    print(f"  Hole penalty: {args.hole_penalty}, goal temperature: {args.goal_temperature}")
    print(f"  Scan cost: {args.scan_cost}")
    print(f"  Seed: {args.seed}")
    print(f"  State space: {n_states} states ({n_pos} positions x 2 scan modes)")
    print()

    print("Generating tensors...")
    t0 = time.time()

    holes = sample_configs(
        grid_size, args.n_configs,
        hole_fraction=args.hole_fraction, seed=args.seed,
        min_hamming=args.min_hamming,
    )
    T = generate_transition_tensor(grid_size, holes, slip_prob=args.slip_prob)
    B = generate_observation_tensor(grid_size, holes, base_noise=args.base_noise,
                                    noise_range=args.noise_range)
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

    # Construct action prior: [1, 1, 1, 1, scan_cost] normalized
    action_prior = np.array([1.0, 1.0, 1.0, 1.0, args.scan_cost], dtype=np.float32)
    action_prior = action_prior / action_prior.sum()

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
