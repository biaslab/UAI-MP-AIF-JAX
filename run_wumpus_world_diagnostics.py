#!/usr/bin/env python
"""Single-episode diagnostic script for Wumpus World with full internal state output."""

import sys
from pathlib import Path
import argparse
import time
import random

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
from environments.wumpus_world import (
    sample_configs,
    generate_transition_tensor,
    generate_observation_tensor,
    generate_goal,
    WumpusWorldEnv,
    pos_to_rc,
    N_ACTIONS,
)
from agents.wumpus_agent import create_agent

ACTION_NAMES = ["left", "down", "right", "up", "scan"]
FEATURE_OBS_NAMES = ["breeze", "stench", "glitter"]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def entropy(p):
    """Shannon entropy in bits."""
    p = jnp.clip(p, 1e-12, 1.0)
    return -float(jnp.sum(p * jnp.log2(p)))


def print_position_grid(belief, grid_size, label="Position belief"):
    """Print a 2D grid of position probabilities."""
    print(f"    {label} (row\\col):")
    header = "        " + "  ".join(f"c={c:>2}" for c in range(grid_size))
    print(header)
    for r in range(grid_size):
        row = f"  r={r:>2}  "
        for c in range(grid_size):
            pos = r * grid_size + c
            p = float(belief[pos])
            if p < 0.0005:
                row += "   .  "
            else:
                row += f"{p:5.3f} "
        print(row)


def print_action_distribution(action_dist):
    """Print actions with bars."""
    bar_width = 40
    for i, name in enumerate(ACTION_NAMES):
        p = float(action_dist[i])
        n_bars = int(p * bar_width)
        bar = "#" * n_bars
        print(f"      {name:>5s}: {p:.4f}  {bar}")


def print_static_summary(q_static, pits, wumpus, gold, grid_size, top_k=5):
    """Print top-k most likely configurations with their entity positions."""
    n_static = q_static.shape[0]
    order = jnp.argsort(-q_static)
    print(f"    Top-{min(top_k, n_static)} static configs:")
    for rank in range(min(top_k, n_static)):
        idx = int(order[rank])
        p = float(q_static[idx])
        if p < 0.001:
            break
        pit_positions = []
        for pos in range(grid_size * grid_size):
            if pits[idx, pos] == 1.0:
                r, c = pos_to_rc(pos, grid_size)
                pit_positions.append(f"({r},{c})")
        pits_str = " ".join(pit_positions) if pit_positions else "none"

        wumpus_pos = int(np.argmax(wumpus[idx]))
        wr, wc = pos_to_rc(wumpus_pos, grid_size)

        gold_pos = int(np.argmax(gold[idx]))
        gr, gc = pos_to_rc(gold_pos, grid_size)

        print(f"      #{idx:>3d}: p={p:.4f}  pits={pits_str}  wumpus=({wr},{wc})  gold=({gr},{gc})")


def print_danger_heatmap(q_static, pits, wumpus, grid_size):
    """Print expected danger probability per cell (marginal over configs)."""
    n_states = grid_size * grid_size
    danger = np.zeros(n_states)
    for pos in range(n_states):
        # P(danger at pos) = Σ_θ q(θ) * (pit[θ,pos] or wumpus[θ,pos])
        for theta in range(q_static.shape[0]):
            if pits[theta, pos] == 1.0 or wumpus[theta, pos] == 1.0:
                danger[pos] += float(q_static[theta])

    print("    Expected danger heatmap (row\\col):")
    header = "        " + "  ".join(f"c={c:>2}" for c in range(grid_size))
    print(header)
    for r in range(grid_size):
        row = f"  r={r:>2}  "
        for c in range(grid_size):
            pos = r * grid_size + c
            d = danger[pos]
            if d < 0.0005:
                row += "   .  "
            else:
                row += f"{d:5.3f} "
        print(row)


# ---------------------------------------------------------------------------
# Diagnostic episode
# ---------------------------------------------------------------------------


def run_diagnostic_episode(agent, env, args, pits, wumpus_arr, gold):
    grid_size = args.grid_size
    n_states = grid_size * grid_size
    n_static = pits.shape[0]
    max_steps = env.max_steps

    max_entropy_pos = jnp.log2(float(n_states))
    max_entropy_static = jnp.log2(float(n_static))

    result = env.reset(seed=args.seed)
    agent = agent.reset()
    theta = env.config_idx

    # Show true config
    true_gold_pos = int(np.argmax(gold[theta]))
    true_wumpus_pos = int(np.argmax(wumpus_arr[theta]))
    true_pit_positions = [pos for pos in range(n_states) if pits[theta, pos] == 1.0]

    print("=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    print()
    print(f"  True config index: {theta}")
    gr, gc = pos_to_rc(true_gold_pos, grid_size)
    wr, wc = pos_to_rc(true_wumpus_pos, grid_size)
    pits_str = ", ".join(f"({pos_to_rc(p, grid_size)[0]},{pos_to_rc(p, grid_size)[1]})" for p in true_pit_positions)
    print(f"  Gold: ({gr},{gc})  Wumpus: ({wr},{wc})  Pits: {pits_str}")
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
        n_obs = len(result.obs)
        obs_names = FEATURE_OBS_NAMES + [f"pos{i}" for i in range(n_obs - len(FEATURE_OBS_NAMES))]
        print("  [OBSERVATION]")
        obs_parts = []
        for i, name in enumerate(obs_names):
            val = int(obs[i])
            obs_parts.append(f"{name}={'YES' if val else 'no'}")
        print(f"    {', '.join(obs_parts)}")
        print()

        # --- STATE INFERENCE + PLANNING (inside agent.step) ---
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
        # Marginalize over scan mode: sum unscanned + scanned halves
        print("  [POSITION BELIEF]")
        q_full = agent.q_current_state
        q_pos = q_full[:n_states] + q_full[n_states:]  # (n_pos,)
        scan_mass = float(q_full[n_states:].sum())
        print_position_grid(q_pos, grid_size)

        map_pos = int(jnp.argmax(q_pos))
        map_r, map_c = pos_to_rc(map_pos, grid_size)
        map_p = float(q_pos[map_pos])
        correct = (map_pos == env._position)
        print(f"    MAP position: ({map_r},{map_c}) p={map_p:.4f} {'CORRECT' if correct else 'WRONG'}")
        print(f"    P(scanned): {scan_mass:.4f}")
        print(f"    Position entropy: {entropy(q_pos):.2f} bits (max={max_entropy_pos:.2f})")
        print()

        # --- STATIC BELIEF ---
        print("  [STATIC BELIEF]")
        q_static = agent.q_static_state
        print_static_summary(q_static, pits, wumpus_arr, gold, grid_size)

        true_config_p = float(q_static[theta])
        true_config_rank = int((q_static > q_static[theta]).sum())
        print(f"    True config #{theta}: p={true_config_p:.4f} (rank {true_config_rank + 1}/{n_static})")
        print(f"    Static entropy: {entropy(q_static):.2f} bits (max={max_entropy_static:.2f})")
        print()

        # --- DANGER HEATMAP ---
        print("  [DANGER HEATMAP]")
        print_danger_heatmap(q_static, pits, wumpus_arr, grid_size)
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
                print("  >>> FOUND GOLD!")
            elif result.reward < 0:
                print("  >>> DIED! (pit or wumpus)")
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
    parser = argparse.ArgumentParser(description="Wumpus World single-episode diagnostic")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--n-pits", type=int, default=2)
    parser.add_argument("--obs-noise", type=float, default=0.4)
    parser.add_argument("--pos-noise", type=float, default=0.1)
    parser.add_argument("--slip-prob", type=float, default=0.1)
    parser.add_argument("--pit-penalty", type=float, default=2.0)
    parser.add_argument("--wumpus-penalty", type=float, default=2.0)
    parser.add_argument("--goal-temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--planning-horizon", type=int, default=15)
    parser.add_argument("--planning-iterations", type=int, default=3)
    parser.add_argument("--planning-method", type=str, default="loopy",
                        choices=["loopy-vbp", "loopy", "region-extended",
                                 "dyn-channel", "nuijten", "vbp-channel",
                                 "precise-info-seeking", "active-inference"])
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.0, help="Inertial momentum coefficient (0.0 = no momentum)")
    parser.add_argument("--scan-cost", type=float, default=0.1)
    parser.add_argument("--receding-horizon", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    grid_size = args.grid_size
    n_states = grid_size * grid_size

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print()
    print(f"Wumpus World {grid_size}x{grid_size}")
    print(f"  Configs: {args.n_configs}, pits: {args.n_pits}")
    print(f"  Obs noise: {args.obs_noise}, pos noise: {args.pos_noise}, slip prob: {args.slip_prob}")
    print(f"  Method: {args.planning_method}")
    print(f"  Horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    print(f"  Iterations: {args.planning_iterations}")
    if args.damping < 1.0:
        print(f"  Damping: {args.damping}")
    if args.momentum > 0:
        print(f"  Momentum: {args.momentum}")
    print(f"  Seed: {args.seed}")
    print()

    print("Generating tensors...")
    t0 = time.time()

    pits, wumpus_arr, gold = sample_configs(
        grid_size, args.n_configs, n_pits=args.n_pits, seed=args.seed,
    )
    T = generate_transition_tensor(grid_size, pits, wumpus_arr, slip_prob=args.slip_prob)
    B = generate_observation_tensor(
        grid_size, pits, wumpus_arr, gold, obs_noise=args.obs_noise,
        pos_noise=args.pos_noise,
    )
    goal = generate_goal(
        grid_size, pits, wumpus_arr, gold,
        pit_penalty=args.pit_penalty, wumpus_penalty=args.wumpus_penalty,
        temperature=args.goal_temperature,
    )

    print(f"  T: {T.shape}  B: {B.shape}")
    print(f"  Generated in {time.time() - t0:.2f}s")
    print()

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

    action_prior = np.array([1.0, 1.0, 1.0, 1.0, args.scan_cost], dtype=np.float32)
    action_prior = action_prior / action_prior.sum()

    agent = create_agent(
        method_key, T, B, goal,
        planning_horizon=args.planning_horizon,
        planning_iterations=args.planning_iterations,
        action_prior=action_prior,
        damping=args.damping,
        momentum=args.momentum,
    )

    env = WumpusWorldEnv(
        grid_size=grid_size, pits=pits, wumpus=wumpus_arr, gold=gold,
        obs_tensor=B, slip_prob=args.slip_prob, max_steps=args.max_steps,
    )

    run_diagnostic_episode(agent, env, args, pits, wumpus_arr, gold)


if __name__ == "__main__":
    main()
