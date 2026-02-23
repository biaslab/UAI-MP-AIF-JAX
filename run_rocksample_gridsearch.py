#!/usr/bin/env python
"""Grid search over RockSample[5,3] hyperparameters.

Compares region-extended, loopy, and loopy-vbp planning methods across
combinations of half_eff_dist, scan_cost, and goal_temperature.
"""

import sys
from pathlib import Path
import json
import argparse
import time
import random
from itertools import product

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
    state_index,
    rc_to_pos,
    RockSampleEnv,
)
from agents.rocksample_agent import create_agent
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
GRID_SIZE = 4
N_ROCKS = 2
GOOD_REWARD = 2.0
BAD_PENALTY = 2.0
EXIT_REWARD = 1.0
SAMPLE_COST = 1.0
POS_NOISE = 0.3
SLIP_PROB = 0.0
MAX_STEPS = 10
PLANNING_HORIZON = 10
TERMINAL_GOAL_ONLY = True
SEED = 0

# Search space
HALF_EFF_DISTS = [0.5, 1.0, 2.0]
SCAN_COSTS = [0.1, 0.5, 1.0]
GOAL_TEMPERATURES = [0.5, 1.0, 2.0]

# Per-method settings from params.yaml
METHODS = {
    "loopy_bp": {"iterations": 30, "damping": 1.0, "label": "loopy"},
    "loopy_vbp": {"iterations": 30, "damping": 1.0, "label": "loopy-vbp"},
    "region_extended": {"iterations": 50, "damping": 0.25, "label": "region-extended"},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def run_episode(agent, env, seed=None, verbose=False):
    """Run a single RockSample episode."""
    result = env.reset(seed=seed)
    agent = agent.reset()

    total_reward = 0.0
    steps = 0
    max_steps = env.max_steps

    while True:
        time_remaining = agent.planning_horizon

        obs = jnp.array(result.obs)
        action, agent = agent.step(obs, time_remaining)

        result = env.step(action)
        total_reward += result.reward
        steps += 1

        if result.terminated or result.truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": result.terminated and result.reward > 0,
    }


def make_action_prior(scan_cost: float) -> np.ndarray:
    """Build normalized action prior with given scan cost."""
    prior = np.array(
        [1.0, 1.0, 1.0, 1.0, scan_cost, SAMPLE_COST],
        dtype=np.float32,
    )
    return prior / prior.sum()


def main():
    parser = argparse.ArgumentParser(description="RockSample grid search")
    parser.add_argument("--episodes", type=int, default=25, help="Episodes per config (default: 100)")
    args = parser.parse_args()

    n_episodes = args.episodes
    set_seed(SEED)

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"\nRockSample[{GRID_SIZE},{N_ROCKS}] grid search")
    print(f"  Search space: {len(HALF_EFF_DISTS)} half_eff_dist × {len(SCAN_COSTS)} scan_cost × {len(GOAL_TEMPERATURES)} goal_temp = {len(HALF_EFF_DISTS) * len(SCAN_COSTS) * len(GOAL_TEMPERATURES)} configs")
    print(f"  Methods: {', '.join(m['label'] for m in METHODS.values())}")
    total_runs = len(HALF_EFF_DISTS) * len(SCAN_COSTS) * len(GOAL_TEMPERATURES) * len(METHODS)
    print(f"  Total: {total_runs} runs × {n_episodes} episodes = {total_runs * n_episodes} episodes")
    print()

    # --- Pre-generate tensors ---
    n_pos = GRID_SIZE * GRID_SIZE
    n_collect = 2 ** N_ROCKS
    n_scan = 2 ** N_ROCKS
    start_pos = rc_to_pos(GRID_SIZE // 2, 0, GRID_SIZE)
    start_state_idx = state_index(start_pos, 0, 0, n_pos, n_collect, n_scan)

    rock_positions = sample_rock_positions(GRID_SIZE, N_ROCKS, seed=SEED)
    qualities = all_quality_configs(N_ROCKS)

    print(f"Rock positions: {rock_positions.tolist()}")
    print("Generating tensors...")
    t0 = time.time()

    # T: depends only on grid_size, rock_positions, slip_prob → generate once
    T = generate_transition_tensor(GRID_SIZE, rock_positions, N_ROCKS, slip_prob=SLIP_PROB)
    print(f"  T: {T.shape} ({T.nbytes / 1024:.1f} KB)")

    # B: depends on half_eff_dist → one per value
    B_cache = {}
    for hed in HALF_EFF_DISTS:
        B_cache[hed] = generate_observation_tensor(
            GRID_SIZE, rock_positions, qualities, N_ROCKS,
            half_eff_dist=hed, pos_noise=POS_NOISE,
        )
    print(f"  B: {len(B_cache)} variants, each {B_cache[HALF_EFF_DISTS[0]].shape}")

    # Goal: depends on goal_temperature → one per value
    goal_cache = {}
    for gt in GOAL_TEMPERATURES:
        goal_cache[gt] = generate_goal(
            GRID_SIZE, rock_positions, qualities, N_ROCKS,
            exit_reward=EXIT_REWARD, good_reward=GOOD_REWARD,
            bad_penalty=BAD_PENALTY, temperature=gt,
        )
    print(f"  Goal: {len(goal_cache)} variants, each {goal_cache[GOAL_TEMPERATURES[0]].shape}")

    # Action priors: depends on scan_cost → one per value
    prior_cache = {sc: make_action_prior(sc) for sc in SCAN_COSTS}

    print(f"  Tensor generation: {time.time() - t0:.2f}s")
    print()

    # --- Grid search ---
    configs = list(product(HALF_EFF_DISTS, SCAN_COSTS, GOAL_TEMPERATURES))
    all_results = []  # list of dicts

    pbar = tqdm(
        total=len(configs) * len(METHODS),
        desc="Grid search",
    )

    for hed, sc, gt in configs:
        B = B_cache[hed]
        goal = goal_cache[gt]
        action_prior = prior_cache[sc]

        env = RockSampleEnv(
            grid_size=GRID_SIZE,
            rock_positions=rock_positions,
            qualities=qualities,
            n_rocks=N_ROCKS,
            obs_tensor=B,
            slip_prob=SLIP_PROB,
            max_steps=MAX_STEPS,
            good_reward=GOOD_REWARD,
            bad_penalty=BAD_PENALTY,
            exit_reward=EXIT_REWARD,
        )

        for method_key, method_cfg in METHODS.items():
            agent = create_agent(
                method_key, T, B, goal,
                rock_positions, qualities, n_pos, start_state_idx,
                planning_horizon=PLANNING_HORIZON,
                planning_iterations=method_cfg["iterations"],
                action_prior=action_prior,
                damping=method_cfg["damping"],
                terminal_goal_only=TERMINAL_GOAL_ONLY,
            )

            rewards = []
            successes = 0
            for i in range(n_episodes):
                ep = run_episode(agent, env, seed=SEED + i)
                rewards.append(ep["total_reward"])
                if ep["success"]:
                    successes += 1

            avg_reward = sum(rewards) / n_episodes
            success_rate = successes / n_episodes

            all_results.append({
                "half_eff_dist": hed,
                "scan_cost": sc,
                "goal_temperature": gt,
                "method": method_cfg["label"],
                "avg_reward": avg_reward,
                "success_rate": success_rate,
            })

            pbar.set_postfix({
                "hed": hed, "sc": sc, "gt": gt,
                "method": method_cfg["label"],
                "reward": f"{avg_reward:.2f}",
            })
            pbar.update(1)

    pbar.close()
    print()

    # --- Analysis ---
    # Group by config, compare region-extended vs others
    from collections import defaultdict
    by_config = defaultdict(dict)
    for r in all_results:
        key = (r["half_eff_dist"], r["scan_cost"], r["goal_temperature"])
        by_config[key][r["method"]] = r

    leaderboard = []
    for key, methods in by_config.items():
        re = methods.get("region-extended")
        loopy = methods.get("loopy")
        lvbp = methods.get("loopy-vbp")
        if not (re and loopy and lvbp):
            continue

        margin_loopy = re["avg_reward"] - loopy["avg_reward"]
        margin_lvbp = re["avg_reward"] - lvbp["avg_reward"]
        min_margin = min(margin_loopy, margin_lvbp)

        leaderboard.append({
            "half_eff_dist": key[0],
            "scan_cost": key[1],
            "goal_temperature": key[2],
            "re_reward": re["avg_reward"],
            "re_success": re["success_rate"],
            "loopy_reward": loopy["avg_reward"],
            "loopy_success": loopy["success_rate"],
            "lvbp_reward": lvbp["avg_reward"],
            "lvbp_success": lvbp["success_rate"],
            "margin_vs_loopy": margin_loopy,
            "margin_vs_lvbp": margin_lvbp,
            "min_margin": min_margin,
        })

    leaderboard.sort(key=lambda x: x["min_margin"], reverse=True)

    # --- Print leaderboard ---
    print("=" * 100)
    print("Leaderboard: configs sorted by region-extended advantage (min margin over loopy & loopy-vbp)")
    print("=" * 100)
    print(f"{'half_eff_dist':>13} {'scan_cost':>10} {'goal_temp':>10} | "
          f"{'RE reward':>10} {'Loopy':>10} {'L-VBP':>10} | "
          f"{'vs Loopy':>10} {'vs L-VBP':>10} {'min':>10}")
    print("-" * 100)
    for row in leaderboard:
        print(f"{row['half_eff_dist']:>13.1f} {row['scan_cost']:>10.1f} {row['goal_temperature']:>10.1f} | "
              f"{row['re_reward']:>10.3f} {row['loopy_reward']:>10.3f} {row['lvbp_reward']:>10.3f} | "
              f"{row['margin_vs_loopy']:>+10.3f} {row['margin_vs_lvbp']:>+10.3f} {row['min_margin']:>+10.3f}")

    # --- Save results ---
    output_path = Path("data/results/rocksample_gridsearch.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "environment": "rocksample",
            "grid_size": GRID_SIZE,
            "n_rocks": N_ROCKS,
            "good_reward": GOOD_REWARD,
            "bad_penalty": BAD_PENALTY,
            "exit_reward": EXIT_REWARD,
            "sample_cost": SAMPLE_COST,
            "pos_noise": POS_NOISE,
            "slip_prob": SLIP_PROB,
            "max_steps": MAX_STEPS,
            "planning_horizon": PLANNING_HORIZON,
            "terminal_goal_only": TERMINAL_GOAL_ONLY,
            "seed": SEED,
            "n_episodes": n_episodes,
            "search_space": {
                "half_eff_dist": HALF_EFF_DISTS,
                "scan_cost": SCAN_COSTS,
                "goal_temperature": GOAL_TEMPERATURES,
            },
            "methods": {k: {"iterations": v["iterations"], "damping": v["damping"]}
                        for k, v in METHODS.items()},
        },
        "results": all_results,
        "leaderboard": leaderboard,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
