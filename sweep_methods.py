#!/usr/bin/env python
"""Sweep parameter configs comparing region-extended vs loopy vs loopy-vbp.

Generates tensors once per config, creates 3 agents, runs all on same episodes.
Prints results table highlighting configs where region-extended wins.
"""

import sys
from pathlib import Path
import itertools
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
)
from agents.frozen_lake_agent import create_agent
from run_frozen_lake import run_episode
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
GRID_SIZE = 4
SEED = 0
EPISODES = 100
MAX_STEPS = 15
PLANNING_HORIZON = 15
SLIP_PROB = 0.1
MIN_HAMMING = 0
HOLE_FRACTION = 0.2

# ---------------------------------------------------------------------------
# Swept parameters
# ---------------------------------------------------------------------------
SWEEP = {
    "n_configs": [5, 10, 20],
    "base_noise": [0.1, 0.3, 0.5],
    "hole_penalty": [2, 4, 8],
    "goal_temperature": [1.0, 3.0],
    "scan_cost": [0.1, 0.5],
}

# ---------------------------------------------------------------------------
# Per-method configs (method_key, iterations, damping)
# ---------------------------------------------------------------------------
METHODS = {
    "loopy_vbp":        {"iterations": 50, "damping": 1.0},
    "loopy_bp":         {"iterations": 50, "damping": 1.0},
    "region_extended":  {"iterations": 50, "damping": 0.5},
}

METHOD_DISPLAY = {
    "loopy_vbp": "loopy-vbp",
    "loopy_bp": "loopy",
    "region_extended": "region-ext",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def run_config(params: dict) -> dict:
    """Run all 3 methods on one parameter config. Returns success rates."""
    n_configs = params["n_configs"]
    base_noise = params["base_noise"]
    hole_penalty = params["hole_penalty"]
    goal_temperature = params["goal_temperature"]
    scan_cost = params["scan_cost"]

    set_seed(SEED)

    holes = sample_configs(
        GRID_SIZE, n_configs,
        hole_fraction=HOLE_FRACTION, seed=SEED,
        min_hamming=MIN_HAMMING,
    )
    T = generate_transition_tensor(GRID_SIZE, holes, slip_prob=SLIP_PROB)
    B = generate_observation_tensor(GRID_SIZE, holes, base_noise=base_noise)
    goal = generate_goal(GRID_SIZE, holes, hole_penalty=hole_penalty,
                         temperature=goal_temperature)

    action_prior = np.array([1.0, 1.0, 1.0, 1.0, scan_cost], dtype=np.float32)
    action_prior = action_prior / action_prior.sum()

    env = FrozenLakeEnv(
        grid_size=GRID_SIZE, holes=holes, obs_tensor=B,
        slip_prob=SLIP_PROB, max_steps=MAX_STEPS,
    )

    results = {}
    for method_key, mcfg in METHODS.items():
        agent = create_agent(
            method_key, T, B, goal, holes,
            planning_horizon=PLANNING_HORIZON,
            planning_iterations=mcfg["iterations"],
            action_prior=action_prior,
            damping=mcfg["damping"],
        )

        successes = 0
        for i in range(EPISODES):
            ep_seed = SEED + i
            ep = run_episode(agent, env, seed=ep_seed)
            if ep["success"]:
                successes += 1

        results[method_key] = successes / EPISODES

    return results


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print()

    keys = list(SWEEP.keys())
    values = list(SWEEP.values())
    combos = list(itertools.product(*values))
    n_combos = len(combos)

    print(f"Sweep: {' x '.join(str(len(v)) for v in values)} = {n_combos} configs")
    print(f"Methods: {', '.join(METHOD_DISPLAY.values())}")
    print(f"Episodes per method: {EPISODES}")
    print(f"Total episodes: {n_combos * len(METHODS) * EPISODES}")
    print()

    # Table header
    header_params = "n_cfg noise pen  temp scan"
    header_methods = "  ".join(f"{METHOD_DISPLAY[m]:>10}" for m in METHODS)
    header = f"{header_params}  |  {header_methods}  | win?"
    print(header)
    print("-" * len(header))

    all_results = []
    wins = 0

    t0 = time.time()
    for idx, combo in enumerate(tqdm(combos, desc="Configs")):
        params = dict(zip(keys, combo))

        results = run_config(params)

        re_rate = results["region_extended"]
        loopy_rate = results["loopy_bp"]
        vbp_rate = results["loopy_vbp"]
        re_wins = re_rate > loopy_rate and re_rate > vbp_rate

        if re_wins:
            wins += 1
            marker = " <<<"
        elif re_rate >= loopy_rate and re_rate >= vbp_rate:
            marker = "  =="
        else:
            marker = ""

        param_str = (
            f"{params['n_configs']:5d} "
            f"{params['base_noise']:5.1f} "
            f"{params['hole_penalty']:3d}  "
            f"{params['goal_temperature']:4.1f} "
            f"{params['scan_cost']:4.1f}"
        )
        result_str = "  ".join(
            f"{results[m]:10.1%}" for m in METHODS
        )
        print(f"{param_str}  |  {result_str}  |{marker}")

        all_results.append({**params, **results, "re_wins": re_wins})

    elapsed = time.time() - t0

    # Summary
    print()
    print("=" * 60)
    print(f"SUMMARY: region-extended wins in {wins}/{n_combos} configs")
    print(f"Total time: {elapsed:.0f}s ({elapsed / n_combos:.1f}s/config)")
    print()

    if wins > 0:
        print("Configs where region-extended wins:")
        print()
        for r in all_results:
            if r["re_wins"]:
                print(
                    f"  n_configs={r['n_configs']}, "
                    f"noise={r['base_noise']}, "
                    f"penalty={r['hole_penalty']}, "
                    f"temp={r['goal_temperature']}, "
                    f"scan={r['scan_cost']}  →  "
                    f"RE={r['region_extended']:.1%}  "
                    f"loopy={r['loopy_bp']:.1%}  "
                    f"vbp={r['loopy_vbp']:.1%}"
                )

    # Best advantage
    if all_results:
        best = max(
            all_results,
            key=lambda r: r["region_extended"] - max(r["loopy_bp"], r["loopy_vbp"]),
        )
        adv = best["region_extended"] - max(best["loopy_bp"], best["loopy_vbp"])
        print()
        print(f"Best region-extended advantage: {adv:+.1%}")
        print(
            f"  n_configs={best['n_configs']}, "
            f"noise={best['base_noise']}, "
            f"penalty={best['hole_penalty']}, "
            f"temp={best['goal_temperature']}, "
            f"scan={best['scan_cost']}  →  "
            f"RE={best['region_extended']:.1%}  "
            f"loopy={best['loopy_bp']:.1%}  "
            f"vbp={best['loopy_vbp']:.1%}"
        )


if __name__ == "__main__":
    main()
