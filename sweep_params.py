#!/usr/bin/env python
"""Parameter sweep for Frozen Lake — reuses run_frozen_lake.run_episode exactly."""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax.numpy as jnp
from environments.frozen_lake import (
    sample_configs, generate_transition_tensor,
    generate_observation_tensor, generate_goal, FrozenLakeEnv,
)
from agents.frozen_lake_agent import create_agent
from run_frozen_lake import run_episode

grid_size = 5
seed = 0
slip_prob = 0.1
max_steps = 30
planning_horizon = 15
n_episodes = 20


def evaluate(method_key, T, B, goal, holes, iters, damp):
    agent = create_agent(method_key, T, B, goal, holes,
                          planning_horizon=planning_horizon,
                          planning_iterations=iters, damping=damp)
    env = FrozenLakeEnv(grid_size=grid_size, holes=holes, obs_tensor=B,
                         slip_prob=slip_prob, max_steps=max_steps)
    ok = 0
    for i in range(n_episodes):
        result = run_episode(agent, env, seed=seed + i, receding_horizon=True)
        if result["success"]:
            ok += 1
    return ok / n_episodes


# --- Sweep parameters ---
n_configs_list = [4]
min_hamming = 4
hole_fraction = 0.2

base_noise_list = [0.05, 0.15, 0.25]
noise_range_list = [0.0, 0.15, 0.30]
hole_penalty_list = [0.0, 0.1, 1.0, 5.0]
goal_temp_list = [0.01, 0.05, 0.1, 0.5]

print(f"Grid: {grid_size}x{grid_size}, slip={slip_prob}, episodes={n_episodes}")
print(f"Planning horizon: {planning_horizon}, max_steps: {max_steps}")
print()

for nc in n_configs_list:
    holes = sample_configs(grid_size, nc, hole_fraction=hole_fraction,
                           seed=seed, min_hamming=min_hamming)
    T = generate_transition_tensor(grid_size, holes, slip_prob=slip_prob)

    print(f"{'='*80}")
    print(f"n_configs={nc}, min_hamming={min_hamming}")
    print(f"{'='*80}")
    print()

    header = f"{'bn':>5} {'nr':>5} {'hp':>5} {'gt':>5}  {'BP':>5} {'RE25':>5} {'RE50':>5} {'DC25':>5}"
    print(header)
    print("-" * len(header))

    for bn in base_noise_list:
        for nr in noise_range_list:
            B = generate_observation_tensor(grid_size, holes, base_noise=bn,
                                             noise_range=nr)
            for hp in hole_penalty_list:
                for gt in goal_temp_list:
                    goal = generate_goal(grid_size, holes, hole_penalty=hp,
                                          temperature=gt)

                    bp = evaluate("bp", T, B, goal, holes, 1, 1.0)
                    re25 = evaluate("region_extended", T, B, goal, holes, 25, 0.25)
                    re50 = evaluate("region_extended", T, B, goal, holes, 50, 0.25)
                    dc25 = evaluate("dyn_channel", T, B, goal, holes, 25, 0.25)

                    marker = ""
                    if re25 > bp:
                        marker = " <<<"
                    elif re50 > bp:
                        marker = " <<50"
                    elif abs(re25 - bp) < 0.05:
                        marker = " =="

                    print(f"{bn:>5.2f} {nr:>5.2f} {hp:>5.1f} {gt:>5.2f}"
                          f"  {bp:>4.0%} {re25:>4.0%} {re50:>4.0%} {dc25:>4.0%}{marker}")
                    sys.stdout.flush()
    print()
