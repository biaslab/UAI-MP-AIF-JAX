#!/usr/bin/env python
"""Quick n_configs scaling check — uses run_episode from run_frozen_lake.py."""

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
base_noise = 0.05
noise_range = 0.15
hole_penalty = 0.1
goal_temperature = 0.1
hole_fraction = 0.2
min_hamming = 4


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


header = f"{'nc':>4} {'BP':>5} {'RE25':>5} {'RE50':>5} {'DC25':>5} {'Nuij':>5}"
print(header)
print("-" * len(header))

for nc in [4, 10, 20, 40]:
    holes = sample_configs(grid_size, nc, hole_fraction=hole_fraction,
                           seed=seed, min_hamming=min_hamming)
    T = generate_transition_tensor(grid_size, holes, slip_prob=slip_prob)
    B = generate_observation_tensor(grid_size, holes, base_noise=base_noise,
                                     noise_range=noise_range)
    goal = generate_goal(grid_size, holes, hole_penalty=hole_penalty,
                          temperature=goal_temperature)

    bp = evaluate("bp", T, B, goal, holes, 1, 1.0)
    re25 = evaluate("region_extended", T, B, goal, holes, 25, 0.25)
    re50 = evaluate("region_extended", T, B, goal, holes, 50, 0.25)
    dc25 = evaluate("dyn_channel", T, B, goal, holes, 25, 0.25)
    nu = evaluate("nuijten", T, B, goal, holes, 30, 1.0)

    print(f"{nc:>4} {bp:>4.0%} {re25:>4.0%} {re50:>4.0%} {dc25:>4.0%} {nu:>4.0%}")
    sys.stdout.flush()
