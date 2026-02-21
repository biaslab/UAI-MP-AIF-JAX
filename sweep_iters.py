#!/usr/bin/env python
"""Test iteration count sweet spot for region-extended on Frozen Lake."""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["JAX_PLATFORMS"] = "cpu"

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
n_episodes = 50

# Best params from sweep
base_noise = 0.05
noise_range = 0.00
goal_temperature = 0.01
hole_penalty = 0.1

for nc in [4, 10, 20]:
    holes = sample_configs(grid_size, nc, hole_fraction=0.2,
                           seed=seed, min_hamming=4)
    T = generate_transition_tensor(grid_size, holes, slip_prob=slip_prob)
    B = generate_observation_tensor(grid_size, holes, base_noise=base_noise,
                                     noise_range=noise_range)
    goal = generate_goal(grid_size, holes, hole_penalty=hole_penalty,
                          temperature=goal_temperature)
    env = FrozenLakeEnv(grid_size=grid_size, holes=holes, obs_tensor=B,
                         slip_prob=slip_prob, max_steps=max_steps)

    print(f"\n{'='*60}")
    print(f"n_configs={nc}, bn={base_noise}, nr={noise_range}, gt={goal_temperature}")
    print(f"{'='*60}")

    # BP baseline
    agent = create_agent("bp", T, B, goal, holes,
                          planning_horizon=planning_horizon,
                          planning_iterations=1, damping=1.0)
    ok = sum(1 for i in range(n_episodes)
             if run_episode(agent, env, seed=seed+i, receding_horizon=True)["success"])
    print(f"  BP:           {ok/n_episodes:.0%}")

    # RE with different iteration counts
    for iters in [5, 10, 15, 20, 25, 30, 40, 50]:
        agent = create_agent("region_extended", T, B, goal, holes,
                              planning_horizon=planning_horizon,
                              planning_iterations=iters, damping=0.25)
        ok = sum(1 for i in range(n_episodes)
                 if run_episode(agent, env, seed=seed+i, receding_horizon=True)["success"])
        print(f"  RE iters={iters:>2d}:  {ok/n_episodes:.0%}")

    # DC for comparison
    for iters in [10, 25]:
        agent = create_agent("dyn_channel", T, B, goal, holes,
                              planning_horizon=planning_horizon,
                              planning_iterations=iters, damping=0.25)
        ok = sum(1 for i in range(n_episodes)
                 if run_episode(agent, env, seed=seed+i, receding_horizon=True)["success"])
        print(f"  DC iters={iters:>2d}:  {ok/n_episodes:.0%}")

    # Nuijten for comparison
    agent = create_agent("nuijten", T, B, goal, holes,
                          planning_horizon=planning_horizon,
                          planning_iterations=25, damping=1.0)
    ok = sum(1 for i in range(n_episodes)
             if run_episode(agent, env, seed=seed+i, receding_horizon=True)["success"])
    print(f"  Nuijten 25:   {ok/n_episodes:.0%}")

    sys.stdout.flush()
