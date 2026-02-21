#!/usr/bin/env python
"""Sweep with min_hamming diversity constraint at n_configs=4."""
import sys
sys.path.insert(0, ".")

import numpy as np
import jax.numpy as jnp
from environments.frozen_lake import (
    sample_configs, generate_transition_tensor,
    generate_observation_tensor, generate_goal, FrozenLakeEnv,
)
from agents.frozen_lake_agent import create_agent

grid_size, seed = 5, 0
n_configs = 4
n_episodes = 50
slip_prob = 0.1
max_steps = 30
base_noise = 0.05
noise_range = 0.15


def run_eval(method_key, T, B, goal, holes, damp, iters):
    agent = create_agent(
        method_key, T, B, goal, holes,
        planning_horizon=15, planning_iterations=iters, damping=damp,
    )
    env = FrozenLakeEnv(
        grid_size=grid_size, holes=holes, obs_tensor=B,
        slip_prob=slip_prob, max_steps=max_steps,
    )
    ok = 0
    for i in range(n_episodes):
        result = env.reset(seed=seed + i)
        a = agent.reset()
        steps = 0
        while True:
            obs = jnp.array(result.obs)
            act, a = a.step(obs, max_steps - steps)
            result = env.step(act)
            steps += 1
            if result.terminated or result.truncated:
                break
        if result.reward > 0:
            ok += 1
    return ok / n_episodes


methods = [
    ("bp",               "BP",         1.0,   1),
    ("loopy_vbp",        "VBP",        1.0,  20),
    ("loopy_bp",         "Loopy",      1.0,  20),
    ("region_extended",  "RE d=0.25",  0.25, 25),
    ("dyn_channel",      "DC d=0.25",  0.25, 25),
    ("nuijten",          "Nuijten",    1.0,  30),
]

hps = [0.1, 1.0, 5.0]

for min_hamming in [0, 4, 6]:
    print(f"\n{'='*60}")
    print(f"min_hamming={min_hamming}")
    print(f"{'='*60}")

    holes = sample_configs(grid_size, n_configs, hole_fraction=0.2,
                           seed=seed, min_hamming=min_hamming)
    T = generate_transition_tensor(grid_size, holes, slip_prob=slip_prob)
    B = generate_observation_tensor(grid_size, holes, base_noise=base_noise,
                                    noise_range=noise_range)

    # Show pairwise Hamming distances
    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            d = int(np.sum(holes[i] != holes[j]))
            print(f"  config {i} vs {j}: hamming={d}")

    for hp in hps:
        goal = generate_goal(grid_size, holes, hole_penalty=hp, temperature=0.1)
        print(f"\n  hp={hp:.1f}  gt=0.1")
        for method_key, label, damp, iters in methods:
            sr = run_eval(method_key, T, B, goal, holes, damp, iters)
            bar = "#" * int(sr * 40)
            print(f"    {label:>12s}: {sr:>5.0%}  {bar}")
