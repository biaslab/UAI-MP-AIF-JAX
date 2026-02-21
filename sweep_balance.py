#!/usr/bin/env python
"""Quick comparison with uninformative-at-holes + best params."""

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

grid_size = 5
n_configs = 4
seed = 0
slip_prob = 0.1
max_steps = 30
planning_horizon = 15
n_episodes = 20


def run_eval(method_key, T, B, goal, holes, iters, damp):
    agent = create_agent(method_key, T, B, goal, holes,
                          planning_horizon=planning_horizon,
                          planning_iterations=iters, damping=damp)
    env = FrozenLakeEnv(grid_size=grid_size, holes=holes, obs_tensor=B,
                         slip_prob=slip_prob, max_steps=max_steps)
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


holes = sample_configs(grid_size, n_configs, hole_fraction=0.2, seed=seed, min_hamming=4)
T = generate_transition_tensor(grid_size, holes, slip_prob=slip_prob)

# Best combo from sweep: bn=0.05, nr=0.00, gt=0.01
# Also test the sweet spot: bn=0.30, nr=0.15, gt=0.05 (closest gap)
configs = [
    (0.05, 0.00, 0.01, 0.1,  "Best RE"),
    (0.05, 0.00, 0.01, 5.0,  "Best RE high hp"),
    (0.30, 0.15, 0.05, 0.1,  "Closest gap"),
    (0.05, 0.15, 0.10, 0.1,  "Current params"),
    (0.05, 0.15, 0.10, 5.0,  "Current+high hp"),
]

methods = [
    ("bp",              "BP",       1.0,  1),
    ("loopy_vbp",       "VBP",      1.0, 20),
    ("loopy_bp",        "Loopy",    1.0, 20),
    ("region_extended", "RE d=0.25",0.25, 25),
    ("dyn_channel",     "DC d=0.25",0.25, 25),
    ("nuijten",         "Nuijten",  1.0,  30),
]

for bn, nr, gt, hp, label in configs:
    B = generate_observation_tensor(grid_size, holes, base_noise=bn, noise_range=nr)
    goal = generate_goal(grid_size, holes, hole_penalty=hp, temperature=gt)

    print(f"\n{label}: bn={bn}, nr={nr}, gt={gt}, hp={hp}")
    for method_key, mname, damp, iters in methods:
        sr = run_eval(method_key, T, B, goal, holes, iters, damp)
        bar = "#" * int(sr * 40)
        print(f"  {mname:>12s}: {sr:>5.0%}  {bar}")
