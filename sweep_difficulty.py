#!/usr/bin/env python
"""Sweep environment difficulty to find where epistemic methods shine."""
import sys
sys.path.insert(0, ".")
import itertools

import jax.numpy as jnp
from environments.frozen_lake import (
    sample_configs, generate_transition_tensor,
    generate_observation_tensor, generate_goal, FrozenLakeEnv,
)
from agents.frozen_lake_agent import create_agent

seed = 0
n_episodes = 20
max_steps = 40
planning_horizon = 15

methods = [
    # (key, label, damping, iters)
    ("bp",               "BP",        1.0,   1),
    ("loopy_vbp",        "LoopyVBP",  1.0,  20),
    ("loopy_bp",         "Loopy",     1.0,  20),
    ("region_extended",  "RegionExt", 0.25, 25),
]

# Difficulty configs: (grid_size, n_configs, hole_fraction, slip_prob, base_noise, noise_range, hp, gt)
difficulty_configs = [
    # Baseline easy
    (4, 3,  0.15, 0.0, 0.05, 0.15, 0.0, 0.1, "4x4 easy"),
    (4, 5,  0.15, 0.0, 0.05, 0.15, 0.0, 0.1, "4x4 nc=5"),
    (4, 10, 0.15, 0.0, 0.05, 0.15, 0.0, 0.1, "4x4 nc=10"),
    (4, 20, 0.15, 0.0, 0.05, 0.15, 0.0, 0.1, "4x4 nc=20"),
    # Add slip
    (4, 5,  0.15, 0.1, 0.05, 0.15, 0.0, 0.1, "4x4 nc=5 slip"),
    (4, 10, 0.15, 0.1, 0.05, 0.15, 0.0, 0.1, "4x4 nc=10 slip"),
    (4, 20, 0.15, 0.1, 0.05, 0.15, 0.0, 0.1, "4x4 nc=20 slip"),
    # More holes
    (4, 10, 0.25, 0.1, 0.05, 0.15, 0.0, 0.1, "4x4 nc=10 more holes"),
    (4, 20, 0.25, 0.1, 0.05, 0.15, 0.0, 0.1, "4x4 nc=20 more holes"),
    # 5x5
    (5, 3,  0.15, 0.0, 0.05, 0.15, 0.0, 0.1, "5x5 easy"),
    (5, 5,  0.15, 0.0, 0.05, 0.15, 0.0, 0.1, "5x5 nc=5"),
    (5, 10, 0.15, 0.0, 0.05, 0.15, 0.0, 0.1, "5x5 nc=10"),
    (5, 5,  0.15, 0.1, 0.05, 0.15, 0.0, 0.1, "5x5 nc=5 slip"),
    (5, 10, 0.15, 0.1, 0.05, 0.15, 0.0, 0.1, "5x5 nc=10 slip"),
    (5, 10, 0.25, 0.1, 0.05, 0.15, 0.0, 0.1, "5x5 nc=10 more holes"),
]


def run_eval(method, T, B, goal, holes, grid_size, slip_prob, iters, damp):
    agent = create_agent(
        method, T, B, goal, holes,
        planning_horizon=planning_horizon, planning_iterations=iters, damping=damp,
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
    return ok


# Header
header = f"{'Config':>25s}"
for _, label, _, _ in methods:
    header += f"  {label:>10s}"
header += "  RE-BP"
print(header)
print("-" * len(header))

for gs, nc, hf, sp, bn, nr, hp, gt, label in difficulty_configs:
    try:
        holes = sample_configs(gs, nc, hole_fraction=hf, seed=seed)
    except ValueError:
        print(f"{label:>25s}  SKIP (can't generate configs)")
        continue

    T = generate_transition_tensor(gs, holes, slip_prob=sp)
    B = generate_observation_tensor(gs, holes, base_noise=bn, noise_range=nr)
    goal = generate_goal(gs, holes, hole_penalty=hp, temperature=gt)

    row = f"{label:>25s}"
    results = {}
    for method_key, mlabel, damp, iters in methods:
        ok = run_eval(method_key, T, B, goal, holes, gs, sp, iters, damp)
        results[mlabel] = ok
        row += f"  {ok*100//n_episodes:>9d}%"

    # RE advantage over BP
    diff = results.get("RegionExt", 0) - results.get("BP", 0)
    diff_pct = diff * 100 // n_episodes
    marker = " <--" if diff_pct > 0 else ""
    row += f"  {diff_pct:>+4d}%{marker}"
    print(row)
