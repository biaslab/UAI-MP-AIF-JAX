#!/usr/bin/env python
"""Focused sweep: damping=0.25 fixed, vary noise + goal params for n_configs=40."""
import sys
sys.path.insert(0, ".")
import itertools

import jax.numpy as jnp
from environments.frozen_lake import (
    sample_configs, generate_transition_tensor,
    generate_observation_tensor, generate_goal, FrozenLakeEnv,
)
from agents.frozen_lake_agent import create_agent

grid_size, n_configs, seed = 5, 40, 0
n_episodes = 10
slip_prob = 0.1
max_steps = 30
planning_horizon = 15
damping = 0.25

holes = sample_configs(grid_size, n_configs, hole_fraction=0.2, seed=seed)

# Sweep axes
base_noises = [0.05, 0.1, 0.2, 0.3]
noise_ranges = [0.0, 0.15, 0.3]
hole_penalties = [0.0, 1.0, 5.0, 10.0]
goal_temps = [0.01, 0.05, 0.1, 0.5]

methods = [
    ("bp", 1, 1.0),
    ("region_extended", 25, 0.25),
]


def run_eval(method, T, B, goal, iters, damp):
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


T = generate_transition_tensor(grid_size, holes, slip_prob=slip_prob)

print(f"Grid: {grid_size}x{grid_size}, configs: {n_configs}, slip: {slip_prob}")
print(f"Damping: {damping} (region-extended only)")
print(f"Episodes: {n_episodes}")
print()

# First pass: sweep with BP to find which noise/goal combos work at all
print("=" * 90)
print("BP BASELINE (find good noise + goal)")
print("=" * 90)
best_bp = []
for bn, nr in itertools.product(base_noises, noise_ranges):
    B = generate_observation_tensor(grid_size, holes, base_noise=bn, noise_range=nr)
    for hp, gt in itertools.product(hole_penalties, goal_temps):
        goal = generate_goal(grid_size, holes, hole_penalty=hp, temperature=gt)
        ok = run_eval("bp", T, B, goal, 1, 1.0)
        best_bp.append((bn, nr, hp, gt, ok))
        if ok >= 4:
            print(f"  *** bn={bn:.2f} nr={nr:.2f} hp={hp:5.1f} gt={gt:.2f} -> {ok*10}%")

best_bp.sort(key=lambda x: -x[4])
print(f"\nTop BP configs:")
for bn, nr, hp, gt, ok in best_bp[:10]:
    print(f"  bn={bn:.2f} nr={nr:.2f} hp={hp:5.1f} gt={gt:.2f} -> {ok*10}%")

# Second pass: test region-extended on the top BP configs
print()
print("=" * 90)
print("REGION-EXTENDED (top BP configs)")
print("=" * 90)
top_configs = best_bp[:20]
re_results = []
for bn, nr, hp, gt, bp_ok in top_configs:
    B = generate_observation_tensor(grid_size, holes, base_noise=bn, noise_range=nr)
    goal = generate_goal(grid_size, holes, hole_penalty=hp, temperature=gt)
    ok = run_eval("region_extended", T, B, goal, 25, damping)
    re_results.append((bn, nr, hp, gt, ok, bp_ok))
    tag = "***" if ok > 0 else "   "
    print(f"  {tag} bn={bn:.2f} nr={nr:.2f} hp={hp:5.1f} gt={gt:.2f} -> RE={ok*10}% (BP={bp_ok*10}%)")

print()
print("=" * 90)
print("BEST REGION-EXTENDED")
print("=" * 90)
re_results.sort(key=lambda x: -x[4])
for bn, nr, hp, gt, ok, bp_ok in re_results[:10]:
    print(f"  bn={bn:.2f} nr={nr:.2f} hp={hp:5.1f} gt={gt:.2f} -> RE={ok*10}% (BP={bp_ok*10}%)")
