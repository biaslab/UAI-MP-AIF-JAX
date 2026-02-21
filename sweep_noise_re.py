#!/usr/bin/env python
"""Test region-extended with high noise (kills epistemic drive) for n_configs=40."""
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
holes = sample_configs(grid_size, n_configs, hole_fraction=0.2, seed=seed)
T = generate_transition_tensor(grid_size, holes, slip_prob=slip_prob)


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


# Wide noise sweep for region-extended
base_noises = [0.05, 0.1, 0.2, 0.3, 0.4]
noise_ranges = [0.0, 0.15, 0.3]
goal_temps = [0.01, 0.05, 0.1]
hole_penalties = [0.0, 5.0, 10.0]

print(f"Grid: {grid_size}x{grid_size}, configs: {n_configs}, damping: 0.25")
print()

results = []
for bn, nr in itertools.product(base_noises, noise_ranges):
    B = generate_observation_tensor(grid_size, holes, base_noise=bn, noise_range=nr)
    for hp, gt in itertools.product(hole_penalties, goal_temps):
        goal = generate_goal(grid_size, holes, hole_penalty=hp, temperature=gt)
        ok = run_eval("region_extended", T, B, goal, 25, 0.25)
        results.append((bn, nr, hp, gt, ok))
        if ok >= 2:
            print(f"*** bn={bn:.2f} nr={nr:.2f} hp={hp:5.1f} gt={gt:.2f} -> {ok*10}%")

print()
results.sort(key=lambda x: -x[4])
print("Top 15 region-extended configs:")
for bn, nr, hp, gt, ok in results[:15]:
    print(f"  bn={bn:.2f} nr={nr:.2f} hp={hp:5.1f} gt={gt:.2f} -> {ok*10}%")

# Also test the best with 30 episodes
print()
if results[0][4] > 0:
    bn, nr, hp, gt, _ = results[0]
    B = generate_observation_tensor(grid_size, holes, base_noise=bn, noise_range=nr)
    goal = generate_goal(grid_size, holes, hole_penalty=hp, temperature=gt)
    ok30 = 0
    agent = create_agent("region_extended", T, B, goal, holes,
                         planning_horizon=15, planning_iterations=25, damping=0.25)
    env = FrozenLakeEnv(grid_size=grid_size, holes=holes, obs_tensor=B,
                        slip_prob=slip_prob, max_steps=max_steps)
    for i in range(30):
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
            ok30 += 1
    print(f"Best config validated (30 ep): bn={bn:.2f} nr={nr:.2f} hp={hp:.1f} gt={gt:.2f} -> {ok30}/30 = {ok30/30:.0%}")
