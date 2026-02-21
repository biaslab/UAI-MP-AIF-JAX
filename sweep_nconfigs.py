#!/usr/bin/env python
"""Find the n_configs sweet spot where region-extended breaks down."""
import sys
sys.path.insert(0, ".")

import jax.numpy as jnp
from environments.frozen_lake import (
    sample_configs, generate_transition_tensor,
    generate_observation_tensor, generate_goal, FrozenLakeEnv,
)
from agents.frozen_lake_agent import create_agent

grid_size, seed = 5, 0
n_episodes = 20
slip_prob = 0.1
max_steps = 30
planning_horizon = 15
# Best params from sweeps
base_noise = 0.05
noise_range = 0.0
hole_penalty = 0.0
goal_temp = 0.1


def run_eval(method, T, B, goal, holes, iters, damp):
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


print(f"Grid: {grid_size}x{grid_size}, slip: {slip_prob}")
print(f"bn={base_noise}, nr={noise_range}, hp={hole_penalty}, gt={goal_temp}")
print(f"Episodes: {n_episodes}")
print()

for n_configs in [3, 5, 8, 10, 15, 20, 30, 40, 50]:
    holes = sample_configs(grid_size, n_configs, hole_fraction=0.2, seed=seed)
    T = generate_transition_tensor(grid_size, holes, slip_prob=slip_prob)
    B = generate_observation_tensor(grid_size, holes, base_noise=base_noise,
                                     noise_range=noise_range)
    goal = generate_goal(grid_size, holes, hole_penalty=hole_penalty,
                          temperature=goal_temp)

    bp_ok = run_eval("bp", T, B, goal, holes, 1, 1.0)
    loopy_ok = run_eval("loopy_bp", T, B, goal, holes, 20, 1.0)
    re_ok = run_eval("region_extended", T, B, goal, holes, 25, 0.25)

    print(
        f"  n_configs={n_configs:3d}  BP={bp_ok*5:>3d}%  "
        f"Loopy={loopy_ok*5:>3d}%  RE={re_ok*5:>3d}%"
    )
