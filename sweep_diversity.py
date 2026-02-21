#!/usr/bin/env python
"""Lean sweep: bp vs region-extended at various damping levels."""

import sys, os, time, random
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


def run_episode(agent, env, seed, receding_horizon=True):
    result = env.reset(seed=seed)
    agent = agent.reset()
    steps = 0
    max_steps = env.max_steps
    while True:
        time_remaining = max_steps - steps if receding_horizon else agent.planning_horizon
        obs = jnp.array(result.obs)
        action, agent = agent.step(obs, time_remaining)
        result = env.step(action)
        steps += 1
        if result.terminated or result.truncated:
            break
    return result.reward > 0


def evaluate(method_key, T, B, goal, holes, env, iters, damping, n_episodes, seed):
    agent = create_agent(method_key, T, B, goal, holes,
                          planning_horizon=15, planning_iterations=iters,
                          damping=damping)
    successes = sum(run_episode(agent, env, seed=seed+i) for i in range(n_episodes))
    return successes / n_episodes


def main():
    grid_size = 5
    hole_penalty = 0.1
    goal_temp = 0.1
    base_noise = 0.05
    noise_range = 0.15
    n_episodes = 20
    seed = 0

    print(f"grid={grid_size}  hp={hole_penalty}  gt={goal_temp}  noise={base_noise}+{noise_range}  eps={n_episodes}")
    print()

    dampings = [0.5, 0.75, 0.9, 0.95, 1.0]
    header = f"{'n_cfg':>5} {'h_frac':>6} {'slip':>5} {'mh':>3}  {'bp':>5}"
    for d in dampings:
        header += f" {'re_'+str(d):>8}"
    print(header)
    print("-" * len(header))

    for n_configs in [4, 8]:
        for hole_fraction in [0.15, 0.25]:
            for slip_prob in [0.0, 0.1]:
                min_hamming = 4
                t0 = time.time()
                random.seed(seed)
                np.random.seed(seed)

                try:
                    holes = sample_configs(grid_size, n_configs,
                                            hole_fraction=hole_fraction, seed=seed,
                                            min_hamming=min_hamming)
                except ValueError:
                    print(f"{n_configs:>5} {hole_fraction:>6.2f} {slip_prob:>5.2f} {min_hamming:>3}  FAIL")
                    continue

                T = generate_transition_tensor(grid_size, holes, slip_prob=slip_prob)
                B = generate_observation_tensor(grid_size, holes,
                                                 base_noise=base_noise,
                                                 noise_range=noise_range)
                goal = generate_goal(grid_size, holes, hole_penalty=hole_penalty,
                                      temperature=goal_temp)
                env = FrozenLakeEnv(grid_size=grid_size, holes=holes, obs_tensor=B,
                                     slip_prob=slip_prob, max_steps=30)

                bp_rate = evaluate("bp", T, B, goal, holes, env, 1, 1.0, n_episodes, seed)
                row = f"{n_configs:>5} {hole_fraction:>6.2f} {slip_prob:>5.2f} {min_hamming:>3}  {bp_rate:>4.0%}"

                for d in dampings:
                    re_rate = evaluate("region_extended", T, B, goal, holes, env, 25, d, n_episodes, seed)
                    diff = re_rate - bp_rate
                    marker = "*" if diff > 0 else " "
                    row += f" {re_rate:>5.0%}{marker}  "

                elapsed = time.time() - t0
                row += f" ({elapsed:.0f}s)"
                print(row)
                sys.stdout.flush()


if __name__ == "__main__":
    main()
