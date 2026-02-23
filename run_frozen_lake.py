#!/usr/bin/env python
"""Run Frozen Lake experiments with Active Inference agents."""

import sys
from pathlib import Path
import json
import argparse
import time
import random

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
from environments.frozen_lake import (
    sample_configs,
    generate_transition_tensor,
    generate_observation_tensor,
    generate_goal,
    FrozenLakeEnv,
)
from agents.frozen_lake_agent import create_agent
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def run_episode(agent, env, seed=None, receding_horizon=False, verbose=False):
    """Run a single Frozen Lake episode."""
    result = env.reset(seed=seed)
    agent = agent.reset()

    total_reward = 0.0
    steps = 0
    max_steps = env.max_steps

    while True:
        if receding_horizon:
            time_remaining = max_steps - steps
        else:
            time_remaining = agent.planning_horizon

        obs = jnp.array(result.obs)
        action, agent = agent.step(obs, time_remaining)

        if verbose:
            print(f"Step {steps}: action={action}, time_remaining={time_remaining}")

        result = env.step(action)
        total_reward += result.reward
        steps += 1

        if result.terminated or result.truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": result.reward > 0,
        "terminated": result.terminated,
        "truncated": result.truncated,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Frozen Lake experiment")
    parser.add_argument("--grid-size", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--n-configs", type=int, default=50, help="Number of hole configurations (n_static)")
    parser.add_argument("--hole-fraction", type=float, default=0.2, help="Fraction of cells that are holes")
    parser.add_argument("--min-hamming", type=int, default=0, help="Min pairwise Hamming distance between configs (0=random)")
    parser.add_argument("--base-noise", type=float, default=0.05, help="Base observation noise (at grid center)")
    parser.add_argument("--noise-range", type=float, default=0.15, help="Additional noise at grid edges")
    parser.add_argument("--slip-prob", type=float, default=0.0, help="Movement slip probability")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--planning-horizon", type=int, default=10, help="Planning horizon")
    parser.add_argument("--planning-iterations", type=int, default=3, help="Planning iterations")
    parser.add_argument("--planning-method", type=str, default="bp",
                        choices=["bp", "loopy-vbp", "loopy", "region-extended",
                                 "reduced-region-extended", "dyn-channel",
                                 "reduced-dyn-channel", "nuijten", "reduced-nuijten"],
                        help="Planning method")
    parser.add_argument("--damping", type=float, default=1.0, help="Channel update damping (0-1)")
    parser.add_argument("--hole-penalty", type=float, default=1.0, help="Hole penalty in goal prior")
    parser.add_argument("--goal-temperature", type=float, default=1.0, help="Goal distribution temperature")
    parser.add_argument("--scan-cost", type=float, default=0.5, help="SCAN action prior weight (lower = more costly)")
    parser.add_argument("--receding-horizon", action="store_true", help="Use receding horizon")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"\nFrozen Lake {args.grid_size}x{args.grid_size}")
    print(f"  Configs: {args.n_configs}, hole fraction: {args.hole_fraction}, min_hamming: {args.min_hamming}")
    print(f"  Base noise: {args.base_noise}, noise range: {args.noise_range}")
    print(f"  Slip prob: {args.slip_prob}")
    print(f"  Hole penalty: {args.hole_penalty}, goal temperature: {args.goal_temperature}")
    print(f"  Scan cost: {args.scan_cost}")
    print()

    print("Generating tensors...")
    t0 = time.time()

    holes = sample_configs(
        args.grid_size, args.n_configs,
        hole_fraction=args.hole_fraction, seed=args.seed,
        min_hamming=args.min_hamming,
    )
    T = generate_transition_tensor(args.grid_size, holes, slip_prob=args.slip_prob)
    B = generate_observation_tensor(args.grid_size, holes, base_noise=args.base_noise,
                                    noise_range=args.noise_range)
    goal = generate_goal(args.grid_size, holes, hole_penalty=args.hole_penalty,
                         temperature=args.goal_temperature)

    print(f"  Transition tensor: {T.shape} ({T.nbytes / 1024:.1f} KB)")
    print(f"  Observation tensor: {B.shape} ({B.nbytes / 1024:.1f} KB)")
    print(f"  Generated in {time.time() - t0:.2f}s")
    print()

    # Map CLI names (hyphens) to agent keys (underscores)
    METHOD_MAP = {
        "bp": "bp",
        "loopy-vbp": "loopy_vbp",
        "loopy": "loopy_bp",
        "region-extended": "region_extended",
        "reduced-region-extended": "reduced_region_extended",
        "dyn-channel": "dyn_channel",
        "reduced-dyn-channel": "reduced_dyn_channel",
        "nuijten": "nuijten",
        "reduced-nuijten": "reduced_nuijten",
    }
    method_key = METHOD_MAP[args.planning_method]

    # Construct action prior: [1, 1, 1, 1, scan_cost] normalized
    action_prior = np.array([1.0, 1.0, 1.0, 1.0, args.scan_cost], dtype=np.float32)
    action_prior = action_prior / action_prior.sum()

    print("Creating agent...")
    agent = create_agent(
        method_key, T, B, goal, holes,
        planning_horizon=args.planning_horizon,
        planning_iterations=args.planning_iterations,
        action_prior=action_prior,
        damping=args.damping,
    )
    print(f"  Method: {args.planning_method}")
    print(f"  Planning horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    print(f"  Planning iterations: {args.planning_iterations}")
    print()

    env = FrozenLakeEnv(
        grid_size=args.grid_size, holes=holes, obs_tensor=B,
        slip_prob=args.slip_prob, max_steps=args.max_steps,
    )

    print(f"Running {args.episodes} episodes...")
    print("-" * 50)

    results = []
    successes = 0
    t0 = time.time()

    pbar = tqdm(range(args.episodes), desc="Episodes")
    for i in pbar:
        seed = args.seed + i
        episode_result = run_episode(
            agent, env, seed=seed, receding_horizon=args.receding_horizon,
            verbose=args.verbose,
        )
        results.append(episode_result)
        if episode_result["success"]:
            successes += 1
        pbar.set_postfix({
            "success": f"{successes / (i + 1):.1%}",
            "steps": episode_result["steps"],
        })

    elapsed = time.time() - t0

    success_rate = successes / args.episodes
    avg_steps = sum(r["steps"] for r in results) / args.episodes
    avg_reward = sum(r["total_reward"] for r in results) / args.episodes

    print("-" * 50)
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Total time: {elapsed:.2f}s ({elapsed / args.episodes * 1000:.1f}ms/episode)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "config": {
                "environment": "frozen_lake",
                "grid_size": args.grid_size,
                "n_configs": args.n_configs,
                "hole_fraction": args.hole_fraction,
                "base_noise": args.base_noise,
                "noise_range": args.noise_range,
                "slip_prob": args.slip_prob,
                "hole_penalty": args.hole_penalty,
                "goal_temperature": args.goal_temperature,
                "scan_cost": args.scan_cost,
                "planning_method": args.planning_method,
                "n_episodes": args.episodes,
                "max_steps": args.max_steps,
                "planning_horizon": args.planning_horizon,
                "planning_iterations": args.planning_iterations,
                "receding_horizon": args.receding_horizon,
                "seed_start": args.seed,
            },
            "results": {
                "success_rate": success_rate,
                "avg_steps": avg_steps,
                "avg_reward": avg_reward,
                "successes": successes,
                "total_time_s": elapsed,
            },
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
