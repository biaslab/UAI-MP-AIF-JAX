#!/usr/bin/env python
"""Run Wumpus World experiments with Active Inference agents."""

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
from environments.wumpus_world import (
    sample_configs,
    generate_transition_tensor,
    generate_observation_tensor,
    generate_goal,
    WumpusWorldEnv,
)
from agents.wumpus_agent import create_agent
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def run_episode(agent, env, seed=None, receding_horizon=False, verbose=False, record=False):
    """Run a single Wumpus World episode.

    When ``record`` is True, also returns a trajectory dict with per-step
    arrays (positions, sensed, actions, observations, rewards) plus the
    active config index and termination flags. ``positions``/``sensed``/
    ``observations`` are length T+1 (initial state plus post-step); ``actions``
    and ``rewards`` are length T.
    """
    result = env.reset(seed=seed)
    agent = agent.reset()

    total_reward = 0.0
    steps = 0
    max_steps = env.max_steps

    if record:
        positions = [int(env._position)]
        sensed = [int(env._sensed)]
        observations = [np.asarray(result.obs, dtype=np.float32)]
        actions = []
        rewards = []
        config_idx = int(env.config_idx)
    else:
        positions = sensed = observations = actions = rewards = None
        config_idx = None

    while True:
        if receding_horizon:
            time_remaining = max_steps - steps
        else:
            time_remaining = agent.planning_horizon

        obs = jnp.array(result.obs)
        action, agent = agent.step(obs, time_remaining)

        if verbose:
            print(f"Step {steps}: action={action}, obs={result.obs}, time_remaining={time_remaining}")

        result = env.step(action)
        total_reward += result.reward
        steps += 1

        if record:
            actions.append(int(action))
            rewards.append(float(result.reward))
            positions.append(int(env._position))
            sensed.append(int(env._sensed))
            observations.append(np.asarray(result.obs, dtype=np.float32))

        if result.terminated or result.truncated:
            break

    summary = {
        "total_reward": total_reward,
        "steps": steps,
        "success": result.reward > 0,
        "terminated": result.terminated,
        "truncated": result.truncated,
    }

    if not record:
        return summary, None

    trajectory = {
        "positions": np.asarray(positions, dtype=np.int32),
        "sensed": np.asarray(sensed, dtype=np.int32),
        "actions": np.asarray(actions, dtype=np.int32),
        "observations": np.stack(observations, axis=0),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": bool(result.terminated),
        "truncated": bool(result.truncated),
        "config_idx": config_idx,
    }
    return summary, trajectory


def main():
    parser = argparse.ArgumentParser(description="Run Wumpus World experiment")
    parser.add_argument("--grid-size", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--n-configs", type=int, default=50, help="Number of configurations (n_static)")
    parser.add_argument("--n-pits", type=int, default=2, help="Number of pits per configuration")
    parser.add_argument("--obs-noise", type=float, default=0.1, help="Observation noise level")
    parser.add_argument("--pos-noise", type=float, default=0.1, help="Position channel noise level")
    parser.add_argument("--slip-prob", type=float, default=0.0, help="Movement slip probability")
    parser.add_argument("--sense-cost", type=float, default=0.5, help="SENSE action prior weight (lower = more costly)")
    parser.add_argument("--pit-penalty", type=float, default=2.0, help="Pit penalty magnitude for goal")
    parser.add_argument("--wumpus-penalty", type=float, default=2.0, help="Wumpus penalty magnitude for goal")
    parser.add_argument("--goal-temperature", type=float, default=1.0, help="Softmax temperature for goal")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--planning-horizon", type=int, default=10, help="Planning horizon")
    parser.add_argument("--planning-iterations", type=int, default=3, help="Planning iterations")
    parser.add_argument("--planning-method", type=str, default="loopy",
                        choices=["loopy-vbp", "loopy", "region-extended",
                                 "dyn-channel", "nuijten", "vbp-channel",
                                 "precise-info-seeking", "active-inference"],
                        help="Planning method")
    parser.add_argument("--damping", type=float, default=1.0, help="Channel update damping (0-1)")
    parser.add_argument("--receding-horizon", action="store_true", help="Use receding horizon")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--record-trajectories", type=int, default=0,
                        help="Number of leading episodes to dump as per-step NPZ trajectories")
    parser.add_argument("--trajectory-dir", type=str, default=None,
                        help="Directory for per-episode trajectory NPZ files")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"\nWumpus World {args.grid_size}x{args.grid_size}")
    print(f"  Configs: {args.n_configs}, pits: {args.n_pits}")
    print(f"  Obs noise: {args.obs_noise}, pos noise: {args.pos_noise}, slip prob: {args.slip_prob}")
    print(f"  Sense cost: {args.sense_cost}")
    print()

    print("Generating tensors...")
    t0 = time.time()

    pits, wumpus_arr, gold = sample_configs(
        args.grid_size, args.n_configs, n_pits=args.n_pits, seed=args.seed,
    )
    T = generate_transition_tensor(args.grid_size, pits, wumpus_arr, slip_prob=args.slip_prob)
    B = generate_observation_tensor(
        args.grid_size, pits, wumpus_arr, gold, obs_noise=args.obs_noise,
        pos_noise=args.pos_noise,
    )
    goal = generate_goal(
        args.grid_size, pits, wumpus_arr, gold,
        gold_reward=1.0, pit_penalty=args.pit_penalty,
        wumpus_penalty=args.wumpus_penalty, temperature=args.goal_temperature,
    )

    print(f"  Transition tensor: {T.shape} ({T.nbytes / 1024:.1f} KB)")
    print(f"  Observation tensor: {B.shape} ({B.nbytes / 1024:.1f} KB)")
    print(f"  Generated in {time.time() - t0:.2f}s")
    print()

    # Map CLI names (hyphens) to agent keys (underscores)
    METHOD_MAP = {
        "loopy-vbp": "loopy_vbp",
        "loopy": "loopy_bp",
        "region-extended": "region_extended",
        "dyn-channel": "dyn_channel",
        "nuijten": "nuijten",
        "vbp-channel": "vbp_channel",
        "precise-info-seeking": "precise_info_seeking",
        "active-inference": "active_inference",
    }
    method_key = METHOD_MAP[args.planning_method]

    # Construct action prior: [1, 1, 1, 1, sense_cost] normalized
    action_prior = np.array([1.0, 1.0, 1.0, 1.0, args.sense_cost], dtype=np.float32)
    action_prior = action_prior / action_prior.sum()

    print("Creating agent...")
    agent = create_agent(
        method_key, T, B, goal,
        planning_horizon=args.planning_horizon,
        planning_iterations=args.planning_iterations,
        action_prior=action_prior,
        damping=args.damping,
    )
    print(f"  Method: {args.planning_method}")
    print(f"  Planning horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    print(f"  Planning iterations: {args.planning_iterations}")
    print()

    env = WumpusWorldEnv(
        grid_size=args.grid_size, pits=pits, wumpus=wumpus_arr, gold=gold,
        obs_tensor=B, slip_prob=args.slip_prob, max_steps=args.max_steps,
    )

    print(f"Running {args.episodes} episodes...")
    print("-" * 50)

    record_n = max(0, int(args.record_trajectories))
    trajectory_dir = None
    if record_n > 0 and args.trajectory_dir:
        trajectory_dir = Path(args.trajectory_dir)
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        print(f"Recording trajectories for first {record_n} episodes -> {trajectory_dir}")

    results = []
    successes = 0
    t0 = time.time()

    pbar = tqdm(range(args.episodes), desc="Episodes")
    for i in pbar:
        seed = args.seed + i
        record_this = trajectory_dir is not None and i < record_n
        episode_result, trajectory = run_episode(
            agent, env, seed=seed, receding_horizon=args.receding_horizon,
            verbose=args.verbose, record=record_this,
        )
        results.append(episode_result)
        if episode_result["success"]:
            successes += 1

        if trajectory is not None:
            theta = trajectory["config_idx"]
            np.savez_compressed(
                trajectory_dir / f"episode_{i:03d}.npz",
                positions=trajectory["positions"],
                sensed=trajectory["sensed"],
                actions=trajectory["actions"],
                observations=trajectory["observations"],
                rewards=trajectory["rewards"],
                terminated=np.bool_(trajectory["terminated"]),
                truncated=np.bool_(trajectory["truncated"]),
                config_idx=np.int32(theta),
                pits=np.asarray(pits[theta], dtype=np.float32),
                wumpus=np.asarray(wumpus_arr[theta], dtype=np.float32),
                gold=np.asarray(gold[theta], dtype=np.float32),
                grid_size=np.int32(args.grid_size),
                seed=np.int32(seed),
                method=np.str_(args.planning_method),
            )

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
                "environment": "wumpus_world",
                "grid_size": args.grid_size,
                "n_configs": args.n_configs,
                "n_pits": args.n_pits,
                "obs_noise": args.obs_noise,
                "pos_noise": args.pos_noise,
                "slip_prob": args.slip_prob,
                "sense_cost": args.sense_cost,
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
