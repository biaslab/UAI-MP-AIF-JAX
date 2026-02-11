#!/usr/bin/env python
"""Run MiniGrid experiments with JAX agent."""

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
from environments.minigrid import (
    generate_observation_tensor,
    generate_orientation_observation_tensor,
    generate_transition_tensor,
    generate_transition_indices,
    generate_observation_indices,
    generate_orientation_indices,
)
from environments.gym_wrapper import MiniGridWrapper, run_experiment
from agents.flat_tensor_agent import FlatTensorAgent, IndexedTensorAgent, AIFIndexedTensorAgent
from utils.tensors import get_dimensions, flatten_state_index


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # JAX uses explicit PRNG keys, no global state to seed


def create_goal_distribution(grid_size: int, goal_x: int, goal_y: int) -> jnp.ndarray:
    """
    Create goal distribution - agent should be at goal location with door open.
    
    Goal state: at (goal_x, goal_y), any orientation, door_key_state=2 (door open)
    """
    dims = get_dimensions(grid_size)
    goal = jnp.zeros(dims["n_states"])
    
    goal_location = goal_x * grid_size + goal_y
    
    for orientation in range(dims["n_orientations"]):
        idx = flatten_state_index(
            goal_location,
            orientation,
            2,  # door_key_state=2 means door is open
            dims["n_locations"],
            dims["n_orientations"],
            dims["n_door_key_states"],
        )
        goal = goal.at[idx].set(1.0)
    
    return goal / goal.sum()


def main():
    parser = argparse.ArgumentParser(description="Run MiniGrid experiment")
    parser.add_argument("--grid-size", type=int, default=3, help="Internal grid size (default: 3, corresponds to MiniGrid 5x5)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--planning-horizon", type=int, default=15, help="Planning horizon (lookahead)")
    parser.add_argument("--receding-horizon", action="store_true", help="Use receding horizon (decrease planning horizon as time runs out)")
    parser.add_argument("--inference-iterations", type=int, default=10, help="State inference iterations")
    parser.add_argument("--planning-iterations", type=int, default=10, help="Planning iterations")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--record", type=str, default=None, 
                        help="Record episodes to video. Comma-separated list: 'first', 'last', or indices like '0,9,99'")
    parser.add_argument("--video-dir", type=str, default="data/videos", help="Directory for video output")
    parser.add_argument("--planning-method", type=str, default="bp", choices=["bp", "loopy"],
                        help="Planning method: 'bp' (standard BP, θ marginalized once) or 'loopy' (loopy BP with θ as variable)")
    parser.add_argument("--full-tensors", action="store_true",
                        help="Use full tensor representation (memory-intensive, for testing)")
    args = parser.parse_args()
    
    record_episodes = []
    if args.record:
        for part in args.record.split(","):
            part = part.strip()
            if part == "first":
                record_episodes.append(0)
            elif part == "last":
                record_episodes.append(args.episodes - 1)
            else:
                record_episodes.append(int(part))
    
    set_seed(args.seed)
    
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"Random seed: {args.seed}")

    grid_size = args.grid_size
    minigrid_size = grid_size + 2
    env_name = f"MiniGrid-DoorKey-{minigrid_size}x{minigrid_size}-v0"

    print(f"\nInternal grid size: {grid_size}x{grid_size}")
    print(f"MiniGrid environment: {env_name} (includes outer walls)")
    print()

    print("Generating tensors (this may take a moment)...")
    t0 = time.time()
    
    if args.full_tensors:
        # Full tensor representation (memory-intensive)
        print("  Using FULL tensor representation (memory-intensive)")
        transition_tensor = jnp.array(generate_transition_tensor(grid_size))
        observation_tensor = jnp.array(generate_observation_tensor(grid_size))
        orientation_tensor = jnp.array(generate_orientation_observation_tensor(grid_size))
        print(f"  Transition tensor: {transition_tensor.shape}")
        print(f"  Observation tensor: {observation_tensor.shape}")
        print(f"  Orientation tensor: {orientation_tensor.shape}")
        
        # Calculate memory usage
        trans_mb = transition_tensor.nbytes / 1024 / 1024
        obs_mb = observation_tensor.nbytes / 1024 / 1024
        ori_mb = orientation_tensor.nbytes / 1024 / 1024
        print(f"  Memory: {trans_mb:.1f} + {obs_mb:.1f} + {ori_mb:.1f} = {trans_mb + obs_mb + ori_mb:.1f} MB")
    else:
        # Index-based representation (memory-efficient, default)
        print("  Using INDEX-based representation (memory-efficient)")
        transition_idx = jnp.array(generate_transition_indices(grid_size))
        observation_idx = jnp.array(generate_observation_indices(grid_size))
        orientation_idx = jnp.array(generate_orientation_indices(grid_size))
        print(f"  Transition indices: {transition_idx.shape} (dtype={transition_idx.dtype})")
        print(f"  Observation indices: {observation_idx.shape} (dtype={observation_idx.dtype})")
        print(f"  Orientation indices: {orientation_idx.shape} (dtype={orientation_idx.dtype})")
        
        # Calculate memory usage
        trans_mb = transition_idx.nbytes / 1024 / 1024
        obs_mb = observation_idx.nbytes / 1024 / 1024
        ori_mb = orientation_idx.nbytes / 1024 / 1024
        print(f"  Memory: {trans_mb:.1f} + {obs_mb:.1f} + {ori_mb:.1f} = {trans_mb + obs_mb + ori_mb:.1f} MB")
    
    print(f"  Generated in {time.time() - t0:.2f}s")
    print()

    goal_x = grid_size - 1
    goal_y = 0
    goal = create_goal_distribution(grid_size, goal_x, goal_y)
    print(f"Goal: position ({goal_x}, {goal_y}) with door open")
    print()

    print("Creating agent...")
    if args.full_tensors:
        agent = FlatTensorAgent.create(
            grid_size=grid_size,
            transition_tensor=transition_tensor,
            observation_tensors=observation_tensor,
            orientation_tensor=orientation_tensor,
            goal=goal,
            planning_horizon=args.planning_horizon,
            n_inference_iterations=args.inference_iterations,
            n_planning_iterations=args.planning_iterations,
        )
    elif args.planning_method == "loopy":
        agent = AIFIndexedTensorAgent.create(
            grid_size=grid_size,
            transition_idx=transition_idx,
            observation_idx=observation_idx,
            orientation_idx=orientation_idx,
            goal=goal,
            planning_horizon=args.planning_horizon,
            n_inference_iterations=args.inference_iterations,
            n_planning_iterations=args.planning_iterations,
        )
    else:
        agent = IndexedTensorAgent.create(
            grid_size=grid_size,
            transition_idx=transition_idx,
            observation_idx=observation_idx,
            orientation_idx=orientation_idx,
            goal=goal,
            planning_horizon=args.planning_horizon,
            n_inference_iterations=args.inference_iterations,
            n_planning_iterations=args.planning_iterations,
        )
    print(f"  Planning method: {args.planning_method}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Planning horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    print(f"  Inference iterations: {args.inference_iterations}")
    print(f"  Planning iterations: {args.planning_iterations}")
    print()

    if record_episodes:
        print(f"Recording episodes: {record_episodes}")
        print(f"Video output: {args.video_dir}")
    
    print(f"\nRunning {args.episodes} episodes...")
    print("-" * 50)
    
    t0 = time.time()
    results = run_experiment(
        agent=agent,
        env_name=env_name,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        receding_horizon=args.receding_horizon,
        seed_start=args.seed,
        verbose=args.verbose,
        record_episodes=record_episodes if record_episodes else None,
        video_dir=args.video_dir if record_episodes else None,
    )
    elapsed = time.time() - t0

    print("-" * 50)
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Average steps: {results['avg_steps']:.1f}")
    print(f"Average reward: {results['avg_reward']:.3f}")
    print(f"Total time: {elapsed:.2f}s ({elapsed / args.episodes * 1000:.1f}ms/episode)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "config": {
                "grid_size": grid_size,
                "env_name": env_name,
                "planning_method": args.planning_method,
                "n_episodes": args.episodes,
                "max_steps": args.max_steps,
                "planning_horizon": args.planning_horizon,
                "receding_horizon": args.receding_horizon,
                "inference_iterations": args.inference_iterations,
                "planning_iterations": args.planning_iterations,
                "seed_start": args.seed,
            },
            "results": {
                "success_rate": results["success_rate"],
                "avg_steps": results["avg_steps"],
                "avg_reward": results["avg_reward"],
                "successes": results["successes"],
                "total_time_s": elapsed,
            },
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
