#!/usr/bin/env python
"""Run RockSample experiments with Active Inference agents."""

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
from environments.rocksample import (
    sample_rock_positions,
    all_quality_configs,
    generate_transition_tensor,
    generate_observation_tensor,
    generate_goal,
    state_index,
    rc_to_pos,
    n_events_for,
    EVENT_OTHER,
    RockSampleEnv,
)
from agents.rocksample_agent import create_agent
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def run_episode(agent, env, seed=None, receding_horizon=False, verbose=False):
    """Run a single RockSample episode."""
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
            print(env.render_ascii())
            print()

        result = env.step(action)
        total_reward += result.reward
        steps += 1

        if result.terminated or result.truncated:
            break

    # Track good-rock collection
    n_good_total = 0
    n_good_collected = 0
    for j in range(env.n_rocks):
        rock_good = env.qualities[env.config_idx, j] == 1.0
        if rock_good:
            n_good_total += 1
            if env._mask & (1 << j):
                n_good_collected += 1

    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": result.terminated and result.reward > 0,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "n_good_total": n_good_total,
        "n_good_collected": n_good_collected,
    }


def main():
    parser = argparse.ArgumentParser(description="Run RockSample experiment")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size (default: 5)")
    parser.add_argument("--n-rocks", type=int, default=3, help="Number of rocks (default: 3)")
    parser.add_argument("--half-eff-dist", type=float, default=2.0, help="Half-efficiency distance for observations")
    parser.add_argument("--pos-noise", type=float, default=0.1, help="Position channel noise")
    parser.add_argument("--slip-prob", type=float, default=0.0, help="Movement slip probability")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps per episode")
    parser.add_argument("--planning-horizon", type=int, default=10, help="Planning horizon")
    parser.add_argument("--planning-iterations", type=int, default=3, help="Planning iterations")
    parser.add_argument("--planning-method", type=str, default="loopy",
                        choices=["loopy-vbp", "loopy", "region-extended",
                                 "dyn-channel", "nuijten", "vbp-channel",
                                 "precise-info-seeking", "active-inference"],
                        help="Planning method")
    parser.add_argument("--damping", type=float, default=1.0, help="Channel update damping (0-1)")
    parser.add_argument("--good-reward", type=float, default=10.0, help="Simulator reward for collecting a good rock")
    parser.add_argument("--bad-penalty", type=float, default=10.0, help="Simulator penalty for collecting a bad rock")
    parser.add_argument("--exit-reward", type=float, default=10.0, help="Simulator reward for reaching exit")
    parser.add_argument("--good-logit", type=float, default=2.0, help="Goal logit per collected good rock")
    parser.add_argument("--bad-logit", type=float, default=4.0, help="Goal logit penalty per collected bad rock")
    parser.add_argument("--exit-logit", type=float, default=2.0, help="Goal logit for the exit column")
    parser.add_argument("--goal-temperature", type=float, default=1.0, help="Goal distribution temperature")
    parser.add_argument("--sense-cost", type=float, default=0.5, help="Per-rock SENSE action prior weight")
    parser.add_argument("--sample-cost", type=float, default=0.5, help="SAMPLE action prior weight")
    parser.add_argument("--receding-horizon", action="store_true", help="Use receding horizon")
    parser.add_argument("--terminal-goal-only", action="store_true",
                        help="Apply goal only at final planning step (not every step)")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    set_seed(args.seed)

    n_configs = 2 ** args.n_rocks

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"\nRockSample[{args.grid_size},{args.n_rocks}]")
    print(f"  Configs (2^k): {n_configs}")
    print(f"  Half-eff dist: {args.half_eff_dist}, pos noise: {args.pos_noise}")
    print(f"  Slip prob: {args.slip_prob}")
    print(f"  Rewards: good={args.good_reward}, bad_penalty={args.bad_penalty}, exit={args.exit_reward}")
    print(f"  Goal logits: good={args.good_logit}, bad={args.bad_logit}, exit={args.exit_logit}")
    print(f"  Goal temperature: {args.goal_temperature}")
    print(f"  Action costs: sense={args.sense_cost}, sample={args.sample_cost}")
    print()

    print("Generating tensors...")
    t0 = time.time()

    n_pos = args.grid_size * args.grid_size
    n_mask = 2 ** args.n_rocks
    n_events = n_events_for(args.n_rocks)
    start_pos = rc_to_pos(args.grid_size // 2, 0, args.grid_size)
    start_state_idx = state_index(start_pos, 0, EVENT_OTHER, n_pos, n_mask, n_events)

    rock_positions = sample_rock_positions(
        args.grid_size, args.n_rocks, seed=args.seed,
    )
    qualities = all_quality_configs(args.n_rocks)
    T = generate_transition_tensor(
        args.grid_size, rock_positions, args.n_rocks,
        slip_prob=args.slip_prob,
    )
    B = generate_observation_tensor(
        args.grid_size, rock_positions, qualities, args.n_rocks,
        half_eff_dist=args.half_eff_dist, pos_noise=args.pos_noise,
    )
    goal = generate_goal(
        args.grid_size, rock_positions, qualities, args.n_rocks,
        good_logit=args.good_logit, bad_logit=args.bad_logit,
        exit_logit=args.exit_logit, temperature=args.goal_temperature,
    )

    print(f"  Rock positions: {rock_positions.tolist()}")
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

    # Action prior: [1]*4 moves + [sense_cost]*k senses + [sample_cost], normalized
    action_prior = np.array(
        [1.0] * 4 + [args.sense_cost] * args.n_rocks + [args.sample_cost],
        dtype=np.float32,
    )
    action_prior = action_prior / action_prior.sum()

    print("Creating agent...")
    agent = create_agent(
        method_key, T, B, goal,
        rock_positions, qualities, n_pos, start_state_idx,
        planning_horizon=args.planning_horizon,
        planning_iterations=args.planning_iterations,
        action_prior=action_prior,
        damping=args.damping,
        terminal_goal_only=args.terminal_goal_only,
    )
    print(f"  Method: {args.planning_method}")
    print(f"  Planning horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    if args.terminal_goal_only:
        print(f"  Terminal goal only: enabled")
    print(f"  Planning iterations: {args.planning_iterations}")
    print()

    env = RockSampleEnv(
        grid_size=args.grid_size,
        rock_positions=rock_positions,
        qualities=qualities,
        n_rocks=args.n_rocks,
        obs_tensor=B,
        slip_prob=args.slip_prob,
        max_steps=args.max_steps,
        good_reward=args.good_reward,
        bad_penalty=args.bad_penalty,
        exit_reward=args.exit_reward,
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
            "reward": f"{episode_result['total_reward']:.1f}",
        })

    elapsed = time.time() - t0

    success_rate = successes / args.episodes
    avg_steps = sum(r["steps"] for r in results) / args.episodes
    rewards = np.array([r["total_reward"] for r in results])
    avg_reward = float(rewards.mean())
    std_reward = float(rewards.std())

    # Good-rock retrieval: only count episodes where good rocks existed
    total_good = sum(r["n_good_total"] for r in results)
    total_good_collected = sum(r["n_good_collected"] for r in results)
    good_rock_retrieval = total_good_collected / total_good if total_good > 0 else float("nan")
    episodes_with_good = sum(1 for r in results if r["n_good_total"] > 0)

    print("-" * 50)
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average reward: {avg_reward:.3f} (std: {std_reward:.3f})")
    print(f"Good rock retrieval: {good_rock_retrieval:.1%} ({total_good_collected}/{total_good} across {episodes_with_good} episodes with good rocks)")
    print(f"Total time: {elapsed:.2f}s ({elapsed / args.episodes * 1000:.1f}ms/episode)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "config": {
                "environment": "rocksample",
                "grid_size": args.grid_size,
                "n_rocks": args.n_rocks,
                "n_configs": n_configs,
                "half_eff_dist": args.half_eff_dist,
                "pos_noise": args.pos_noise,
                "slip_prob": args.slip_prob,
                "good_reward": args.good_reward,
                "bad_penalty": args.bad_penalty,
                "exit_reward": args.exit_reward,
                "good_logit": args.good_logit,
                "bad_logit": args.bad_logit,
                "exit_logit": args.exit_logit,
                "goal_temperature": args.goal_temperature,
                "sense_cost": args.sense_cost,
                "sample_cost": args.sample_cost,
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
                "std_reward": std_reward,
                "good_rock_retrieval": good_rock_retrieval,
                "total_good_rocks": total_good,
                "total_good_collected": total_good_collected,
                "episodes_with_good_rocks": episodes_with_good,
                "successes": successes,
                "total_time_s": elapsed,
            },
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
