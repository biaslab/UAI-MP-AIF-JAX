#!/usr/bin/env python
"""Unified convergence sweep: systematic damping × method × seed experiments.

Runs convergence analysis for all 4 environments, 6 planning methods, multiple
damping values, and multiple seeds. Outputs per-run JSON and a summary CSV.

Usage:
    # Quick test: one env, one method, one seed
    JAX_PLATFORMS=cpu uv run python run_convergence_sweep.py \
      --environment frozen-lake --methods region-extended --damping 0.25 0.5 \
      --seeds 0 --n-iterations 10 --planning-horizon 5

    # Full frozen lake sweep
    JAX_PLATFORMS=cpu uv run python run_convergence_sweep.py \
      --environment frozen-lake --n-iterations 200

    # All environments use the same interface
    JAX_PLATFORMS=cpu uv run python run_convergence_sweep.py \
      --environment wumpus-world --methods region-extended --damping 0.25 --seeds 0
"""

import sys
from pathlib import Path
import argparse
import json
import time
import random
from collections import namedtuple

import yaml

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp

from inference.convergence import (
    loopy_bp_convergence,
    region_extended_convergence,
    dyn_channel_convergence,
    nuijten_mp_convergence,
    vbp_channel_convergence,
    precise_info_seeking_convergence,
    active_inference_convergence,
)

ConvergenceSetup = namedtuple('ConvergenceSetup', [
    'T', 'B', 'goal', 'q_current', 'q_static', 'action_prior',
])

ALL_METHODS = [
    "loopy", "loopy-vbp", "region-extended", "dyn-channel",
    "nuijten", "vbp-channel", "precise-info-seeking", "active-inference",
]

METHODS_WITH_DAMPING = {"region-extended", "dyn-channel", "loopy-vbp", "vbp-channel", "precise-info-seeking", "active-inference"}
METHODS_NO_DAMPING = {"loopy", "nuijten"}

CONVERGENCE_FUNCS = {
    "loopy": loopy_bp_convergence,
    "loopy-vbp": vbp_channel_convergence,
    "region-extended": region_extended_convergence,
    "dyn-channel": dyn_channel_convergence,
    "nuijten": nuijten_mp_convergence,
    "vbp-channel": vbp_channel_convergence,
    "precise-info-seeking": precise_info_seeking_convergence,
    "active-inference": active_inference_convergence,
}

# Methods that need observation_tensor argument
METHODS_WITH_OBS = {
    "region-extended", "dyn-channel", "nuijten",
    "loopy-vbp", "vbp-channel", "precise-info-seeking", "active-inference",
}


# =============================================================================
# Environment setup functions
# =============================================================================


def setup_frozen_lake(args, seed):
    """Generate tensors and initial beliefs for Frozen Lake."""
    from environments.frozen_lake import (
        sample_configs,
        generate_transition_tensor,
        generate_observation_tensor,
        generate_goal,
    )

    grid_size = args.grid_size
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos  # doubled for scan mode

    holes = sample_configs(
        grid_size, args.n_configs,
        hole_fraction=args.hole_fraction, seed=seed,
        min_hamming=args.min_hamming,
    )
    T = jnp.array(
        generate_transition_tensor(grid_size, holes, slip_prob=args.slip_prob),
        dtype=jnp.float32,
    )
    B = jnp.array(
        generate_observation_tensor(
            grid_size, holes,
            base_noise=args.base_noise, noise_range=args.noise_range,
        ),
        dtype=jnp.float32,
    )
    goal = jnp.array(
        generate_goal(
            grid_size, holes,
            hole_penalty=args.hole_penalty,
            temperature=args.goal_temperature,
        ),
        dtype=jnp.float32,
    )

    # Directional channels only (last n_pos of B)
    B_dir = B[n_states:]

    # Initial beliefs: known start at position 0
    q_current = jnp.zeros(n_states, dtype=jnp.float32).at[0].set(1.0)
    n_static = holes.shape[0]
    q_static = jnp.ones(n_static, dtype=jnp.float32) / n_static

    # Action prior with scan cost
    action_prior = np.array(
        [1.0, 1.0, 1.0, 1.0, args.scan_cost], dtype=np.float32,
    )
    action_prior = action_prior / action_prior.sum()
    action_prior = jnp.array(action_prior)

    return ConvergenceSetup(T, B_dir, goal, q_current, q_static, action_prior)


def setup_wumpus_world(args, seed):
    """Generate tensors and initial beliefs for Wumpus World."""
    from environments.wumpus_world import (
        sample_configs,
        generate_transition_tensor,
        generate_observation_tensor,
        generate_goal,
    )

    grid_size = args.grid_size
    n_pos = grid_size * grid_size
    n_states = 2 * n_pos  # doubled for scan mode

    pits, wumpus_arr, gold = sample_configs(
        grid_size, args.n_configs, n_pits=args.n_pits, seed=seed,
    )
    T = jnp.array(
        generate_transition_tensor(grid_size, pits, wumpus_arr, slip_prob=args.slip_prob),
        dtype=jnp.float32,
    )
    B = jnp.array(
        generate_observation_tensor(
            grid_size, pits, wumpus_arr, gold,
            obs_noise=args.obs_noise, pos_noise=args.pos_noise,
        ),
        dtype=jnp.float32,
    )
    goal = jnp.array(
        generate_goal(
            grid_size, pits, wumpus_arr, gold,
            pit_penalty=args.pit_penalty, wumpus_penalty=args.wumpus_penalty,
            temperature=args.goal_temperature,
        ),
        dtype=jnp.float32,
    )

    # Initial beliefs: uniform over doubled state space
    n_static = pits.shape[0]
    q_current = jnp.ones(n_states, dtype=jnp.float32) / n_states
    q_static = jnp.ones(n_static, dtype=jnp.float32) / n_static

    # Action prior with scan cost
    action_prior = np.array(
        [1.0, 1.0, 1.0, 1.0, args.scan_cost], dtype=np.float32,
    )
    action_prior = action_prior / action_prior.sum()
    action_prior = jnp.array(action_prior)

    return ConvergenceSetup(T, B, goal, q_current, q_static, action_prior)


def setup_rocksample(args, seed):
    """Generate tensors and initial beliefs for RockSample."""
    from environments.rocksample import (
        sample_rock_positions,
        all_quality_configs,
        generate_transition_tensor,
        generate_observation_tensor,
        generate_goal,
        rc_to_pos,
        state_index,
    )

    grid_size = args.grid_size
    n_rocks = args.n_rocks
    n_pos = grid_size * grid_size
    n_collect = 2 ** n_rocks
    n_scan = 2 ** n_rocks
    n_states = n_pos * n_collect * n_scan

    start_pos = rc_to_pos(grid_size // 2, 0, grid_size)
    start_state_idx = state_index(start_pos, 0, 0, n_pos, n_collect, n_scan)

    rock_positions = sample_rock_positions(grid_size, n_rocks, seed=seed)
    qualities = all_quality_configs(n_rocks)

    T = jnp.array(
        generate_transition_tensor(grid_size, rock_positions, n_rocks, slip_prob=args.slip_prob),
        dtype=jnp.float32,
    )
    B = jnp.array(
        generate_observation_tensor(
            grid_size, rock_positions, qualities, n_rocks,
            half_eff_dist=args.half_eff_dist, pos_noise=args.pos_noise,
        ),
        dtype=jnp.float32,
    )
    goal = jnp.array(
        generate_goal(
            grid_size, rock_positions, qualities, n_rocks,
            exit_reward=args.exit_reward, good_reward=args.good_reward,
            bad_penalty=args.bad_penalty, temperature=args.goal_temperature,
        ),
        dtype=jnp.float32,
    )

    # Rock-quality observation channels only (θ-dependent, skip position channels)
    B_rock = B[n_pos:]

    # Initial beliefs: known start state
    q_current = jnp.zeros(n_states, dtype=jnp.float32).at[start_state_idx].set(1.0)
    n_static = n_collect  # 2^n_rocks quality configs
    q_static = jnp.ones(n_static, dtype=jnp.float32) / n_static

    # Action prior: [left, down, right, up, scan, sample]
    action_prior = np.array(
        [1.0, 1.0, 1.0, 1.0, args.scan_cost, args.sample_cost],
        dtype=np.float32,
    )
    action_prior = action_prior / action_prior.sum()
    action_prior = jnp.array(action_prior)

    return ConvergenceSetup(T, B_rock, goal, q_current, q_static, action_prior)


def setup_minigrid(args, seed):
    """Generate tensors and initial beliefs for MiniGrid DoorKey.

    Uses --observe-first: creates a MiniGridWrapper with the given seed,
    takes 1 observation + inference step to get non-uniform initial beliefs.
    """
    from environments.minigrid import (
        get_valid_static_configs,
        generate_transition_tensor,
        generate_observation_tensor,
        generate_orientation_observation_tensor,
        soften_observation_tensor,
    )
    from environments.gym_wrapper import MiniGridWrapper, StepResult
    from agents.flat_tensor_agent import (
        LoopyBPAgent, RegionExtendedAgent,
        DynChannelLoopyBPAgent, NuijtenMPAgent, VBPChannelAgent,
        PreciseInfoSeekingAgent, ActiveInferenceAgent,
    )
    from inference.state_inference import state_inference_step
    from utils.tensors import get_dimensions, flatten_state_index

    grid_size = args.grid_size
    fov_size = args.fov_size
    obs_alpha = args.obs_alpha

    minigrid_size = grid_size + 2
    env_name = f"MiniGrid-DoorKey-{minigrid_size}x{minigrid_size}-v0"
    valid_configs = get_valid_static_configs(grid_size)
    dims = get_dimensions(grid_size, n_static_override=len(valid_configs))

    transition_tensor = jnp.array(
        generate_transition_tensor(grid_size, valid_configs), dtype=jnp.float32,
    )
    obs_np = generate_observation_tensor(grid_size, valid_configs, fov_size=fov_size)
    if obs_alpha > 0.0:
        obs_np = soften_observation_tensor(obs_np, fov_size, obs_alpha)
    observation_tensor = jnp.array(obs_np, dtype=jnp.float32)
    orientation_tensor = jnp.array(
        generate_orientation_observation_tensor(grid_size), dtype=jnp.float32,
    )

    # Goal: reach (grid_size-1, grid_size-1) with key collected + door open
    goal_x = grid_size - 1
    goal_y = grid_size - 1
    goal = jnp.zeros(dims["n_states"])
    goal_location = goal_x * grid_size + goal_y
    for orientation in range(dims["n_orientations"]):
        idx = flatten_state_index(
            goal_location, orientation, 2,
            dims["n_locations"], dims["n_orientations"], dims["n_door_key_states"],
        )
        goal = goal.at[idx].set(1.0)
    goal = goal / goal.sum()

    # Create a temporary agent for state inference
    agent = LoopyBPAgent.create(
        grid_size=grid_size,
        transition_tensor=transition_tensor,
        observation_tensors=observation_tensor,
        orientation_tensor=orientation_tensor,
        goal=goal,
        planning_horizon=args.planning_horizon,
        n_inference_iterations=10,
        n_planning_iterations=10,
    )

    q_state = agent.q_state
    q_static = agent.q_static

    # Observe first: take 1 observation + inference step for non-uniform beliefs
    if args.observe_first:
        env = MiniGridWrapper(
            env_name=env_name, max_steps=100,
            fov_size=fov_size, obs_alpha=obs_alpha,
        )
        result = env.reset(seed=seed)

        action_onehot = jnp.zeros(dims["n_actions"]).at[0].set(1.0)
        q_state, q_static = state_inference_step(
            q_old_state=q_state,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            obs_tensors=agent.observation_tensors,
            ori_tensor=agent.orientation_tensor,
            vision_obs=result.vision_obs,
            ori_obs=result.orientation_obs,
            action_onehot=action_onehot,
            n_iterations=10,
        )
        q_state.block_until_ready()
        env.close()

    # Flatten observation tensor for planning: 5D→4D
    obs_flat = observation_tensor
    if obs_flat.ndim == 5:
        obs_flat = obs_flat.reshape(
            obs_flat.shape[0] * obs_flat.shape[1], *obs_flat.shape[2:],
        )

    # MiniGrid has no scan/sample cost → no action prior
    return ConvergenceSetup(transition_tensor, obs_flat, goal, q_state, q_static, None)


# =============================================================================
# Convergence dispatch
# =============================================================================


def call_convergence(method, setup, horizon, n_iterations, damping=1.0):
    """Dispatch to the correct convergence function.

    Returns (action_dist, vfe_trace).
    """
    func = CONVERGENCE_FUNCS[method]

    kwargs = dict(
        q_current_state=setup.q_current,
        q_static_state=setup.q_static,
        transition_tensor=setup.T,
        goal=setup.goal,
        horizon=horizon,
        n_iterations=n_iterations,
    )

    if setup.action_prior is not None:
        kwargs["action_prior"] = setup.action_prior

    if method in METHODS_WITH_OBS:
        kwargs["observation_tensor"] = setup.B

    if method in METHODS_WITH_DAMPING:
        kwargs["damping"] = damping

    result = func(**kwargs)
    # All return (action_dist, ..., vfe_trace)
    return result[0], result[-1]


# =============================================================================
# Output helpers
# =============================================================================


def save_result_json(output_dir, env_name, method, damping, seed, args,
                     vfe_trace, action_dist, elapsed):
    """Save a single run result as JSON."""
    method_dir = output_dir / env_name / method / f"damping_{damping}"
    method_dir.mkdir(parents=True, exist_ok=True)

    vfe_values = np.array(vfe_trace).tolist()

    # Convergence check
    converged = False
    convergence_iteration = -1
    final_delta = float("nan")
    if len(vfe_values) >= 2:
        final_delta = abs(vfe_values[-1] - vfe_values[-2])
        converged = final_delta < 1e-4
        # Find first iteration where convergence criterion is met
        for i in range(1, len(vfe_values)):
            if abs(vfe_values[i] - vfe_values[i - 1]) < 1e-4:
                convergence_iteration = i
                break

    data = {
        "config": {
            "environment": env_name,
            "method": method,
            "damping": damping,
            "seed": seed,
            "horizon": args.planning_horizon,
            "n_iterations": args.n_iterations,
            "grid_size": args.grid_size,
        },
        "results": {
            "vfe_trace": vfe_values,
            "action_dist": np.array(action_dist).tolist(),
            "converged": converged,
            "convergence_iteration": convergence_iteration,
            "final_vfe": vfe_values[-1] if vfe_values else float("nan"),
            "final_delta": final_delta,
            "elapsed_s": elapsed,
        },
    }

    json_path = method_dir / f"seed_{seed}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def write_summary_csv(output_dir, env_name, all_results):
    """Write aggregated summary CSV for one environment."""
    csv_path = output_dir / env_name / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w") as f:
        f.write("method,damping,seed,final_vfe,final_delta,converged,"
                "convergence_iteration,elapsed_s\n")
        for r in all_results:
            f.write(
                f"{r['method']},{r['damping']},{r['seed']},"
                f"{r['final_vfe']:.8f},{r['final_delta']:.8f},"
                f"{r['converged']},{r['convergence_iteration']},"
                f"{r['elapsed']:.2f}\n"
            )

    print(f"Summary CSV saved to {csv_path}")


# =============================================================================
# Environment-specific argument defaults
# =============================================================================


ENV_DEFAULTS = {
    "frozen-lake": dict(
        grid_size=4, n_configs=15, hole_fraction=0.2, min_hamming=4,
        base_noise=0.4, noise_range=0.1, slip_prob=0.1,
        planning_horizon=15, hole_penalty=2.0, goal_temperature=1.0,
        scan_cost=0.1,
    ),
    "wumpus-world": dict(
        grid_size=5, n_configs=25, n_pits=4,
        obs_noise=0.1, pos_noise=0.4, slip_prob=0.01,
        planning_horizon=7, pit_penalty=1.0, wumpus_penalty=1.0,
        goal_temperature=1.0, scan_cost=0.7,
    ),
    "rocksample": dict(
        grid_size=5, n_rocks=2, half_eff_dist=2.0, pos_noise=0.3,
        slip_prob=0.0, planning_horizon=15,
        good_reward=2.0, bad_penalty=3.0, exit_reward=1.0,
        goal_temperature=0.5, scan_cost=0.5, sample_cost=1.0,
        terminal_goal_only=True,
    ),
    "minigrid": dict(
        grid_size=3, planning_horizon=15, fov_size=7,
        obs_alpha=0.0, observe_first=True,
    ),
}

SETUP_FUNCS = {
    "frozen-lake": setup_frozen_lake,
    "wumpus-world": setup_wumpus_world,
    "rocksample": setup_rocksample,
    "minigrid": setup_minigrid,
}


def load_sweep_params():
    """Load convergence_sweep config from params.yaml."""
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("convergence_sweep", {})


# =============================================================================
# Main
# =============================================================================


def main():
    sweep_cfg = load_sweep_params()
    parser = argparse.ArgumentParser(
        description="Unified convergence sweep across environments, methods, and seeds")

    parser.add_argument("--environment", type=str, required=True,
                        choices=list(SETUP_FUNCS.keys()))
    parser.add_argument("--methods", type=str, nargs="+",
                        default=sweep_cfg.get("methods", ALL_METHODS),
                        choices=ALL_METHODS)
    parser.add_argument("--damping", type=float, nargs="+",
                        default=sweep_cfg.get("damping", [0.1, 0.25, 0.5, 0.75, 1.0]))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=sweep_cfg.get("seeds", list(range(10))))
    parser.add_argument("--n-iterations", type=int,
                        default=sweep_cfg.get("n_iterations", 200))
    parser.add_argument("--planning-horizon", type=int, default=None,
                        help="Override planning horizon (default: env-specific)")
    parser.add_argument("--output-dir", type=str, default="data/convergence_sweep")

    # Frozen Lake specific
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--n-configs", type=int, default=None)
    parser.add_argument("--hole-fraction", type=float, default=None)
    parser.add_argument("--min-hamming", type=int, default=None)
    parser.add_argument("--base-noise", type=float, default=None)
    parser.add_argument("--noise-range", type=float, default=None)
    parser.add_argument("--slip-prob", type=float, default=None)
    parser.add_argument("--hole-penalty", type=float, default=None)
    parser.add_argument("--goal-temperature", type=float, default=None)
    parser.add_argument("--scan-cost", type=float, default=None)

    # Wumpus World specific
    parser.add_argument("--n-pits", type=int, default=None)
    parser.add_argument("--obs-noise", type=float, default=None)
    parser.add_argument("--pos-noise", type=float, default=None)
    parser.add_argument("--pit-penalty", type=float, default=None)
    parser.add_argument("--wumpus-penalty", type=float, default=None)

    # RockSample specific
    parser.add_argument("--n-rocks", type=int, default=None)
    parser.add_argument("--half-eff-dist", type=float, default=None)
    parser.add_argument("--good-reward", type=float, default=None)
    parser.add_argument("--bad-penalty", type=float, default=None)
    parser.add_argument("--exit-reward", type=float, default=None)
    parser.add_argument("--sample-cost", type=float, default=None)
    parser.add_argument("--terminal-goal-only", action="store_true", default=None)

    # MiniGrid specific
    parser.add_argument("--fov-size", type=int, default=None)
    parser.add_argument("--obs-alpha", type=float, default=None)
    parser.add_argument("--observe-first", action="store_true", default=None)

    args = parser.parse_args()

    # Apply env-specific defaults for unset arguments
    env = args.environment
    defaults = ENV_DEFAULTS[env]
    for key, val in defaults.items():
        arg_key = key.replace("-", "_")
        if getattr(args, arg_key, None) is None:
            setattr(args, arg_key, val)

    # Ensure all needed attributes exist (avoid AttributeError for other envs)
    for attr in ["n_configs", "hole_fraction", "min_hamming", "base_noise",
                 "noise_range", "hole_penalty", "n_pits", "obs_noise",
                 "pos_noise", "pit_penalty", "wumpus_penalty", "n_rocks",
                 "half_eff_dist", "good_reward", "bad_penalty", "exit_reward",
                 "sample_cost", "terminal_goal_only", "fov_size", "obs_alpha",
                 "observe_first", "scan_cost", "slip_prob", "goal_temperature",
                 "grid_size", "planning_horizon"]:
        if not hasattr(args, attr) or getattr(args, attr) is None:
            setattr(args, attr, None)

    setup_fn = SETUP_FUNCS[env]
    output_dir = Path(args.output_dir)

    print(f"JAX devices: {jax.devices()}")
    print(f"Environment: {env}")
    print(f"Methods: {args.methods}")
    print(f"Damping values: {args.damping}")
    print(f"Seeds: {args.seeds}")
    print(f"Iterations: {args.n_iterations}  Horizon: {args.planning_horizon}")
    print(f"Output: {output_dir / env}")
    print()

    all_results = []
    total_runs = 0
    for method in args.methods:
        damping_vals = args.damping if method in METHODS_WITH_DAMPING else [1.0]
        total_runs += len(damping_vals) * len(args.seeds)

    run_idx = 0

    for seed in args.seeds:
        print(f"Setting up tensors for seed={seed}...")
        random.seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        setup = setup_fn(args, seed)
        print(f"  Done in {time.time() - t0:.1f}s")

        for method in args.methods:
            damping_vals = args.damping if method in METHODS_WITH_DAMPING else [1.0]

            for damping in damping_vals:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] {env} method={method} "
                      f"damping={damping} seed={seed} ...", end=" ", flush=True)

                t0 = time.time()
                action_dist, vfe_trace = call_convergence(
                    method, setup,
                    args.planning_horizon, args.n_iterations, damping,
                )
                action_dist.block_until_ready()
                elapsed = time.time() - t0

                vfe_values = np.array(vfe_trace)
                final_vfe = float(vfe_values[-1])
                final_delta = abs(float(vfe_values[-1] - vfe_values[-2])) if len(vfe_values) >= 2 else float("nan")
                converged = final_delta < 1e-4

                # Find convergence iteration
                convergence_iteration = -1
                for i in range(1, len(vfe_values)):
                    if abs(vfe_values[i] - vfe_values[i - 1]) < 1e-4:
                        convergence_iteration = i
                        break

                print(f"done in {elapsed:.1f}s  VFE={final_vfe:.4f}  "
                      f"delta={final_delta:.6f}{'*' if converged else ''}")

                save_result_json(
                    output_dir, env, method, damping, seed, args,
                    vfe_trace, action_dist, elapsed,
                )

                all_results.append(dict(
                    method=method, damping=damping, seed=seed,
                    final_vfe=final_vfe, final_delta=final_delta,
                    converged=converged,
                    convergence_iteration=convergence_iteration,
                    elapsed=elapsed,
                ))

    # Write summary CSV
    print()
    write_summary_csv(output_dir, env, all_results)

    # Print summary table
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = (f"{'Method':<22s} {'Damping':>7s} {'Seeds':>5s} "
              f"{'Med VFE':>10s} {'Med Δ':>10s} {'Conv%':>6s} {'Med iter':>8s}")
    print(header)
    print("-" * 80)

    # Group by (method, damping)
    from itertools import groupby
    sorted_results = sorted(all_results, key=lambda r: (r["method"], r["damping"]))
    for (method, damping), group in groupby(sorted_results, key=lambda r: (r["method"], r["damping"])):
        group = list(group)
        n = len(group)
        vfes = [r["final_vfe"] for r in group]
        deltas = [r["final_delta"] for r in group]
        conv_count = sum(1 for r in group if r["converged"])
        conv_iters = [r["convergence_iteration"] for r in group if r["convergence_iteration"] >= 0]
        med_vfe = float(np.median(vfes))
        med_delta = float(np.median(deltas))
        med_iter = float(np.median(conv_iters)) if conv_iters else float("nan")
        conv_pct = 100.0 * conv_count / n
        print(f"{method:<22s} {damping:>7.2f} {n:>5d} "
              f"{med_vfe:>10.4f} {med_delta:>10.6f} {conv_pct:>5.0f}% "
              f"{med_iter:>8.0f}" if not np.isnan(med_iter) else
              f"{method:<22s} {damping:>7.2f} {n:>5d} "
              f"{med_vfe:>10.4f} {med_delta:>10.6f} {conv_pct:>5.0f}%      n/a")

    print()
    print(f"  * = converged (|ΔVFE| < 1e-4)")
    print(f"Total runs: {total_runs}")


if __name__ == "__main__":
    main()
