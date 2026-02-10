#!/usr/bin/env python
"""Diagnostic tool comparing BP and AIF planning step-by-step.

Runs a single episode with both planners receiving identical observations.
Outputs per-step action distributions, planned trajectories, and AIF internals
to reveal where and why the planners diverge.
"""

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
from environments.minigrid import generate_transition_indices, generate_observation_indices, generate_orientation_indices
from environments.gym_wrapper import MiniGridWrapper
from inference.state_inference import state_inference_step_indexed
from inference.planning import planning_indexed
from inference.aif_planning import aif_planning_indexed
from inference.diagnostic_planning import (
    diagnostic_planning_indexed,
    diagnostic_aif_planning_indexed,
    BPDiagnostics,
    AIFDiagnostics,
)
from utils.tensors import (
    get_dimensions,
    flatten_state_index,
    unflatten_state_index,
    unflatten_static_index,
    location_to_coords,
)

# MiniGrid action names
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

# Orientation names
ORI_NAMES = ["right", "down", "left", "up"]

# Door-key state names
DKS_NAMES = ["none", "has_key", "door_open"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def create_goal_distribution(grid_size: int, goal_x: int, goal_y: int) -> jnp.ndarray:
    dims = get_dimensions(grid_size)
    goal = jnp.zeros(dims["n_states"])
    goal_location = goal_x * grid_size + goal_y
    for orientation in range(dims["n_orientations"]):
        idx = flatten_state_index(
            goal_location, orientation, 2,
            dims["n_locations"], dims["n_orientations"], dims["n_door_key_states"],
        )
        goal = goal.at[idx].set(1.0)
    return goal / goal.sum()


def format_state_label(flat_idx: int, dims: dict, grid_size: int) -> str:
    """Format a flat state index as a human-readable label."""
    loc, ori, dks = unflatten_state_index(
        flat_idx, dims["n_locations"], dims["n_orientations"], dims["n_door_key_states"]
    )
    x, y = location_to_coords(loc, grid_size)
    return f"({x},{y}) {ORI_NAMES[ori]} {DKS_NAMES[dks]}"


def format_static_label(static_idx: int, dims: dict, grid_size: int) -> str:
    """Format a flat static index as human-readable label."""
    key_pos, door_pos = unflatten_static_index(
        static_idx, dims["n_key_positions"], dims["n_door_positions"]
    )
    kx, ky = location_to_coords(key_pos, grid_size)
    dx, dy = location_to_coords(door_pos, grid_size)
    return f"key@({kx},{ky}) door@({dx},{dy})"


def format_action_dist(q_u: jnp.ndarray) -> str:
    """Format action distribution as compact string."""
    parts = []
    for i in range(len(q_u)):
        if q_u[i] > 0.01:
            parts.append(f"{ACTION_NAMES[i]}={q_u[i]:.3f}")
    return " ".join(parts)


def top_k_states(probs: jnp.ndarray, dims: dict, grid_size: int, k: int = 5) -> list[tuple[str, float]]:
    """Return top-k states with labels and probabilities."""
    indices = jnp.argsort(probs)[::-1][:k]
    result = []
    for idx in indices:
        idx_int = int(idx)
        p = float(probs[idx_int])
        if p < 1e-6:
            break
        label = format_state_label(idx_int, dims, grid_size)
        result.append((label, p))
    return result


def top_k_static(probs: jnp.ndarray, dims: dict, grid_size: int, k: int = 3) -> list[tuple[str, float]]:
    """Return top-k static configs with labels and probabilities."""
    indices = jnp.argsort(probs)[::-1][:k]
    result = []
    for idx in indices:
        idx_int = int(idx)
        p = float(probs[idx_int])
        if p < 1e-6:
            break
        label = format_static_label(idx_int, dims, grid_size)
        result.append((label, p))
    return result


def kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> float:
    """KL(p || q) with epsilon smoothing."""
    eps = 1e-10
    p_safe = jnp.clip(p, eps, 1.0)
    q_safe = jnp.clip(q, eps, 1.0)
    return float(jnp.sum(p_safe * jnp.log(p_safe / q_safe)))


def entropy(p: jnp.ndarray) -> float:
    """Shannon entropy."""
    eps = 1e-10
    p_safe = jnp.clip(p, eps, 1.0)
    return float(-jnp.sum(p_safe * jnp.log(p_safe)))


def print_planned_trajectory(diag, dims: dict, grid_size: int, label: str, top_k: int):
    """Print the planned trajectory (argmax state at each time step)."""
    horizon = diag.q_u.shape[0]
    print(f"  Planned trajectory ({label}):")
    for t in range(min(horizon, 8)):  # Show at most 8 steps
        state_idx = int(jnp.argmax(diag.q_state[t]))
        state_label = format_state_label(state_idx, dims, grid_size)
        state_prob = float(diag.q_state[t, state_idx])
        action_str = format_action_dist(diag.q_u[t]) if t < horizon else ""
        chosen = ACTION_NAMES[int(jnp.argmax(diag.q_u[t]))] if t < horizon else ""
        print(f"    t={t}: {state_label} (p={state_prob:.3f}) -> {chosen}  [{action_str}]")
    if horizon > 8:
        print(f"    ... ({horizon - 8} more steps)")
    # Final state
    state_idx = int(jnp.argmax(diag.q_state[horizon]))
    state_label = format_state_label(state_idx, dims, grid_size)
    state_prob = float(diag.q_state[horizon, state_idx])
    print(f"    t={horizon}: {state_label} (p={state_prob:.3f}) [terminal]")


def print_iteration_convergence(diag, label: str):
    """Print per-iteration convergence info."""
    print(f"  {label} iteration convergence:")
    for i, q_u in enumerate(diag.q_u_history):
        action_idx = int(jnp.argmax(q_u[0]))
        action_prob = float(q_u[0, action_idx])
        action_name = ACTION_NAMES[action_idx]
        if i > 0:
            prev_q_u = diag.q_u_history[i - 1]
            delta = float(jnp.max(jnp.abs(q_u[0] - prev_q_u[0])))
            print(f"    iter {i}: {action_name}={action_prob:.4f}  delta={delta:.6f}")
        else:
            print(f"    iter {i}: {action_name}={action_prob:.4f}")


def verify_diagnostic_matches_production(
    q_current_state, q_static_state, transition_idx, observation_idx, goal,
    horizon, n_iterations
):
    """Verify diagnostic wrappers produce the same q_u[0] as production."""
    # BP check
    bp_prod = planning_indexed(
        q_current_state, q_static_state, transition_idx, goal, horizon, n_iterations
    )
    bp_diag = diagnostic_planning_indexed(
        q_current_state, q_static_state, transition_idx, goal, horizon, n_iterations
    )
    bp_diff = float(jnp.max(jnp.abs(bp_prod - bp_diag.q_u[0])))

    # AIF check
    aif_prod = aif_planning_indexed(
        q_current_state, q_static_state, transition_idx, observation_idx, goal,
        horizon, n_iterations
    )
    aif_diag = diagnostic_aif_planning_indexed(
        q_current_state, q_static_state, transition_idx, observation_idx, goal,
        horizon, n_iterations
    )
    aif_diff = float(jnp.max(jnp.abs(aif_prod - aif_diag.q_u[0])))

    print("Correctness verification:")
    status_bp = "PASS" if bp_diff < 1e-5 else "FAIL"
    status_aif = "PASS" if aif_diff < 1e-5 else "FAIL"
    print(f"  BP  diagnostic vs production: max|diff| = {bp_diff:.2e} [{status_bp}]")
    print(f"  AIF diagnostic vs production: max|diff| = {aif_diff:.2e} [{status_aif}]")
    if bp_diff >= 1e-5 or aif_diff >= 1e-5:
        print("  WARNING: Diagnostic output does not match production!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="BP vs AIF planning diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_diagnostics.py --grid-size 3 --seed 0
  python run_diagnostics.py --grid-size 3 --seed 0 --follow aif --verbose-iterations
  python run_diagnostics.py --grid-size 3 --seed 0 --output-json diagnostics.json
""",
    )
    # Same as run_experiment.py
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--planning-horizon", type=int, default=15)
    parser.add_argument("--receding-horizon", action="store_true")
    parser.add_argument("--inference-iterations", type=int, default=10)
    parser.add_argument("--planning-iterations", type=int, default=10)
    # New diagnostic-specific args
    parser.add_argument("--follow", choices=["bp", "aif"], default="bp",
                        help="Which agent's actions drive the environment (default: bp)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top states to show (default: 5)")
    parser.add_argument("--steps-limit", type=int, default=None,
                        help="Early stop after this many steps")
    parser.add_argument("--verbose-iterations", action="store_true",
                        help="Show per-iteration convergence details")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Dump diagnostics to JSON file")
    args = parser.parse_args()

    set_seed(args.seed)

    grid_size = args.grid_size
    minigrid_size = grid_size + 2
    env_name = f"MiniGrid-DoorKey-{minigrid_size}x{minigrid_size}-v0"
    dims = get_dimensions(grid_size)

    print(f"JAX devices: {jax.devices()}")
    print(f"Grid: {grid_size}x{grid_size} (MiniGrid {minigrid_size}x{minigrid_size})")
    print(f"Seed: {args.seed}")
    print(f"Follow: {args.follow}")
    print(f"Planning: horizon={args.planning_horizon}, iterations={args.planning_iterations}")
    print(f"Inference: iterations={args.inference_iterations}")
    print()

    # Generate tensors
    print("Generating tensors...")
    t0 = time.time()
    transition_idx = jnp.array(generate_transition_indices(grid_size))
    observation_idx = jnp.array(generate_observation_indices(grid_size))
    orientation_idx = jnp.array(generate_orientation_indices(grid_size))
    print(f"  Done in {time.time() - t0:.2f}s")
    print()

    # Goal
    goal_x = grid_size - 1
    goal_y = 0
    goal = create_goal_distribution(grid_size, goal_x, goal_y)
    print(f"Goal: ({goal_x}, {goal_y}) with door open")
    print()

    # Correctness verification using initial uniform beliefs
    print("Verifying diagnostic wrappers...")
    n_valid_locations = dims["n_locations"] - 2 * grid_size
    state_probs = jnp.zeros(dims["n_states"])
    for loc in range(n_valid_locations):
        for ori in range(dims["n_orientations"]):
            idx = flatten_state_index(
                loc, ori, 0,
                dims["n_locations"], dims["n_orientations"], dims["n_door_key_states"],
            )
            state_probs = state_probs.at[idx].set(1.0)
    state_probs = state_probs / state_probs.sum()
    static_probs = jnp.ones(dims["n_static"]) / dims["n_static"]

    verify_diagnostic_matches_production(
        state_probs, static_probs, transition_idx, observation_idx, goal,
        args.planning_horizon, args.planning_iterations,
    )

    # Set up environment
    env = MiniGridWrapper(env_name=env_name, max_steps=args.max_steps)
    result = env.reset(seed=args.seed)

    # Agent state (shared between both planners since they get identical observations)
    q_state = state_probs
    q_static = static_probs
    last_action = 0

    total_reward = 0.0
    steps = 0
    divergence_count = 0
    first_divergence_step = None

    # For JSON output
    json_steps = []

    max_steps_env = env.max_steps
    steps_limit = args.steps_limit or max_steps_env

    print("=" * 70)
    print("EPISODE START")
    print("=" * 70)

    while True:
        # State inference (shared — both planners see the same observation)
        q_state, q_static = state_inference_step_indexed(
            q_old_state=q_state,
            q_static_state=q_static,
            transition_idx=transition_idx,
            obs_idx=observation_idx,
            ori_idx=orientation_idx,
            vision_obs=result.vision_obs,
            ori_obs=result.orientation_obs,
            action_idx=last_action,
            n_iterations=args.inference_iterations,
        )

        if args.receding_horizon:
            time_remaining = max_steps_env - steps
            horizon = min(time_remaining, args.planning_horizon)
        else:
            horizon = args.planning_horizon

        # Run both planners
        bp_diag = diagnostic_planning_indexed(
            q_state, q_static, transition_idx, goal, horizon, args.planning_iterations
        )
        aif_diag = diagnostic_aif_planning_indexed(
            q_state, q_static, transition_idx, observation_idx, goal,
            horizon, args.planning_iterations
        )

        bp_action = int(jnp.argmax(bp_diag.q_u[0]))
        aif_action = int(jnp.argmax(aif_diag.q_u[0]))
        actions_agree = bp_action == aif_action
        kl_bp_aif = kl_divergence(bp_diag.q_u[0], aif_diag.q_u[0])

        if not actions_agree:
            divergence_count += 1
            if first_divergence_step is None:
                first_divergence_step = steps

        # Choose action
        if args.follow == "bp":
            env_action = bp_action
        else:
            env_action = aif_action

        # Print step header
        follow_label = "BP" if args.follow == "bp" else "AIF"
        print(f"\nSTEP {steps} | horizon={horizon} | env_action={ACTION_NAMES[env_action]} ({follow_label})")
        print("-" * 50)

        # State inference results
        print("--- STATE INFERENCE ---")
        top_states = top_k_states(q_state, dims, grid_size, args.top_k)
        for label, p in top_states:
            print(f"  {label}  p={p:.4f}")
        top_statics = top_k_static(q_static, dims, grid_size, k=3)
        print(f"  Static top-3:")
        for label, p in top_statics:
            print(f"    {label}  p={p:.4f}")

        # BP planning
        print("--- BP PLANNING ---")
        print(f"  Action: {ACTION_NAMES[bp_action]} (p={float(bp_diag.q_u[0, bp_action]):.4f})")
        print(f"  Dist: {format_action_dist(bp_diag.q_u[0])}")
        print_planned_trajectory(bp_diag, dims, grid_size, "BP", args.top_k)
        if args.verbose_iterations:
            print_iteration_convergence(bp_diag, "BP")

        # AIF planning
        print("--- AIF PLANNING ---")
        print(f"  Action: {ACTION_NAMES[aif_action]} (p={float(aif_diag.q_u[0, aif_action]):.4f})")
        print(f"  Dist: {format_action_dist(aif_diag.q_u[0])}")
        print_planned_trajectory(aif_diag, dims, grid_size, "AIF", args.top_k)

        # AIF-specific internals
        top_theta = top_k_static(aif_diag.q_theta, dims, grid_size, k=3)
        print(f"  q_theta top-3:")
        for label, p in top_theta:
            print(f"    {label}  p={p:.4f}")

        # K_mod vs reduced_tensor divergence (use K_mod[0] as representative)
        kmod_diff = float(jnp.max(jnp.abs(aif_diag.K_mod[0] - aif_diag.reduced_tensor)))
        print(f"  K_mod[0] max|diff from reduced|: {kmod_diff:.6f}")

        # Obs message diagnostics (use obs_msgs_to_x[0] as representative)
        obs_x_entropy = entropy(jax.nn.softmax(aif_diag.obs_msgs_to_x[0]))
        print(f"  obs_msgs_to_x[0] entropy: {obs_x_entropy:.4f}")

        # r_x diagnostics: how far from uniform (use r_x[0] as representative)
        r_x_uniform = jnp.ones_like(aif_diag.r_x[0]) / aif_diag.r_x.shape[1]
        r_x_diff = float(jnp.max(jnp.abs(aif_diag.r_x[0] - r_x_uniform)))
        print(f"  r_x[0] max|diff from uniform|: {r_x_diff:.6f}")

        if args.verbose_iterations:
            print_iteration_convergence(aif_diag, "AIF")
            # Show q_theta evolution
            print("  AIF q_theta per iteration:")
            for i, qt in enumerate(aif_diag.q_theta_history):
                top1_idx = int(jnp.argmax(qt))
                top1_label = format_static_label(top1_idx, dims, grid_size)
                top1_p = float(qt[top1_idx])
                print(f"    iter {i}: {top1_label} p={top1_p:.4f}")

        # Divergence summary
        print("--- DIVERGENCE ---")
        agree_str = "AGREE" if actions_agree else "DISAGREE"
        print(f"  Actions: {agree_str}  BP={ACTION_NAMES[bp_action]}  AIF={ACTION_NAMES[aif_action]}")
        print(f"  KL(BP||AIF): {kl_bp_aif:.6f}")

        # Collect JSON step data
        if args.output_json:
            step_data = {
                "step": steps,
                "horizon": horizon,
                "env_action": env_action,
                "env_action_name": ACTION_NAMES[env_action],
                "follow": args.follow,
                "state_inference": {
                    "top_states": [{"label": l, "prob": p} for l, p in top_states],
                    "top_static": [{"label": l, "prob": p} for l, p in top_statics],
                },
                "bp": {
                    "action": bp_action,
                    "action_name": ACTION_NAMES[bp_action],
                    "action_dist": [float(x) for x in bp_diag.q_u[0]],
                },
                "aif": {
                    "action": aif_action,
                    "action_name": ACTION_NAMES[aif_action],
                    "action_dist": [float(x) for x in aif_diag.q_u[0]],
                    "q_theta_top3": [{"label": l, "prob": p} for l, p in top_theta],
                    "kmod_diff": kmod_diff,
                    "obs_x_entropy": obs_x_entropy,
                    "r_x_diff": r_x_diff,
                },
                "divergence": {
                    "actions_agree": actions_agree,
                    "kl_bp_aif": kl_bp_aif,
                },
            }
            json_steps.append(step_data)

        # Step environment
        last_action = env_action
        result = env.step(env_action)
        total_reward += result.reward
        steps += 1

        if result.terminated or result.truncated or steps >= steps_limit:
            break

    env.close()

    # Episode summary
    success = result.reward > 0
    outcome = "SUCCESS" if success else ("TRUNCATED" if result.truncated else "TERMINATED")

    print("\n" + "=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)
    print(f"  Outcome: {outcome}")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Divergences: {divergence_count}/{steps}")
    if first_divergence_step is not None:
        print(f"  First divergence at step: {first_divergence_step}")
    else:
        print(f"  No divergences — BP and AIF chose identical actions every step")

    # JSON dump
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "config": {
                "grid_size": grid_size,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "planning_horizon": args.planning_horizon,
                "receding_horizon": args.receding_horizon,
                "inference_iterations": args.inference_iterations,
                "planning_iterations": args.planning_iterations,
                "follow": args.follow,
            },
            "summary": {
                "outcome": outcome,
                "steps": steps,
                "total_reward": total_reward,
                "divergence_count": divergence_count,
                "first_divergence_step": first_divergence_step,
            },
            "steps": json_steps,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDiagnostics saved to {args.output_json}")


if __name__ == "__main__":
    main()
