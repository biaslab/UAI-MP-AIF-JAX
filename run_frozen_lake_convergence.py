#!/usr/bin/env python
"""Convergence analysis for Frozen Lake: track VFE per iteration across damping values.

Usage:
    python run_frozen_lake_convergence.py
    python run_frozen_lake_convergence.py --method region-extended --damping 0.25 0.5 1.0
    python run_frozen_lake_convergence.py --method dyn-channel --n-iterations 30
"""

import sys
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
from environments.frozen_lake import (
    sample_configs,
    generate_transition_tensor,
    generate_observation_tensor,
    generate_goal,
    pos_to_rc,
)
from inference.convergence import (
    region_extended_convergence,
    dyn_channel_convergence,
    vbp_channel_convergence,
    precise_info_seeking_convergence,
    active_inference_convergence,
)

ACTION_NAMES = ["left", "down", "right", "up"]

# Methods that support damping (channel-based)
CONVERGENCE_FUNCS = {
    "region-extended": region_extended_convergence,
    "dyn-channel": dyn_channel_convergence,
    "vbp-channel": vbp_channel_convergence,
    "precise-info-seeking": precise_info_seeking_convergence,
    "active-inference": active_inference_convergence,
}


def run_convergence(method, q_current, q_static, T, B_dir, goal,
                    horizon, n_iterations, damping):
    """Run convergence analysis and return (action_dist, vfe_trace)."""
    func = CONVERGENCE_FUNCS[method]

    result = func(
        q_current_state=q_current,
        q_static_state=q_static,
        transition_tensor=T,
        observation_tensor=B_dir,
        goal=goal,
        horizon=horizon,
        n_iterations=n_iterations,
        damping=damping,
    )

    # All return (action_dist, ..., vfe_trace) — vfe_trace is always last
    action_dist = result[0]
    vfe_trace = result[-1]
    return action_dist, vfe_trace


def main():
    parser = argparse.ArgumentParser(
        description="Frozen Lake convergence analysis: VFE per iteration")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--hole-fraction", type=float, default=0.2)
    parser.add_argument("--min-hamming", type=int, default=4)
    parser.add_argument("--obs-noise", type=float, default=0.15)
    parser.add_argument("--slip-prob", type=float, default=0.0)
    parser.add_argument("--planning-horizon", type=int, default=15)
    parser.add_argument("--n-iterations", type=int, default=25)
    parser.add_argument("--method", type=str, default="region-extended",
                        choices=list(CONVERGENCE_FUNCS.keys()))
    parser.add_argument("--damping", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 0.75, 1.0],
                        help="Damping values to test")
    parser.add_argument("--hole-penalty", type=float, default=5.0)
    parser.add_argument("--goal-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=0,
                        help="Starting position index (default: 0 = top-left)")
    parser.add_argument("--plot", nargs="?", const="auto", default=None,
                        help="Save convergence plot to data/frozen_lake_convergence/")
    args = parser.parse_args()

    grid_size = args.grid_size
    n_pos = grid_size * grid_size
    n_states = n_pos

    print(f"JAX devices: {jax.devices()}")
    print(f"Frozen Lake {grid_size}x{grid_size}  method={args.method}")
    print(f"  Configs: {args.n_configs}  hole_fraction: {args.hole_fraction}  "
          f"min_hamming: {args.min_hamming}")
    print(f"  Horizon: {args.planning_horizon}  Iterations: {args.n_iterations}")
    print(f"  Damping values: {args.damping}")
    print(f"  hole_penalty={args.hole_penalty}  goal_temp={args.goal_temperature}")
    print()

    # Generate tensors
    print("Generating tensors...")
    t0 = time.time()
    holes = sample_configs(grid_size, args.n_configs,
                           hole_fraction=args.hole_fraction, seed=args.seed,
                           min_hamming=args.min_hamming)
    T = jnp.array(
        generate_transition_tensor(grid_size, holes, slip_prob=args.slip_prob),
        dtype=jnp.float32)
    B = jnp.array(
        generate_observation_tensor(grid_size, holes, obs_noise=args.obs_noise),
        dtype=jnp.float32)
    goal = jnp.array(
        generate_goal(grid_size, holes, hole_penalty=args.hole_penalty,
                      temperature=args.goal_temperature),
        dtype=jnp.float32)
    print(f"  T: {T.shape}  B: {B.shape}  goal: {goal.shape}")
    print(f"  Done in {time.time() - t0:.2f}s")
    print()

    # Neighbor-sensor channels only (last 4 of B)
    B_dir = B[n_states:]
    print(f"  Neighbor-sensor obs tensor: {B_dir.shape}")

    # Initial beliefs
    q_current = jnp.zeros(n_states, dtype=jnp.float32).at[args.start_pos].set(1.0)
    n_static = holes.shape[0]
    q_static = jnp.ones(n_static, dtype=jnp.float32) / n_static

    r, c = pos_to_rc(args.start_pos, grid_size)
    print(f"  Start position: ({r},{c})")
    print()

    # Print configs summary
    print("Hole configurations (sample):")
    for i in range(min(5, n_static)):
        hpos = [pos_to_rc(j, grid_size) for j in range(n_pos) if holes[i, j] == 1.0]
        print(f"  #{i}: {hpos}")
    if n_static > 5:
        print(f"  ... ({n_static} total)")
    print()

    # Run convergence for each damping value
    results = {}
    for d in args.damping:
        print(f"{'='*60}")
        print(f"Damping = {d}")
        print(f"{'='*60}")

        t0 = time.time()
        action_dist, vfe_trace = run_convergence(
            args.method, q_current, q_static, T, B_dir, goal,
            args.planning_horizon, args.n_iterations, d,
        )
        action_dist.block_until_ready()
        elapsed = time.time() - t0

        vfe_values = np.array(vfe_trace)
        results[d] = {"action_dist": np.array(action_dist), "vfe": vfe_values}

        print(f"  Time: {elapsed:.2f}s")
        print()

        # VFE trace
        print("  VFE per iteration:")
        for i, v in enumerate(vfe_values):
            delta = ""
            if i > 0:
                change = v - vfe_values[i-1]
                delta = f"  (Δ={change:+.4f})"
            print(f"    iter {i:>3d}: VFE = {v:.6f}{delta}")
        print()

        # Check convergence
        if len(vfe_values) >= 3:
            last_delta = abs(vfe_values[-1] - vfe_values[-2])
            converged = last_delta < 1e-4
            print(f"  Final Δ: {last_delta:.6f}  {'CONVERGED' if converged else 'NOT converged'}")
        print()

        # Action distribution
        print("  Action distribution:")
        for i, name in enumerate(ACTION_NAMES):
            p = float(action_dist[i])
            bar = "#" * int(p * 40)
            print(f"    {name:>5s}: {p:.4f}  {bar}")
        print()

    # Summary comparison
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    header = f"{'Damping':>8s} | {'Final VFE':>12s} | {'ΔVFE (last)':>12s} | Action dist"
    print(header)
    print("-" * len(header) + "-" * 30)
    for d in args.damping:
        vfe = results[d]["vfe"]
        action = results[d]["action_dist"]
        final_vfe = vfe[-1]
        last_delta = abs(vfe[-1] - vfe[-2]) if len(vfe) >= 2 else float("nan")
        action_str = " ".join(f"{ACTION_NAMES[i]}={action[i]:.3f}" for i in range(4))
        converge_mark = " *" if last_delta < 1e-4 else ""
        print(f"{d:>8.3f} | {final_vfe:>12.4f} | {last_delta:>12.6f}{converge_mark} | {action_str}")

    print()
    print("  * = converged (ΔVFe < 1e-4)")

    # Plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if args.plot == "auto":
                from datetime import datetime
                basename = datetime.now().strftime('%Y%m%d_%H%M%S')
            else:
                basename = args.plot

            out_dir = Path("data/frozen_lake_convergence") / basename
            out_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            for d in args.damping:
                vfe = results[d]["vfe"]
                ax.plot(range(len(vfe)), vfe, "o-", markersize=3, label=f"d={d}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("VFE")
            ax.set_title(f"VFE Convergence: {args.method} (Frozen Lake {grid_size}x{grid_size}, "
                         f"T={args.planning_horizon})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            png_path = out_dir / "convergence.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"\nPlot saved to {png_path}")
        except ImportError:
            print("\nmatplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
