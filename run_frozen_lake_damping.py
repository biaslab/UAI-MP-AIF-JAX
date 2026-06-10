#!/usr/bin/env python
"""Convergence analysis: region-extended VFE vs iteration for different damping values.

Runs a single Frozen Lake instance with region-extended planning at 100 iterations
for several damping values. Outputs pgfplots-compatible CSV and a PDF figure.

Usage:
    JAX_PLATFORMS=cpu uv run python run_frozen_lake_damping.py
    JAX_PLATFORMS=cpu uv run python run_frozen_lake_damping.py --damping 0.1 0.25 0.5 0.75 1.0
    JAX_PLATFORMS=cpu uv run python run_frozen_lake_damping.py --n-iterations 50
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
from inference.convergence import region_extended_convergence

ACTION_NAMES = ["left", "down", "right", "up"]


def main():
    parser = argparse.ArgumentParser(
        description="Frozen Lake damping convergence: VFE traces for LaTeX")
    # Frozen Lake instance (defaults from params.yaml)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--n-configs", type=int, default=15)
    parser.add_argument("--hole-fraction", type=float, default=0.2)
    parser.add_argument("--min-hamming", type=int, default=4)
    parser.add_argument("--obs-noise", type=float, default=0.15)
    parser.add_argument("--slip-prob", type=float, default=0.1)
    parser.add_argument("--planning-horizon", type=int, default=15)
    parser.add_argument("--hole-penalty", type=float, default=2.0)
    parser.add_argument("--goal-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=0)
    # Convergence settings
    parser.add_argument("--n-iterations", type=int, default=100)
    parser.add_argument("--damping", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 0.75, 1.0])
    # Output
    parser.add_argument("--output-dir", type=str,
                        default="data/frozen_lake_damping")
    args = parser.parse_args()

    grid_size = args.grid_size
    n_pos = grid_size * grid_size
    n_states = n_pos

    print(f"JAX devices: {jax.devices()}")
    print(f"Frozen Lake {grid_size}x{grid_size}  region-extended")
    print(f"  Configs: {args.n_configs}  hole_fraction={args.hole_fraction}  "
          f"min_hamming={args.min_hamming}")
    print(f"  obs_noise={args.obs_noise}  slip_prob={args.slip_prob}")
    print(f"  Horizon: {args.planning_horizon}  Iterations: {args.n_iterations}")
    print(f"  hole_penalty={args.hole_penalty}  goal_temp={args.goal_temperature}")
    print(f"  Damping values: {args.damping}")
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

    # Uniform prior over the 4 movement actions
    action_prior = None

    # Initial beliefs
    q_current = jnp.zeros(n_states, dtype=jnp.float32).at[args.start_pos].set(1.0)
    n_static = holes.shape[0]
    q_static = jnp.ones(n_static, dtype=jnp.float32) / n_static

    r, c = pos_to_rc(args.start_pos, grid_size)
    print(f"  Start position: ({r},{c})")

    # Print configs summary
    print("  Hole configurations (sample):")
    for i in range(min(5, n_static)):
        hpos = [pos_to_rc(j, grid_size) for j in range(n_pos) if holes[i, j] == 1.0]
        print(f"    #{i}: {hpos}")
    if n_static > 5:
        print(f"    ... ({n_static} total)")
    print()

    # --- Run convergence for each damping value ---
    results = {}
    for d in args.damping:
        print(f"Damping = {d} ...")
        t0 = time.time()

        result = region_extended_convergence(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=T,
            observation_tensor=B_dir,
            goal=goal,
            horizon=args.planning_horizon,
            n_iterations=args.n_iterations,
            damping=d,
            action_prior=action_prior,
        )

        action_dist = result[0]
        vfe_trace = result[-1]
        action_dist.block_until_ready()
        elapsed = time.time() - t0

        vfe_values = np.array(vfe_trace)
        results[d] = {
            "action_dist": np.array(action_dist),
            "vfe": vfe_values,
            "time": elapsed,
        }

        last_delta = abs(vfe_values[-1] - vfe_values[-2]) if len(vfe_values) >= 2 else float("nan")
        converged = last_delta < 1e-4
        action_str = " ".join(f"{ACTION_NAMES[i]}={float(action_dist[i]):.3f}"
                              for i in range(min(4, len(action_dist))))
        print(f"  {elapsed:.1f}s  final_VFE={vfe_values[-1]:.4f}  "
              f"delta={last_delta:.6f}{'*' if converged else ''}  [{action_str}]")

    print()

    # --- Output directory ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Column names: replace dots with underscores for pgfplots compatibility ---
    def col_name(d):
        return f"d{str(d).replace('.', '_')}"

    n_iter = args.n_iterations
    header_parts = ["iteration"] + [col_name(d) for d in args.damping]

    # CSV (comma-separated)
    csv_path = out_dir / "vfe_traces.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(header_parts) + "\n")
        for i in range(n_iter):
            row = [str(i + 1)]
            for d in args.damping:
                row.append(f"{results[d]['vfe'][i]:.8f}")
            f.write(",".join(row) + "\n")
    print(f"CSV saved to {csv_path}")

    # DAT (space-separated, pgfplots default)
    dat_path = out_dir / "vfe_traces.dat"
    with open(dat_path, "w") as f:
        f.write("  ".join(header_parts) + "\n")
        for i in range(n_iter):
            row = [str(i + 1)]
            for d in args.damping:
                row.append(f"{results[d]['vfe'][i]:.8f}")
            f.write("  ".join(row) + "\n")
    print(f"DAT saved to {dat_path}")

    # --- Standalone .tex file with pgfplots ---
    tex_path = out_dir / "vfe_convergence.tex"
    colors = ["blue", "red!80!black", "green!60!black", "orange", "purple", "brown"]
    marks = ["*", "square*", "triangle*", "diamond*", "pentagon*", "star"]
    plot_lines = []
    legend_entries = []
    for i, d in enumerate(args.damping):
        c = colors[i % len(colors)]
        m = marks[i % len(marks)]
        plot_lines.append(
            f"    \\addplot[{c}, mark={m}, mark repeat=10, mark size=1.5pt] "
            f"table[col sep=comma, x=iteration, y={col_name(d)}] {{vfe_traces.csv}};"
        )
        legend_entries.append(f"$\\lambda = {d}$")

    tex_content = f"""\
\\documentclass[tikz]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=newest}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=10cm,
    height=7cm,
    xlabel={{Iteration}},
    ylabel={{VFE}},
    grid=major,
    grid style={{gray!30}},
    legend pos=south east,
    legend cell align={{left}},
    cycle list name=color list,
]
{chr(10).join(plot_lines)}
    \\legend{{{", ".join(legend_entries)}}}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"TEX saved to {tex_path}")

    # --- PDF figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.usetex": False,
            "figure.figsize": (5.5, 3.5),
        })
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        iters = np.arange(1, n_iter + 1)
        for d in args.damping:
            ax.plot(iters, results[d]["vfe"], "-", linewidth=1.2,
                    label=f"$\\lambda = {d}$")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("VFE")
        ax.legend()
        ax.grid(True, alpha=0.3, linewidth=0.5)
        fig.tight_layout()

        pdf_path = out_dir / "vfe_convergence.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"PDF saved to {pdf_path}")
    except ImportError:
        print("matplotlib not available -- skipping PDF")

    # --- Summary table ---
    print()
    print("=" * 70)
    print(f"{'Damping':>8s} | {'Final VFE':>12s} | {'delta (last)':>14s} | {'Time':>6s}")
    print("-" * 70)
    for d in args.damping:
        vfe = results[d]["vfe"]
        final_vfe = vfe[-1]
        last_delta = abs(vfe[-1] - vfe[-2]) if len(vfe) >= 2 else float("nan")
        mark = " *" if last_delta < 1e-4 else ""
        print(f"{d:>8.3f} | {final_vfe:>12.4f} | {last_delta:>14.8f}{mark} | {results[d]['time']:>5.1f}s")
    print("  * = converged (delta < 1e-4)")


if __name__ == "__main__":
    main()
