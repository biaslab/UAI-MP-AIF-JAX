#!/usr/bin/env python
"""Side-by-side trajectory comparison across planning methods for one Wumpus World config.

Usage:
    uv run python plot_wumpus_trajectory.py --config-idx 18
    uv run python plot_wumpus_trajectory.py --config-idx 11 --format png
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from environments.wumpus_world import pos_to_rc


METHOD_LABELS = {
    "loopy": "BP",
    "dyn-channel": "RM-MP",
    "nuijten": "Nuijten MP",
    "vbp-channel": "VBP",
    "active-inference": "Active Inference",
}

DEFAULT_METHODS = ["loopy", "dyn-channel", "nuijten", "vbp-channel", "active-inference"]

EMPTY, PIT, WUMPUS, GOLD = 0, 1, 2, 3
TERRAIN_COLORS = ["#f5f5f5", "#e57373", "#9575cd", "#ffd54f"]
TERRAIN_LETTERS = {PIT: "P", WUMPUS: "W", GOLD: "G"}
TERRAIN_CMAP = ListedColormap(TERRAIN_COLORS)

# Arrow color gradient: early steps light, late steps dark.
PATH_CMAP = LinearSegmentedColormap.from_list(
    "path_gradient", ["#bdbdbd", "#212121"], N=256
)


def load_trajectory(traj_dir, method, config_idx):
    method_dir = traj_dir / method
    if not method_dir.exists():
        raise FileNotFoundError(f"Method directory missing: {method_dir}")
    for f in sorted(method_dir.glob("episode_*.npz")):
        d = np.load(f)
        if int(d["config_idx"]) == config_idx:
            return {k: d[k] for k in d.files}
    raise FileNotFoundError(f"No trajectory with config_idx={config_idx} under {method_dir}")


def load_episode(traj_dir, method, episode_idx):
    path = traj_dir / method / f"episode_{episode_idx:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def list_episode_indices(traj_dir, methods):
    """Return sorted list of episode indices recorded for ALL given methods."""
    sets = []
    for m in methods:
        method_dir = traj_dir / m
        if not method_dir.exists():
            return []
        idxs = set()
        for f in method_dir.glob("episode_*.npz"):
            stem = f.stem.replace("episode_", "")
            try:
                idxs.add(int(stem))
            except ValueError:
                continue
        sets.append(idxs)
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def build_terrain(traj):
    g = int(traj["grid_size"])
    grid = np.full((g, g), EMPTY, dtype=int)
    for code, key in [(PIT, "pits"), (WUMPUS, "wumpus"), (GOLD, "gold")]:
        for pos, val in enumerate(traj[key]):
            if val > 0.5:
                r, c = pos_to_rc(pos, g)
                grid[r, c] = code
    return grid


def classify_outcome(traj, terrain):
    g = int(traj["grid_size"])
    total_reward = float(np.sum(traj["rewards"]))
    if bool(traj["truncated"]):
        return total_reward, "timeout"
    if bool(traj["terminated"]) and total_reward > 0:
        return total_reward, "success"
    if bool(traj["terminated"]) and total_reward < 0:
        last_pos = int(traj["positions"][-1])
        r, c = pos_to_rc(last_pos, g)
        cell = terrain[r, c]
        if cell == PIT:
            return total_reward, "pit"
        if cell == WUMPUS:
            return total_reward, "wumpus"
    return total_reward, "fail"


def find_sense_steps(traj):
    """Indices of actions that triggered a SENSE (the bit is transient)."""
    sensed = traj["sensed"]
    return [t - 1 for t in range(1, len(sensed)) if sensed[t] == 1]


def draw_panel(ax, traj, method_label):
    g = int(traj["grid_size"])
    terrain = build_terrain(traj)
    positions = traj["positions"]
    actions = traj["actions"]

    ax.imshow(terrain, cmap=TERRAIN_CMAP, vmin=0, vmax=3, interpolation="nearest")

    for i in range(g + 1):
        ax.axhline(i - 0.5, color="white", lw=1.2)
        ax.axvline(i - 0.5, color="white", lw=1.2)

    for r in range(g):
        for c in range(g):
            cell = terrain[r, c]
            if cell in TERRAIN_LETTERS:
                ax.text(c, r, TERRAIN_LETTERS[cell],
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color="black")

    n_moves = max(len(actions), 1)
    norm = Normalize(vmin=0, vmax=max(n_moves - 1, 1))
    for t in range(len(actions)):
        p_from = int(positions[t])
        p_to = int(positions[t + 1])
        if p_from == p_to:
            continue
        r_from, c_from = pos_to_rc(p_from, g)
        r_to, c_to = pos_to_rc(p_to, g)
        color = PATH_CMAP(norm(t))
        ax.annotate(
            "",
            xy=(c_to, r_to), xytext=(c_from, r_from),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.0,
                mutation_scale=14,
                shrinkA=14, shrinkB=14,
            ),
        )

    p0 = int(positions[0])
    r0, c0 = pos_to_rc(p0, g)
    ax.text(c0 + 0.3, r0 - 0.3, "A",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="#0d47a1",
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#0d47a1", lw=1))

    sense_steps = find_sense_steps(traj)
    for sense_step in sense_steps:
        p_sense = int(positions[sense_step])
        r_s, c_s = pos_to_rc(p_sense, g)
        ax.text(c_s - 0.3, r_s - 0.3, "S",
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="#212121",
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#212121", lw=1))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(g - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(method_label, fontsize=10)

    total_reward, outcome = classify_outcome(traj, terrain)
    n_steps = len(actions)
    lines = [f"R={total_reward:+.0f}, {n_steps} steps, {outcome}"]
    def lab(name, v):
        return name if v > 0.5 else f"¬{name}"
    for sense_step in sense_steps[:3]:  # avoid overcrowding the caption
        obs = traj["observations"][sense_step + 1]
        r_s, c_s = pos_to_rc(int(positions[sense_step]), g)
        lines.append(f"SENSE@({r_s},{c_s}): {lab('b', obs[0])}, {lab('s', obs[1])}, {lab('g', obs[2])}")
    if len(sense_steps) > 3:
        lines.append(f"(+{len(sense_steps) - 3} more SENSE)")
    ax.set_xlabel("\n".join(lines), fontsize=8)


def add_legend(fig):
    """Place a shared legend and a step-gradient colorbar below the row of panels."""
    handles = [
        mpatches.Patch(facecolor=TERRAIN_COLORS[PIT], edgecolor="black", label="P  pit"),
        mpatches.Patch(facecolor=TERRAIN_COLORS[WUMPUS], edgecolor="black", label="W  wumpus"),
        mpatches.Patch(facecolor=TERRAIN_COLORS[GOLD], edgecolor="black", label="G  gold"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#0d47a1",
               markersize=9, markeredgewidth=1.2, linestyle="None",
               label="A  agent start"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#212121",
               markersize=9, markeredgewidth=1.2, linestyle="None",
               label="S  SENSE action"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.02),
    )

    # Gradient colorbar showing step order (early -> late)
    cbar_ax = fig.add_axes([0.35, -0.01, 0.30, 0.018])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=PATH_CMAP)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["early", "late"])
    cbar.ax.tick_params(labelsize=7, length=0, pad=2)
    cbar.outline.set_visible(False)
    cbar_ax.set_title("arrow color = step order", fontsize=7, pad=2)


def render_figure(traj_dir, methods, episode_idx, config_idx, out_path, suptitle=True):
    """Render one comparison figure across methods for a single episode/config."""
    trajectories = []
    for method in methods:
        try:
            if episode_idx is not None:
                traj = load_episode(traj_dir, method, episode_idx)
            else:
                traj = load_trajectory(traj_dir, method, config_idx)
            trajectories.append((method, traj))
        except FileNotFoundError as e:
            print(f"WARNING: skipping {method}: {e}", file=sys.stderr)

    if not trajectories:
        return False

    if episode_idx is not None:
        configs = {int(t["config_idx"]) for _, t in trajectories}
        if len(configs) != 1:
            print(f"WARNING: episode {episode_idx} has differing configs across methods: {configs}",
                  file=sys.stderr)
        config_idx = next(iter(configs))

    n = len(trajectories)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.9))
    if n == 1:
        axes = [axes]

    for ax, (method, traj) in zip(axes, trajectories):
        draw_panel(ax, traj, METHOD_LABELS.get(method, method))

    if suptitle:
        title = f"Wumpus World — config {config_idx}"
        if episode_idx is not None:
            title += f" (episode {episode_idx})"
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    else:
        fig.tight_layout(rect=[0, 0.10, 1, 1.0])

    add_legend(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Wumpus World trajectory comparison plot")
    parser.add_argument("--trajectory-dir", default="data/trajectories/wumpus_world")
    parser.add_argument("--config-idx", type=int, default=None,
                        help="Which config to render. Ignored if --all-episodes is set.")
    parser.add_argument("--all-episodes", action="store_true",
                        help="Render one figure per recorded episode (across all methods).")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output", default=None,
                        help="Output path (without extension) for single-figure mode.")
    parser.add_argument("--output-dir", default="data/plots/wumpus_world",
                        help="Output directory for --all-episodes mode.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    parser.add_argument("--no-suptitle", action="store_true",
                        help="Omit the figure-level title (useful for paper inclusion).")
    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)
    suptitle = not args.no_suptitle

    if args.all_episodes:
        episode_indices = list_episode_indices(traj_dir, args.methods)
        if not episode_indices:
            print("No common episode indices found across methods.", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.output_dir)
        any_ok = False
        for ep in episode_indices:
            sample_method = args.methods[0]
            try:
                cfg = int(load_episode(traj_dir, sample_method, ep)["config_idx"])
            except FileNotFoundError:
                continue
            out_path = out_dir / f"method_comparison_episode_{ep:03d}_config_{cfg:02d}.{args.format}"
            ok = render_figure(traj_dir, args.methods, ep, cfg, out_path, suptitle=suptitle)
            any_ok = any_ok or ok
        if not any_ok:
            sys.exit(1)
        return

    config_idx = args.config_idx if args.config_idx is not None else 18
    if args.output is None:
        out_path = Path(args.output_dir) / f"method_comparison_config_{config_idx:02d}.{args.format}"
    else:
        out_path = Path(args.output)
        if out_path.suffix == "":
            out_path = out_path.with_suffix(f".{args.format}")

    ok = render_figure(traj_dir, args.methods, None, config_idx, out_path, suptitle=suptitle)
    if not ok:
        print(f"No trajectories found for config_idx={config_idx}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
