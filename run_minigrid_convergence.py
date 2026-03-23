#!/usr/bin/env python
"""Convergence analysis: run 1 planning step and track VFE per iteration."""

import sys
from pathlib import Path
import argparse
import json
import time
import random
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
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
from inference.convergence import (
    loopy_bp_convergence,
    region_extended_convergence,
    dyn_channel_convergence,
    nuijten_mp_convergence,
    vbp_channel_convergence,
    precise_info_seeking_convergence,
    active_inference_convergence,
)
from utils.tensors import get_dimensions, flatten_state_index


def _flatten_obs(obs):
    """Flatten 5D obs tensor (fov_w, fov_h, ...) to 4D (n_channels, ...)."""
    if obs.ndim == 5:
        return obs.reshape(obs.shape[0] * obs.shape[1], *obs.shape[2:])
    return obs


def create_goal_distribution(grid_size, goal_x, goal_y):
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


def create_agent(args, transition_tensor, observation_tensor, orientation_tensor, goal):
    method = args.planning_method
    common = dict(
        grid_size=args.grid_size,
        observation_tensors=observation_tensor,
        orientation_tensor=orientation_tensor,
        goal=goal,
        planning_horizon=args.planning_horizon,
        n_inference_iterations=10,
        n_planning_iterations=args.planning_iterations,
    )
    if method == "loopy":
        return LoopyBPAgent.create(transition_tensor=transition_tensor, **common)
    elif method == "region-extended":
        return RegionExtendedAgent.create(
            transition_tensor=transition_tensor, **common)
    elif method == "dyn-channel":
        return DynChannelLoopyBPAgent.create(
            transition_tensor=transition_tensor, **common)
    elif method == "nuijten":
        return NuijtenMPAgent.create(
            transition_tensor=transition_tensor, **common)
    elif method == "vbp-channel":
        return VBPChannelAgent.create(
            transition_tensor=transition_tensor, damping=args.damping, momentum=args.momentum, **common)
    elif method == "precise-info-seeking":
        return PreciseInfoSeekingAgent.create(
            transition_tensor=transition_tensor, damping=args.damping, momentum=args.momentum, **common)
    elif method == "active-inference":
        return ActiveInferenceAgent.create(
            transition_tensor=transition_tensor, damping=args.damping, momentum=args.momentum, **common)
    else:
        raise ValueError(f"Unknown planning method: {method}")


def call_convergence_planning(method, q_current, q_static, agent, horizon,
                               n_iterations, damping=1.0, momentum=0.0):
    """Dispatch to the correct convergence planning function."""
    if method == "loopy":
        action_dist, vfe_trace = loopy_bp_convergence(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            goal=agent.goal,
            horizon=horizon,
            n_iterations=n_iterations,
        )
        return action_dist, vfe_trace
    elif method == "region-extended":
        action_dist, _, _, vfe_trace = region_extended_convergence(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=_flatten_obs(agent.observation_tensors),
            goal=agent.goal,
            horizon=horizon,
            n_iterations=n_iterations,
            damping=damping,
            momentum=momentum,
        )
        return action_dist, vfe_trace
    elif method == "dyn-channel":
        action_dist, _, vfe_trace = dyn_channel_convergence(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=_flatten_obs(agent.observation_tensors),
            goal=agent.goal,
            horizon=horizon,
            n_iterations=n_iterations,
            damping=damping,
            momentum=momentum,
        )
        return action_dist, vfe_trace
    elif method == "nuijten":
        action_dist, _, _, vfe_trace = nuijten_mp_convergence(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=_flatten_obs(agent.observation_tensors),
            goal=agent.goal,
            horizon=horizon,
            n_iterations=n_iterations,
        )
        return action_dist, vfe_trace
    elif method == "vbp-channel":
        action_dist, _, vfe_trace = vbp_channel_convergence(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=_flatten_obs(agent.observation_tensors),
            goal=agent.goal,
            horizon=horizon,
            n_iterations=n_iterations,
            damping=damping,
            momentum=momentum,
        )
        return action_dist, vfe_trace
    elif method == "precise-info-seeking":
        action_dist, _, _, vfe_trace = precise_info_seeking_convergence(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=_flatten_obs(agent.observation_tensors),
            goal=agent.goal,
            horizon=horizon,
            n_iterations=n_iterations,
            damping=damping,
            momentum=momentum,
        )
        return action_dist, vfe_trace
    elif method == "active-inference":
        action_dist, _, _, vfe_trace = active_inference_convergence(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=_flatten_obs(agent.observation_tensors),
            goal=agent.goal,
            horizon=horizon,
            n_iterations=n_iterations,
            damping=damping,
            momentum=momentum,
        )
        return action_dist, vfe_trace
    else:
        raise ValueError(f"Unknown planning method: {method}")


def generate_tikz(n_iterations, vfe_values, method, grid_size, horizon, damping):
    """Generate a standalone pgfplots .tex file with VFE convergence data."""
    coords = "\n".join(f"    ({i}, {v:.6f})" for i, v in enumerate(vfe_values))
    damp_str = f", damping={damping}" if damping < 1.0 else ""
    title = f"{method} (grid={grid_size}, T={horizon}{damp_str})"
    return f"""\
% Auto-generated by run_minigrid_convergence.py
\\documentclass[tikz]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
    xlabel={{Iteration}},
    ylabel={{VFE}},
    title={{{title}}},
    grid=major,
    mark=*,
    mark size=1.5pt,
]
\\addplot coordinates {{
{coords}
}};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""


def create_onehot(idx, size):
    return jnp.zeros(size).at[idx].set(1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Convergence analysis: track VFE per planning iteration")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--planning-horizon", type=int, default=15)
    parser.add_argument("--planning-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planning-method", type=str, default="loopy",
                        choices=["loopy", "region-extended",
                                 "dyn-channel", "nuijten",
                                 "vbp-channel", "precise-info-seeking",
                                 "active-inference"])
    parser.add_argument("--fov-size", type=int, default=7)
    parser.add_argument("--no-orientation", action="store_true")
    parser.add_argument("--damping", type=float, default=1.0,
                        help="Channel update damping (1.0 = no damping, 0.5 = equal blend)")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="Inertial momentum coefficient (0.0 = no momentum)")
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    parser.add_argument("--plot", nargs="?", const="auto", default=None,
                        help="Save VFE plot + TikZ to data/convergence/ (optional: custom basename)")
    parser.add_argument("--observe-first", action="store_true",
                        help="Take 1 observation + inference step before planning")
    parser.add_argument("--obs-alpha", type=float, default=0.0,
                        help="Observation softening rate per Manhattan distance (0.0 = no softening)")
    args = parser.parse_args()

    if args.fov_size < 3 or args.fov_size % 2 == 0:
        parser.error("--fov-size must be odd and >= 3")

    random.seed(args.seed)
    np.random.seed(args.seed)

    grid_size = args.grid_size
    minigrid_size = grid_size + 2
    env_name = f"MiniGrid-DoorKey-{minigrid_size}x{minigrid_size}-v0"
    valid_configs = get_valid_static_configs(grid_size)
    dims = get_dimensions(grid_size, n_static_override=len(valid_configs))

    print(f"JAX devices: {jax.devices()}")
    print(f"Method: {args.planning_method}")
    print(f"Grid: {grid_size}x{grid_size}  Horizon: {args.planning_horizon}  "
          f"Iterations: {args.planning_iterations}")
    if args.damping < 1.0:
        print(f"Channel damping: {args.damping}")
    if args.momentum > 0:
        print(f"Momentum: {args.momentum}")
    if args.obs_alpha > 0.0:
        print(f"Observation softening: alpha={args.obs_alpha}")
    print()

    # Generate tensors
    print("Generating tensors...")
    t0 = time.time()
    transition_tensor = jnp.array(generate_transition_tensor(grid_size, valid_configs), dtype=jnp.float32)
    obs_np = generate_observation_tensor(grid_size, valid_configs, fov_size=args.fov_size)
    if args.obs_alpha > 0.0:
        obs_np = soften_observation_tensor(obs_np, args.fov_size, args.obs_alpha)
    observation_tensor = jnp.array(obs_np, dtype=jnp.float32)
    orientation_tensor = jnp.array(
        generate_orientation_observation_tensor(grid_size), dtype=jnp.float32)
    print(f"  Done in {time.time() - t0:.2f}s")
    print()

    goal_x = grid_size - 1
    goal_y = grid_size - 1
    goal = create_goal_distribution(grid_size, goal_x, goal_y)

    agent = create_agent(args, transition_tensor, observation_tensor,
                          orientation_tensor, goal)

    q_state = agent.q_state
    q_static = agent.q_static

    # Optional: observe first to get non-uniform beliefs
    if args.observe_first:
        print("Taking initial observation...")
        env = MiniGridWrapper(env_name=env_name, max_steps=100, fov_size=args.fov_size, obs_alpha=args.obs_alpha)
        result = env.reset(seed=args.seed)

        if args.no_orientation:
            uniform_ori = jnp.ones(4) / 4
            result = StepResult(
                vision_obs=result.vision_obs,
                orientation_obs=uniform_ori,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                info=result.info,
            )

        action_onehot = create_onehot(0, dims["n_actions"])
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
        print("  Done (1 inference step from initial observation)")
        print()

    # Run convergence planning
    horizon = args.planning_horizon
    n_iterations = args.planning_iterations

    print(f"Running {n_iterations} planning iterations...")
    t0 = time.time()
    action_dist, vfe_trace = call_convergence_planning(
        args.planning_method, q_state, q_static, agent, horizon,
        n_iterations, damping=args.damping, momentum=args.momentum,
    )
    action_dist.block_until_ready()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")
    print()

    # Print VFE trace
    vfe_values = np.array(vfe_trace)
    print("VFE per iteration:")
    for i, v in enumerate(vfe_values):
        print(f"  Iteration {i:>3d}: VFE = {v:.6f}")
    print()

    # Print action distribution
    action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
    print("Action distribution:")
    for i, name in enumerate(action_names):
        p = float(action_dist[i])
        bar = "#" * int(p * 40)
        print(f"  {name:>7s}: {p:.4f}  {bar}")
    print()

    # JSON output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "config": {
                "grid_size": grid_size,
                "planning_method": args.planning_method,
                "planning_horizon": horizon,
                "planning_iterations": n_iterations,
                "fov_size": args.fov_size,
                "damping": args.damping,
                "momentum": args.momentum,
                "obs_alpha": args.obs_alpha,
                "observe_first": args.observe_first,
                "seed": args.seed,
            },
            "vfe_trace": vfe_values.tolist(),
            "action_dist": np.array(action_dist).tolist(),
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")

    # Plot
    if args.plot:
        if args.plot == "auto":
            basename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.planning_method}"
        else:
            basename = args.plot
        out_dir = Path("data/convergence") / basename
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / "convergence.png"
        tex_path = out_dir / "convergence.tex"

        # Save TikZ
        tikz_src = generate_tikz(
            n_iterations, vfe_values, args.planning_method,
            grid_size, horizon, args.damping,
        )
        tex_path.write_text(tikz_src)
        print(f"TikZ saved to {tex_path}")

        # Save PNG
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(n_iterations), vfe_values, "o-", markersize=4)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("VFE")
            damp_str = f" (damping={args.damping})" if args.damping < 1.0 else ""
            ax.set_title(
                f"VFE Convergence: {args.planning_method}{damp_str}\n"
                f"grid={grid_size}, horizon={horizon}, iters={n_iterations}"
            )
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"PNG saved to {png_path}")
        except ImportError:
            print("matplotlib not available — skipping PNG")


if __name__ == "__main__":
    main()
