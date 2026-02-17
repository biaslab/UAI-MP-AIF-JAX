#!/usr/bin/env python
"""Single-episode diagnostic script with full internal state output."""

import sys
from pathlib import Path
import argparse
import time
import random

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
from environments.minigrid import (
    generate_transition_tensor,
    generate_transition_indices,
    generate_observation_tensor,
    generate_observation_indices,
    generate_orientation_indices,
    soften_observation_tensor,
    N_CELL_TYPES,
)
from environments.gym_wrapper import MiniGridWrapper, StepResult
from agents.flat_tensor_agent import (
    IndexedTensorAgent, LoopyBPAgent, RegionExtendedAgent,
    ReducedRegionExtendedAgent, NuijtenMPAgent, ReducedNuijtenMPAgent,
)
from inference.state_inference import state_inference_step_indexed
from inference.planning import planning
from inference.loopy_bp import loopy_bp_planning
from inference.region_extended_loopy_bp import region_extended_loopy_bp_planning
from inference.reduced_region_extended import reduced_region_extended_planning
from inference.nuijten_mp import nuijten_mp_planning, reduced_nuijten_mp_planning
from utils.tensors import (
    get_dimensions, flatten_state_index, unflatten_state_index,
    unflatten_static_index, location_to_coords,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
ORIENTATION_NAMES = ["right", "down", "left", "up"]
DOOR_KEY_STATE_NAMES = ["no_key", "has_key", "door_open"]

CELL_TYPE_ABBR = [
    "unse",  # 0  unseen
    "empt",  # 1  empty
    "wall",  # 2  wall
    "flor",  # 3  floor
    "door",  # 4  door
    "key ",  # 5  key
    "ball",  # 6  ball
    "box ",  # 7  box
    "goal",  # 8  goal
    "lava",  # 9  lava
    "agnt",  # 10 agent
]

# ---------------------------------------------------------------------------
# Helper functions — marginal extraction
# ---------------------------------------------------------------------------


def compute_location_marginal(q_state, dims):
    """Reshape flat state belief and sum out orientation and door_key_state."""
    q = q_state.reshape(dims["n_locations"], dims["n_orientations"], dims["n_door_key_states"])
    return q.sum(axis=(1, 2))


def compute_orientation_marginal(q_state, dims):
    q = q_state.reshape(dims["n_locations"], dims["n_orientations"], dims["n_door_key_states"])
    return q.sum(axis=(0, 2))


def compute_door_key_state_marginal(q_state, dims):
    q = q_state.reshape(dims["n_locations"], dims["n_orientations"], dims["n_door_key_states"])
    return q.sum(axis=(0, 1))


def compute_key_position_marginal(q_static, dims):
    q = q_static.reshape(dims["n_door_positions"], dims["n_key_positions"])
    return q.sum(axis=0)


def compute_door_position_marginal(q_static, dims):
    q = q_static.reshape(dims["n_door_positions"], dims["n_key_positions"])
    return q.sum(axis=1)


# ---------------------------------------------------------------------------
# Helper functions — display
# ---------------------------------------------------------------------------


def entropy(p):
    """Shannon entropy in bits. Clips zeros to avoid log(0)."""
    p = jnp.clip(p, 1e-12, 1.0)
    return -float(jnp.sum(p * jnp.log2(p)))


def print_location_grid(loc_marginal, grid_size):
    """Print a 2D grid of location probabilities."""
    header = "        " + "  ".join(f"x={x+1:>2}" for x in range(grid_size))
    print(header)
    for y in range(grid_size):
        row = f"  y={y+1:>2}  "
        for x in range(grid_size):
            loc = x * grid_size + y
            p = float(loc_marginal[loc])
            if p < 0.0005:
                row += "   .  "
            else:
                row += f"{p:5.3f} "
        print(row)


def print_action_distribution(action_dist):
    """Print all 7 actions with names and a bar chart."""
    bar_width = 40
    for i, name in enumerate(ACTION_NAMES):
        p = float(action_dist[i])
        n_bars = int(p * bar_width)
        bar = "#" * n_bars
        print(f"      {name:>7s}: {p:.4f}  {bar}")


def print_orientation_belief(ori_marginal):
    """Print orientation marginal."""
    parts = [f"{ORIENTATION_NAMES[i]}: {float(ori_marginal[i]):.4f}" for i in range(4)]
    print(f"    {  '  '.join(parts)}")


def print_door_key_state_belief(dks_marginal):
    """Print door/key state marginal."""
    parts = [f"{DOOR_KEY_STATE_NAMES[i]}: {float(dks_marginal[i]):.4f}" for i in range(3)]
    print(f"    {  '  '.join(parts)}")


def print_static_belief_summary(q_static, dims, grid_size):
    """Print top-3 key and door positions as grid coords."""
    key_m = compute_key_position_marginal(q_static, dims)
    door_m = compute_door_position_marginal(q_static, dims)

    # Top-3 key positions
    key_order = jnp.argsort(-key_m)
    parts = []
    for rank in range(min(3, len(key_m))):
        idx = int(key_order[rank])
        x, y = location_to_coords(idx, grid_size)
        parts.append(f"({x},{y})={float(key_m[idx]):.2f}")
    print(f"    Key position (top 3): {'  '.join(parts)}")

    # Top-3 door positions
    door_order = jnp.argsort(-door_m)
    parts = []
    for rank in range(min(3, len(door_m))):
        idx = int(door_order[rank])
        x, y = location_to_coords(idx, grid_size)
        parts.append(f"({x},{y})={float(door_m[idx]):.2f}")
    print(f"    Door position (top 3): {'  '.join(parts)}")


def print_observation_summary(vision_obs, orientation_obs, fov_size):
    """Print FOV cell types and orientation."""
    # Orientation
    ori_idx = int(jnp.argmax(orientation_obs))
    ori_p = float(orientation_obs[ori_idx])
    if ori_p > 0.99:
        print(f"    Orientation obs: {ORIENTATION_NAMES[ori_idx]}")
    else:
        print(f"    Orientation obs: uniform (disabled)")

    # Vision grid
    print(f"    Vision ({fov_size}x{fov_size} FOV):")
    for j in range(fov_size):
        row = "       "
        for i in range(fov_size):
            cell_idx = int(jnp.argmax(vision_obs[i, j]))
            row += f" {CELL_TYPE_ABBR[cell_idx]}"
        print(row)


# ---------------------------------------------------------------------------
# Planning dispatch
# ---------------------------------------------------------------------------


def call_planning(method, q_current, q_static, agent, horizon, damping=1.0):
    """Dispatch to the correct planning function based on method string."""
    if method == "bp":
        return planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            goal=agent.goal,
            horizon=horizon,
            n_iterations=agent.n_planning_iterations,
        )
    elif method == "loopy":
        return loopy_bp_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            goal=agent.goal,
            horizon=horizon,
            n_iterations=agent.n_planning_iterations,
        )
    elif method == "region-extended":
        result = region_extended_loopy_bp_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=agent.observation_tensor,
            goal=agent.goal,
            horizon=horizon,
            n_iterations=agent.n_planning_iterations,
            damping=damping,
        )
        return result[0]
    elif method == "reduced-aif":
        result = reduced_region_extended_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=agent.observation_tensor,
            goal=agent.goal,
            horizon=horizon,
            n_iterations=agent.n_planning_iterations,
            damping=damping,
        )
        return result[0]
    elif method == "nuijten":
        result = nuijten_mp_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=agent.observation_tensor,
            goal=agent.goal,
            horizon=horizon,
            n_iterations=agent.n_planning_iterations,
        )
        return result[0]
    elif method == "reduced-nuijten":
        result = reduced_nuijten_mp_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=agent.transition_tensor,
            observation_tensor=agent.observation_tensor,
            goal=agent.goal,
            horizon=horizon,
            n_iterations=agent.n_planning_iterations,
        )
        return result[0]
    else:
        raise ValueError(f"Unknown planning method: {method}")


# ---------------------------------------------------------------------------
# Diagnostic episode
# ---------------------------------------------------------------------------


def run_diagnostic_episode(agent, env, args, dims, grid_size):
    """Run a single episode with full diagnostic output at every step."""
    fov_size = args.fov_size
    uniform_orientation = jnp.ones(4) / 4

    # Reset
    result = env.reset(seed=args.seed)
    agent = agent.reset()

    if args.no_orientation:
        result = StepResult(
            vision_obs=result.vision_obs,
            orientation_obs=uniform_orientation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )

    q_state = agent.q_state
    q_static = agent.q_static
    last_action = 0
    max_steps = env.max_steps

    max_entropy_state = jnp.log2(float(dims["n_states"]))
    max_entropy_static = jnp.log2(float(dims["n_static"]))

    # Print initial beliefs
    print("=" * 70)
    print("INITIAL BELIEFS")
    print("=" * 70)
    print()
    print("  [STATE BELIEF]")
    loc_m = compute_location_marginal(q_state, dims)
    print("    Location belief grid (y\\x):")
    print_location_grid(loc_m, grid_size)
    ori_m = compute_orientation_marginal(q_state, dims)
    print("    Orientation belief:")
    print_orientation_belief(ori_m)
    dks_m = compute_door_key_state_marginal(q_state, dims)
    print("    Door/key state belief:")
    print_door_key_state_belief(dks_m)
    print(f"    State entropy: {entropy(q_state):.2f} bits (max={max_entropy_state:.2f})")
    print()
    print("  [STATIC BELIEF]")
    print_static_belief_summary(q_static, dims, grid_size)
    print(f"    Static entropy: {entropy(q_static):.2f} bits (max={max_entropy_static:.2f})")
    print()

    total_reward = 0.0
    step_num = 0

    while True:
        print("=" * 70)
        print(f"STEP {step_num}")
        print("=" * 70)
        print()

        # --- OBSERVATION ---
        print("  [OBSERVATION]")
        print_observation_summary(result.vision_obs, result.orientation_obs, fov_size)
        print()

        # --- STATE INFERENCE ---
        print("  [STATE INFERENCE]")
        t0 = time.time()
        q_state, q_static = state_inference_step_indexed(
            q_old_state=q_state,
            q_static_state=q_static,
            transition_idx=agent.transition_idx,
            obs_idx=agent.observation_idx,
            ori_idx=agent.orientation_idx,
            vision_obs=result.vision_obs,
            ori_obs=result.orientation_obs,
            action_idx=last_action,
            n_iterations=agent.n_inference_iterations,
        )
        # Block until computation completes for accurate timing
        q_state.block_until_ready()
        inference_ms = (time.time() - t0) * 1000
        print(f"    Inference time: {inference_ms:.1f}ms")

        loc_m = compute_location_marginal(q_state, dims)
        print("    Location belief grid (y\\x):")
        print_location_grid(loc_m, grid_size)

        ori_m = compute_orientation_marginal(q_state, dims)
        print("    Orientation belief:")
        print_orientation_belief(ori_m)

        dks_m = compute_door_key_state_marginal(q_state, dims)
        print("    Door/key state belief:")
        print_door_key_state_belief(dks_m)

        # MAP state
        map_idx = int(jnp.argmax(q_state))
        map_loc, map_ori, map_dks = unflatten_state_index(
            map_idx, dims["n_locations"], dims["n_orientations"], dims["n_door_key_states"]
        )
        map_x, map_y = location_to_coords(map_loc, grid_size)
        map_p = float(q_state[map_idx])
        print(f"    MAP state: ({map_x},{map_y}) facing {ORIENTATION_NAMES[map_ori]}, "
              f"{DOOR_KEY_STATE_NAMES[map_dks]} (p={map_p:.4f})")
        print(f"    State entropy: {entropy(q_state):.2f} bits (max={max_entropy_state:.2f})")
        print()

        # --- STATIC BELIEF ---
        print("  [STATIC BELIEF]")
        print_static_belief_summary(q_static, dims, grid_size)
        print(f"    Static entropy: {entropy(q_static):.2f} bits (max={max_entropy_static:.2f})")
        print()

        # --- PLANNING ---
        if args.receding_horizon:
            time_remaining = max_steps - step_num
        else:
            time_remaining = agent.planning_horizon
        horizon = min(time_remaining, agent.planning_horizon)

        print("  [PLANNING]")
        print(f"    Horizon: {horizon} (time_remaining={time_remaining})")

        t0 = time.time()
        action_dist = call_planning(args.planning_method, q_state, q_static, agent, horizon,
                                     damping=args.damping)
        action_dist.block_until_ready()
        planning_ms = (time.time() - t0) * 1000
        print(f"    Planning time: {planning_ms:.1f}ms")

        print("    Action distribution:")
        print_action_distribution(action_dist)

        action = int(jnp.argmax(action_dist))
        print(f"    Selected: {ACTION_NAMES[action]}")
        print()

        # --- EXECUTE ---
        result = env.step(action)
        if args.no_orientation:
            result = StepResult(
                vision_obs=result.vision_obs,
                orientation_obs=uniform_orientation,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                info=result.info,
            )
        total_reward += result.reward
        last_action = action

        print(f"  [EXECUTE] action={ACTION_NAMES[action]}")
        print(f"    Reward: {result.reward}  Terminated: {result.terminated}  Truncated: {result.truncated}")
        print()

        step_num += 1

        if result.terminated or result.truncated:
            break

    # --- EPISODE SUMMARY ---
    print("=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)
    success = result.reward > 0
    print(f"  Success: {success}")
    print(f"  Total steps: {step_num}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final state entropy: {entropy(q_state):.2f} bits")
    print(f"  Final static entropy: {entropy(q_static):.2f} bits")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


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


def create_agent(args, transition_tensor, transition_idx, observation_tensor,
                  observation_idx, orientation_idx, goal):
    """Create the appropriate agent based on planning method."""
    method = args.planning_method
    common = dict(
        grid_size=args.grid_size,
        transition_idx=transition_idx,
        observation_idx=observation_idx,
        orientation_idx=orientation_idx,
        goal=goal,
        planning_horizon=args.planning_horizon,
        n_inference_iterations=args.inference_iterations,
        n_planning_iterations=args.planning_iterations,
    )
    if method == "loopy":
        return LoopyBPAgent.create(transition_tensor=transition_tensor, **common)
    elif method == "region-extended":
        return RegionExtendedAgent.create(
            transition_tensor=transition_tensor, observation_tensor=observation_tensor, **common)
    elif method == "reduced-aif":
        return ReducedRegionExtendedAgent.create(
            transition_tensor=transition_tensor, observation_tensor=observation_tensor, **common)
    elif method == "nuijten":
        return NuijtenMPAgent.create(
            transition_tensor=transition_tensor, observation_tensor=observation_tensor, **common)
    elif method == "reduced-nuijten":
        return ReducedNuijtenMPAgent.create(
            transition_tensor=transition_tensor, observation_tensor=observation_tensor, **common)
    else:  # bp
        return IndexedTensorAgent.create(transition_tensor=transition_tensor, **common)


def main():
    parser = argparse.ArgumentParser(description="Single-episode diagnostic with full internal state output")
    parser.add_argument("--grid-size", type=int, default=3, help="Internal grid size (default: 3)")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--planning-horizon", type=int, default=15, help="Planning horizon")
    parser.add_argument("--receding-horizon", action="store_true", help="Decrease horizon as time runs out")
    parser.add_argument("--inference-iterations", type=int, default=10, help="State inference iterations")
    parser.add_argument("--planning-iterations", type=int, default=10, help="Planning iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--planning-method", type=str, default="bp",
                        choices=["bp", "loopy", "region-extended", "reduced-aif", "nuijten", "reduced-nuijten"],
                        help="Planning method")
    parser.add_argument("--fov-size", type=int, default=7, help="Field-of-view size (odd, >= 3)")
    parser.add_argument("--no-orientation", action="store_true",
                        help="Replace orientation observation with uniform")
    parser.add_argument("--damping", type=float, default=1.0,
                        help="Channel update damping (1.0 = no damping, 0.5 = equal blend)")
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
    dims = get_dimensions(grid_size)

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"Random seed: {args.seed}")
    print(f"Internal grid size: {grid_size}x{grid_size}")
    print(f"MiniGrid environment: {env_name}")
    print(f"Planning method: {args.planning_method}")
    print(f"FOV size: {args.fov_size}x{args.fov_size}")
    if args.no_orientation:
        print("Orientation observation: DISABLED (uniform)")
    if args.damping < 1.0:
        print(f"Channel damping: {args.damping}")
    if args.obs_alpha > 0.0:
        print(f"Observation softening: alpha={args.obs_alpha}")
    print(f"Planning horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    print(f"Inference iterations: {args.inference_iterations}")
    print(f"Planning iterations: {args.planning_iterations}")
    print(f"Max steps: {args.max_steps}")
    print()

    print("Generating tensors...")
    t0 = time.time()
    transition_tensor = jnp.array(generate_transition_tensor(grid_size), dtype=jnp.float32)
    transition_idx = jnp.array(generate_transition_indices(grid_size))
    obs_np = generate_observation_tensor(grid_size, fov_size=args.fov_size)
    if args.obs_alpha > 0.0:
        obs_np = soften_observation_tensor(obs_np, args.fov_size, args.obs_alpha)
    observation_tensor = jnp.array(obs_np, dtype=jnp.float32)
    observation_idx = jnp.array(generate_observation_indices(grid_size, fov_size=args.fov_size))
    orientation_idx = jnp.array(generate_orientation_indices(grid_size))
    print(f"  Transition tensor: {transition_tensor.shape}")
    print(f"  Transition indices: {transition_idx.shape}")
    print(f"  Observation tensor: {observation_tensor.shape}")
    print(f"  Observation indices: {observation_idx.shape}")
    print(f"  Orientation indices: {orientation_idx.shape}")
    print(f"  Generated in {time.time() - t0:.2f}s")
    print()

    print(f"Dimensions: {dims}")
    print()

    goal_x = grid_size - 1
    goal_y = 0
    goal = create_goal_distribution(grid_size, goal_x, goal_y)
    print(f"Goal: position ({goal_x}, {goal_y}) with door open")
    print()

    agent = create_agent(args, transition_tensor, transition_idx, observation_tensor,
                         observation_idx, orientation_idx, goal)
    env = MiniGridWrapper(env_name=env_name, max_steps=args.max_steps, fov_size=args.fov_size, obs_alpha=args.obs_alpha)

    run_diagnostic_episode(agent, env, args, dims, grid_size)

    env.close()


if __name__ == "__main__":
    main()
