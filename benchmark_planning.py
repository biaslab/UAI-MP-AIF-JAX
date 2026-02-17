"""Benchmark planning algorithms to measure wall-clock performance."""

import time
import numpy as np
import jax
import jax.numpy as jnp

from environments.minigrid import (
    generate_transition_tensor,
    generate_observation_tensor,
    N_ORIENTATIONS,
    N_DOOR_KEY_STATES,
)
from inference.loopy_bp import loopy_bp_planning
from inference.loopy_vbp import loopy_vbp_planning
from inference.region_extended_loopy_bp import region_extended_loopy_bp_planning
from inference.reduced_region_extended import reduced_region_extended_planning
from inference.dyn_channel_loopy_bp import dyn_channel_loopy_bp_planning
from inference.nuijten_mp import nuijten_mp_planning


def setup_tensors(grid_size=4):
    """Generate shared tensors for all planners."""
    n_loc = grid_size * grid_size
    n_key = n_loc - 2 * grid_size
    n_door = n_loc - 2 * grid_size
    n_states = n_loc * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static = n_key * n_door

    print(f"Grid size: {grid_size}")
    print(f"n_states={n_states}, n_static={n_static}, n_actions=7")
    print("Generating tensors...")

    transition_tensor = jnp.array(generate_transition_tensor(grid_size), dtype=jnp.float32)
    observation_tensor = jnp.array(generate_observation_tensor(grid_size), dtype=jnp.float32)

    q_state = jnp.ones(n_states) / n_states
    q_static = jnp.ones(n_static) / n_static
    goal = jnp.zeros(n_states).at[0].set(1.0)

    print("Tensors ready.\n")
    return transition_tensor, observation_tensor, q_state, q_static, goal


def block_ready(result):
    """Recursively call block_until_ready on all JAX arrays in a result."""
    if isinstance(result, jnp.ndarray):
        result.block_until_ready()
    elif isinstance(result, (tuple, list)):
        for item in result:
            block_ready(item)
    return result


def benchmark(fn, args, n_runs=20):
    """Warmup once, then time n_runs calls. Returns times in seconds."""
    # JIT warmup
    block_ready(fn(*args))

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        block_ready(result)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def main():
    horizon = 15
    n_iterations = 10
    n_runs = 20

    T, B, q_state, q_static, goal = setup_tensors(grid_size=4)

    planners = [
        ("loopy_bp", loopy_bp_planning,
         (q_state, q_static, T, goal, horizon, n_iterations)),
        ("loopy_vbp", loopy_vbp_planning,
         (q_state, q_static, T, goal, horizon, n_iterations)),
        ("region_extended", region_extended_loopy_bp_planning,
         (q_state, q_static, T, B, goal, horizon, n_iterations)),
        ("reduced_region_ext", reduced_region_extended_planning,
         (q_state, q_static, T, B, goal, horizon, n_iterations)),
        ("dyn_channel", dyn_channel_loopy_bp_planning,
         (q_state, q_static, T, B, goal, horizon, n_iterations)),
        ("nuijten_mp", nuijten_mp_planning,
         (q_state, q_static, T, B, goal, horizon, n_iterations)),
    ]

    print(f"Benchmark: horizon={horizon}, iterations={n_iterations}, runs={n_runs}")
    print(f"Platform: {jax.default_backend()}")
    print(f"{'='*55}")
    print(f"{'Planner':<22} {'Median (ms)':>12} {'IQR (ms)':>12}")
    print(f"{'-'*55}")

    for name, fn, args in planners:
        print(f"  Benchmarking {name}...", end="", flush=True)
        times = benchmark(fn, args, n_runs=n_runs)
        times_ms = np.array(times) * 1000
        median = np.median(times_ms)
        q25, q75 = np.percentile(times_ms, [25, 75])
        iqr = q75 - q25
        print(f"\r{name:<22} {median:>12.2f} {iqr:>12.2f}")

    print(f"{'='*55}")


if __name__ == "__main__":
    main()
