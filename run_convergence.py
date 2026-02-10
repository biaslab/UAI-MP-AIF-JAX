#!/usr/bin/env python
"""Convergence analysis of AIF planning free energy.

Computes the modified free energy at each AIF iteration:
  F = VFE_MF + Σ_t [H(q(x_{t-1}, u_t)) - H(q(x_t, x_{t-1}, u_t))
                    + H(q(y_t, x_t, θ)) - H(q(x_t, θ))]
and plots convergence over iterations.

The VFE_MF is the standard mean-field variational free energy using the
reduced (θ-marginalised) transition tensor. The correction terms use
the Bethe-like factor beliefs from the AIF message-passing scheme:
  - Dynamics triplet q(x_t, x_{t-1}, u_t) built from K_mod and augmented messages
  - Observation factor belief q_obs(x_t, θ) built from cavity messages
"""

import sys
from pathlib import Path
import argparse
import time
import random

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environments.minigrid import (
    generate_transition_indices,
    generate_observation_indices,
    generate_orientation_indices,
)
from environments.gym_wrapper import MiniGridWrapper
from inference.state_inference import state_inference_step_indexed
from inference.planning import marginalize_static_indexed
from inference.aif_planning import (
    N_CELL_TYPES,
    N_FOV,
    compute_cavities,
    compute_modified_kernel,
    compute_all_obs_msgs_to_x,
    compute_obs_msgs_to_theta_per_t,
    aif_forward_pass,
    aif_backward_pass_with_messages,
    compute_theta_messages_from_dynamics,
    update_q_theta,
    channel_update_dynamics,
    channel_update_obs,
)
from inference.messages import EPSILON
from utils.tensors import get_dimensions, flatten_state_index


ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


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


def h(p):
    """Shannon entropy of a probability vector (any shape, flattened)."""
    p_flat = p.ravel()
    return -jnp.sum(p_flat * jnp.log(p_flat + EPSILON))


def compute_full_marginals(q_state, backward_msgs, log_obs_msgs_to_x, horizon):
    """Combine forward, backward, and obs messages into full x_t marginals.

    q_full(x_t) ∝ q_state[t] · backward_msgs[t] · exp(obs_msgs_to_x[t])
    x_0 is clamped (initial state belief, not updated by planning).
    """
    marginals = [q_state[0]]
    for t in range(1, horizon + 1):
        obs_factor_t = jnp.exp(log_obs_msgs_to_x[t])
        q_full = q_state[t] * backward_msgs[t] * obs_factor_t
        q_full = q_full / (q_full.sum() + EPSILON)
        marginals.append(q_full)
    return jnp.stack(marginals)


def compute_fe_terms(
    q_state, q_u, q_theta, reduced_tensor, K_mod,
    backward_msgs, log_obs_msgs_to_x, goal, action_prior,
    prior_theta, horizon,
):
    """Compute VFE and entropy correction terms for one iteration snapshot.

    Called after forward-backward pass with current beliefs and messages.

    Args:
        K_mod: (T, n_states, n_states, n_actions) per-timestep modified kernels
        log_obs_msgs_to_x: (T+1, n_states) per-timestep log obs messages

    Returns dict with:
      total, VFE, E_dyn, E_goal, E_u, E_theta,
      H_MF, H_theta, H_states, H_actions,
      dyn_correction, obs_correction
    """
    q_full = compute_full_marginals(q_state, backward_msgs, log_obs_msgs_to_x, horizon)

    # === VFE (mean-field): F_MF = E - H_MF ===

    # Dynamics energy: -Σ_t <ln T_red(x_t|x_{t-1},u_t)>_{q(x_t)q(x_{t-1})q(u_t)}
    log_T = jnp.log(reduced_tensor + EPSILON)
    E_dyn = 0.0
    for t in range(horizon):
        E_dyn -= float(jnp.einsum("i,j,k,ijk->", q_full[t + 1], q_full[t], q_u[t], log_T))

    # Goal energy: -<ln goal(x_T)>_{q(x_T)}
    E_goal = -float(jnp.sum(q_full[horizon] * jnp.log(goal + EPSILON)))

    # Action prior energy: -Σ_t <ln p(u_t)>_{q(u_t)}
    log_ap = jnp.log(action_prior + EPSILON)
    E_u = sum(-float(jnp.sum(q_u[t] * log_ap)) for t in range(horizon))

    # Theta prior energy: -<ln p(θ)>_{q(θ)}
    E_theta = -float(jnp.sum(q_theta * jnp.log(prior_theta + EPSILON)))

    # Mean-field entropy: H[q(θ)] + Σ_t H[q(x_t)] + Σ_t H[q(u_t)]
    H_theta = float(h(q_theta))
    H_states = sum(float(h(q_full[t])) for t in range(horizon + 1))
    H_actions = sum(float(h(q_u[t])) for t in range(horizon))
    H_MF = H_theta + H_states + H_actions

    VFE = E_dyn + E_goal + E_u + E_theta - H_MF

    # === Dynamics entropy correction ===
    # Σ_t [H(q(x_{t-1}, u_t)) - H(q(x_t, x_{t-1}, u_t))]
    # = Σ_t [-H(x_t | x_{t-1}, u_t)]  under q_dyn
    dyn_corr = 0.0
    for t in range(horizon):
        obs_factor_t = jnp.exp(log_obs_msgs_to_x[t])
        obs_factor_tp1 = jnp.exp(log_obs_msgs_to_x[t + 1])

        fwd_aug = q_state[t] * obs_factor_t
        fwd_aug = fwd_aug / (fwd_aug.sum() + EPSILON)
        bwd_aug = backward_msgs[t + 1] * obs_factor_tp1
        bwd_aug = bwd_aug / (bwd_aug.sum() + EPSILON)

        # Dynamics triplet belief using per-timestep K_mod
        q_trip = (
            K_mod[t]
            * fwd_aug[None, :, None]
            * bwd_aug[:, None, None]
            * q_u[t][None, None, :]
        )
        q_trip = q_trip / (q_trip.sum() + EPSILON)

        # Separator: q(x_{t-1}, u_t) = Σ_{x_t} q_trip
        q_sep = q_trip.sum(axis=0)  # (n_states, n_actions)

        dyn_corr += float(h(q_sep)) - float(h(q_trip))

    # === Observation entropy correction ===
    # Σ_t [H(q(y_t, x_t, θ)) - H(q(x_t, θ))]
    #
    # For deterministic p(y|x,θ): H(q(y,x,θ)) = H(q_obs(x,θ)).
    # q_obs(x,θ) ∝ m_{x→obs}(x) · q(θ), which factorises, so
    #   H(q_obs) = H(m_{x→obs}) + H(q_θ).
    # The separator q(x,θ) = q_full(x) · q(θ) under mean-field, so
    #   H(q_sep) = H(q_full(x)) + H(q_θ).
    # Correction per t = H(m_{x→obs}) - H(q_full(x)).
    obs_corr = 0.0
    for t in range(1, horizon + 1):
        # Cavity message to x_t (excluding obs contribution)
        m_x = q_state[t] * backward_msgs[t]
        m_x = m_x / (m_x.sum() + EPSILON)
        obs_corr += float(h(m_x)) - float(h(q_full[t]))

    total = VFE + dyn_corr + obs_corr

    return {
        "total": total,
        "VFE": VFE,
        "E_dyn": E_dyn,
        "E_goal": E_goal,
        "E_u": E_u,
        "E_theta": E_theta,
        "H_MF": H_MF,
        "H_theta": H_theta,
        "H_states": H_states,
        "H_actions": H_actions,
        "dyn_correction": dyn_corr,
        "obs_correction": obs_corr,
    }


def run_aif_iterations(
    q_current_state, q_static_state, transition_idx, observation_idx,
    goal, horizon, n_iterations,
):
    """Run AIF planning iterations, returning FE terms at each iteration.

    Mirrors aif_planning_indexed body_fn but unrolls in Python and
    computes free energy after each forward-backward pass.
    Uses per-timestep channels r_x[t] and r_y[t] with cavity messages.
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_idx.shape[2]

    action_prior = jnp.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0])

    q_u = jnp.tile(action_prior, (horizon, 1))
    q_state = jnp.concatenate([
        q_current_state[None, :],
        jnp.ones((horizon, n_states)) / n_states,
    ], axis=0)
    q_theta = q_static_state

    r_x = jnp.ones((horizon, n_states, n_states, n_actions)) / n_states
    r_y = jnp.ones((horizon + 1, N_FOV, N_CELL_TYPES, n_states, n_static)) / N_CELL_TYPES

    # Initialize per-timestep factor→θ messages to zero (cavities = prior)
    log_dyn_msgs_per_t = jnp.zeros((horizon, n_static))
    log_obs_msgs_per_t = jnp.zeros((horizon, n_static))

    obs_idx_flat = observation_idx.reshape(N_FOV, n_states, n_static)
    prior_theta = q_static_state
    log_prior = jnp.log(prior_theta + EPSILON)

    fe_history = []
    action_history = []

    for _ in range(n_iterations):
        # Step 1: Compute per-timestep cavity messages for θ
        cavities_dyn, cavities_obs = compute_cavities(
            log_prior, log_dyn_msgs_per_t, log_obs_msgs_per_t
        )

        # Step 2: Per-timestep reduced tensors using cavity_dyn[t]
        reduced_per_t = jax.vmap(
            lambda q: marginalize_static_indexed(transition_idx, q, n_states)
        )(cavities_dyn)

        # Step 3: Per-timestep modified kernel K_mod[t] = T_t / r_x[t]
        K_mod = compute_modified_kernel(reduced_per_t, r_x)

        # Step 4: θ for obs: uniform at t=0 (no obs factor), cavities for t=1..T
        uniform_theta = jnp.ones(n_static) / n_static
        q_theta_for_obs = jnp.concatenate(
            [uniform_theta[None, :], cavities_obs], axis=0
        )

        # Step 5: Per-timestep obs messages to x using per-timestep θ cavities
        log_obs_msgs_to_x = compute_all_obs_msgs_to_x(obs_idx_flat, r_y, q_theta_for_obs)
        maxes = log_obs_msgs_to_x.max(axis=1, keepdims=True)
        log_obs_msgs_to_x = log_obs_msgs_to_x - maxes
        # No observation factor at x₀ (degree 2: only p(x₀) and dyn_1)
        log_obs_msgs_to_x = log_obs_msgs_to_x.at[0].set(0.0)

        # Step 6: Forward pass
        q_state = aif_forward_pass(K_mod, q_state, action_prior, log_obs_msgs_to_x, horizon)

        # Step 7: Backward pass
        q_u, backward_msgs = aif_backward_pass_with_messages(
            K_mod, q_state, goal, action_prior, log_obs_msgs_to_x, horizon
        )

        # --- Compute FE after forward-backward ---
        # Use the first timestep's reduced tensor for FE computation
        reduced = reduced_per_t[0]
        fe = compute_fe_terms(
            q_state, q_u, q_theta, reduced, K_mod,
            backward_msgs, log_obs_msgs_to_x, goal, action_prior,
            prior_theta, horizon,
        )
        fe_history.append(fe)
        action_history.append(q_u[0].copy())

        # Step 8: Per-timestep obs→θ messages (uses backward_msgs)
        log_obs_msgs_per_t = compute_obs_msgs_to_theta_per_t(
            obs_idx_flat, r_y, q_state, backward_msgs, horizon
        )

        # Step 9: Per-timestep dyn→θ messages
        log_dyn_msgs_per_t = compute_theta_messages_from_dynamics(
            transition_idx, q_state, backward_msgs,
            action_prior, r_x, log_obs_msgs_to_x, horizon
        )

        # Step 10: Update q_theta from per-timestep messages
        q_theta = update_q_theta(prior_theta, log_dyn_msgs_per_t, log_obs_msgs_per_t)

        # Step 11: Update per-timestep dynamics channel
        r_x = channel_update_dynamics(
            K_mod, q_state, q_u, backward_msgs, log_obs_msgs_to_x, horizon
        )

        # Step 12: Update per-timestep obs channel
        r_y = channel_update_obs(obs_idx_flat, n_states, n_static, horizon)

    return fe_history, action_history


def plot_convergence(fe_history, action_history, output_path, step_label=""):
    """Create 2x2 convergence plot."""
    n_iters = len(fe_history)
    iters = list(range(n_iters))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"AIF Planning Free Energy Convergence{step_label}", fontsize=14)

    # Top-left: Total FE and components
    ax = axes[0, 0]
    ax.plot(iters, [f["total"] for f in fe_history], "k-o", label="Total (VFE + corr)", linewidth=2, markersize=4)
    ax.plot(iters, [f["VFE"] for f in fe_history], "b--", label="VFE (mean-field)")
    ax.plot(iters, [f["dyn_correction"] for f in fe_history], "r--", label="Dyn correction")
    ax.plot(iters, [f["obs_correction"] for f in fe_history], "g--", label="Obs correction")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Free Energy")
    ax.set_title("Free Energy Decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Energy components
    ax = axes[0, 1]
    ax.plot(iters, [f["E_dyn"] for f in fe_history], "b-o", label="E_dyn", markersize=4)
    ax.plot(iters, [f["E_goal"] for f in fe_history], "r-s", label="E_goal", markersize=4)
    ax.plot(iters, [f["E_u"] for f in fe_history], "g-^", label="E_u (prior)", markersize=4)
    ax.plot(iters, [f["E_theta"] for f in fe_history], "m-d", label="E_theta (prior)", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Entropy components
    ax = axes[1, 0]
    ax.plot(iters, [f["H_MF"] for f in fe_history], "k-o", label="H_MF (total)", linewidth=2, markersize=4)
    ax.plot(iters, [f["H_states"] for f in fe_history], "b--", label="H_states")
    ax.plot(iters, [f["H_actions"] for f in fe_history], "r--", label="H_actions")
    ax.plot(iters, [f["H_theta"] for f in fe_history], "g--", label="H_theta")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.set_title("Mean-field Entropy Components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Action distribution at t=0
    ax = axes[1, 1]
    for a_idx in range(len(ACTION_NAMES)):
        probs = [float(ah[a_idx]) for ah in action_history]
        if max(probs) > 0.01:
            ax.plot(iters, probs, "-o", label=ACTION_NAMES[a_idx], markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Probability")
    ax.set_title("Action Distribution q(u_0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="AIF planning free energy convergence analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_convergence.py --grid-size 3 --seed 0
  python run_convergence.py --grid-size 3 --seed 0 --planning-iterations 50
  python run_convergence.py --grid-size 3 --seed 0 --env-steps 3
""",
    )
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--planning-horizon", type=int, default=15)
    parser.add_argument("--inference-iterations", type=int, default=10)
    parser.add_argument("--planning-iterations", type=int, default=20)
    parser.add_argument(
        "--env-steps", type=int, default=1,
        help="Number of environment steps to analyze (default: 1)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="convergence_plots",
        help="Directory for output plots (default: convergence_plots)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    grid_size = args.grid_size
    minigrid_size = grid_size + 2
    env_name = f"MiniGrid-DoorKey-{minigrid_size}x{minigrid_size}-v0"
    dims = get_dimensions(grid_size)

    print(f"JAX devices: {jax.devices()}")
    print(f"Grid: {grid_size}x{grid_size}")
    print(f"Seed: {args.seed}")
    print(f"Planning: horizon={args.planning_horizon}, iterations={args.planning_iterations}")
    print(f"Env steps to analyze: {args.env_steps}")
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

    # Initial beliefs
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

    # Set up environment
    env = MiniGridWrapper(env_name=env_name, max_steps=args.max_steps)
    result = env.reset(seed=args.seed)

    q_state = state_probs
    q_static = static_probs
    last_action = 0

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.env_steps):
        # State inference
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

        horizon = args.planning_horizon

        print(f"Step {step}: Running {args.planning_iterations} AIF iterations (horizon={horizon})...")
        t0 = time.time()
        fe_history, action_history = run_aif_iterations(
            q_state, q_static, transition_idx, observation_idx,
            goal, horizon, args.planning_iterations,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.2f}s")

        # Print table
        header = f"  {'Iter':>4} | {'Total':>10} | {'VFE':>10} | {'DynCorr':>10} | {'ObsCorr':>10} | Action"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for i, fe in enumerate(fe_history):
            action_idx = int(jnp.argmax(action_history[i]))
            action_p = float(action_history[i][action_idx])
            print(
                f"  {i:>4} | {fe['total']:>10.4f} | {fe['VFE']:>10.4f} | "
                f"{fe['dyn_correction']:>10.4f} | {fe['obs_correction']:>10.4f} | "
                f"{ACTION_NAMES[action_idx]}={action_p:.3f}"
            )

        # Convergence check
        if len(fe_history) >= 2:
            delta = abs(fe_history[-1]["total"] - fe_history[-2]["total"])
            print(f"  Final |delta|: {delta:.6f}")

        # Plot
        plot_path = output_dir / f"convergence_step{step}.png"
        plot_convergence(
            fe_history, action_history, plot_path,
            step_label=f" (env step {step})",
        )

        # Take env action (use final iteration's chosen action)
        env_action = int(jnp.argmax(action_history[-1]))
        last_action = env_action
        result = env.step(env_action)

        if result.terminated or result.truncated:
            print(f"\nEpisode ended at step {step}")
            break

    env.close()
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
