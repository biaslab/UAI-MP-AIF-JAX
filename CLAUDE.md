# MinigridMP-AIF-JAX

Active Inference agents using message passing on factor graphs, implemented in JAX. Compares multiple belief propagation variants for goal-directed planning across four environments.

## Project Structure

```
agents/              # Agent implementations per environment
environments/        # Tensor generation & environment wrappers
inference/           # Planning algorithms (7 loopy BP variants) & state inference
utils/              # Index conversion utilities
tests/              # Unit & integration tests
```

## Environments

| Environment | Entry point | Config key | Description |
|---|---|---|---|
| MiniGrid DoorKey | `run_minigrid.py` | `experiment` | Partially observable gridworld with key-door puzzle |
| Frozen Lake | `run_frozen_lake.py` | `frozen_lake` | Slippery gridworld with holes; configurable layouts |
| Wumpus World | `run_wumpus_world.py` | `wumpus_world` | Gridworld with pits and wumpus; event-gated SENSE action for breeze/stench/glitter |
| RockSample | `run_rocksample.py` | `rocksample` | Canonical RockSample; per-rock SENSE actions, event-gated distance-dependent observations |

Each environment also has `run_*_diagnostics.py` for single-episode inspection.

## Key Files

- `inference/loopy_bp.py` — Loopy BP with θ as variable node
- `inference/loopy_vbp.py` — Loopy VBP (value iteration, θ as variable)
- `inference/region_extended_loopy_bp.py` — Adds observation factors to planning graph
- `inference/dyn_channel_loopy_bp.py` — Dynamic channel loopy BP with observation factors
- `inference/nuijten_mp.py` — Region beliefs without kernels
- `inference/vbp_channel.py` — VBP with action channel reparameterization
- `inference/precise_info_seeking.py` — VBP action channels + obs channel reparameterization
- `inference/active_inference.py` — VBP action channels + dyn channels + obs channels (Active Inference)
- `inference/state_inference.py` — Loopy BP for Bayesian state estimation
- `inference/messages.py` — Low-level message operations + shared utilities (LOG_ZERO, safe_log, marginalize_static)
- `agents/flat_tensor_agent.py` — MiniGrid agents (8 planning variants)
- `agents/frozen_lake_agent.py` — Frozen Lake agents
- `agents/wumpus_agent.py` — Wumpus World agents
- `agents/rocksample_agent.py` — RockSample agents
- `environments/minigrid.py` — MiniGrid transition & observation tensor generation
- `environments/frozen_lake.py` — Frozen Lake environment
- `environments/wumpus_world.py` — Wumpus World environment
- `environments/rocksample.py` — RockSample environment
- `utils/tensors.py` — Index flattening, coordinate conversion, one-hot creation

## Planning Methods

Available via `--planning-method`:

| Method | Module | θ handling |
|---|---|---|
| `loopy` | `loopy_bp.py` | Variable node |
| `loopy-vbp` | `loopy_vbp.py` | Variable node (VBP) |
| `region-extended` | `region_extended_loopy_bp.py` | Variable node + obs factors |
| `dyn-channel` | `dyn_channel_loopy_bp.py` | Variable node + dynamic channels |
| `nuijten` | `nuijten_mp.py` | Variable node, no kernels |
| `vbp-channel` | `vbp_channel.py` | Variable node + action channels |
| `precise-info-seeking` | `precise_info_seeking.py` | Variable node + action channels + obs channels |
| `active-inference` | `active_inference.py` | Variable node + action channels + dyn channels + obs channels |

## Running

```bash
# MiniGrid experiment
uv run python run_minigrid.py --grid-size 3 --episodes 100 --planning-method region-extended

# Frozen Lake
uv run python run_frozen_lake.py --grid-size 5 --n-configs 10 --planning-method loopy

# Wumpus World
uv run python run_wumpus_world.py --grid-size 4 --n-configs 50 --planning-method dyn-channel

# RockSample
uv run python run_rocksample.py --grid-size 5 --n-rocks 3 --n-configs 8 --planning-method loopy

# Single-episode diagnostics (any environment)
uv run python run_frozen_lake_diagnostics.py --planning-method region-extended

# Convergence analysis
uv run python run_minigrid_convergence.py --damping 0.25 --obs-alpha 0.01

# Tests
uv run python run_tests.py
```

Use `--help` on any script for all parameters. Per-method iteration counts and damping are configured in `params.yaml`.

## Experiments & DVC

Experiments are managed with DVC. Configuration lives in `params.yaml`, pipeline in `dvc.yaml`. Results go to `data/results/`, videos to `data/videos/`, convergence plots to `data/convergence/`.

```bash
uv run dvc repro          # Run full pipeline
uv run dvc repro -s frozen_lake  # Run single stage
```

## Important Details

**MiniGrid agent representations:**
All agents use indexed tensor storage for efficient state inference. Eight planning variants available: `LoopyBPAgent`, `LoopyVBPAgent`, `RegionExtendedAgent`, `DynChannelLoopyBPAgent`, `NuijtenMPAgent`, `VBPChannelAgent`, `PreciseInfoSeekingAgent`, `ActiveInferenceAgent`.

**JAX requirements:**
- All inference functions are JIT-compiled (`@jax.jit`)
- First run is slow (compilation), subsequent runs are fast
- Horizon and iteration counts must be compile-time constants (static args)
- Use `jax.lax.fori_loop` inside JIT functions
- Computation is in log-space throughout to avoid underflow

**State representation (MiniGrid):**
- Dynamic state: (location, orientation, door_key_state) → flat index
- Static state: (key_position, door_position) → flat index
- See `flatten_state_index()` in `utils/tensors.py`

**Coordinates:** Our (x,y) matches MiniGrid's inner grid with wall offset subtracted: `our = mg - 1`.

## Testing

Run `uv run python run_tests.py` before committing. Tests cover tensor generation, inference algorithms, and agent integration for all four environments.

## Conventions

- Always run Python scripts with `uv run python` (e.g. `uv run python run_tests.py`)
