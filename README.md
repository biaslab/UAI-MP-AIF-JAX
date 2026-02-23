# What Type of Inference Is Active Inference?

Companion code for the paper *"What Type of Inference Is Active Inference?"*. Implements Active Inference agents using message passing on factor graphs in JAX, comparing multiple belief propagation variants for goal-directed planning across four environments.

## Reproducing Paper Results

```bash
# Install dependencies
uv sync

# Run all experiments (Frozen Lake, Wumpus World, RockSample)
uv run dvc repro

# View results
uv run dvc metrics show
```

Configuration lives in `params.yaml`, the pipeline in `dvc.yaml`. Results go to `data/results/`. DVC only re-runs stages whose parameters or code changed.

## Planning Methods

Available via `--planning-method`:

| Flag | Paper name | Module | Description |
|---|---|---|---|
| `bp` | Standard BP | `planning.py` | Marginalizes static parameter once; forward-backward on temporal graph |
| `vbp` | Value BP | `vbp.py` | Value iteration variant (temperature to zero) |
| `loopy-vbp` | Loopy VBP | `loopy_vbp.py` | Loopy BP with value iteration messages |
| `loopy` | Loopy BP | `loopy_bp.py` | Treats static parameter as variable node |
| `region-extended` | Active Inference | `region_extended_loopy_bp.py` | Adds observation factors to planning graph |
| `reduced-region-extended` | Reduced Active Inference | `reduced_region_extended.py` | Fixed parameter + kernel reparameterization + observation factors |
| `dyn-channel` | Risk-minimizing | `dyn_channel_loopy_bp.py` | Dynamic channel messages |
| `reduced-dyn-channel` | Reduced risk-minimizing | `reduced_dyn_channel.py` | Fixed parameter + dynamic channels |
| `nuijten` | Nuijten MP | `nuijten_mp.py` | Region beliefs without kernels |
| `reduced-nuijten` | Reduced Nuijten MP | `nuijten_mp.py` | Fixed parameter, no kernels |

## Environments

### Frozen Lake

Slippery gridworld where the agent must reach a goal while avoiding holes. Hole layouts are randomized across configurations; observations are noisy and distance-dependent.

```bash
uv run python run_frozen_lake.py --grid-size 5 --n-configs 10 --episodes 1000 \
    --planning-method region-extended --planning-horizon 15 --damping 0.25
```

Environment-specific arguments: `--n-configs`, `--hole-fraction`, `--min-hamming`, `--base-noise`, `--noise-range`, `--slip-prob`, `--hole-penalty`, `--goal-temperature`, `--scan-cost`.

### Wumpus World

Gridworld with pits and a wumpus. The agent receives indirect observations (stench, breeze) about neighboring cells. Multiple static configurations vary pit and wumpus placement.

```bash
uv run python run_wumpus_world.py --grid-size 4 --n-configs 50 --episodes 1000 \
    --planning-method dyn-channel --planning-horizon 7 --damping 0.25
```

Environment-specific arguments: `--n-configs`, `--n-pits`, `--obs-noise`, `--pos-noise`, `--slip-prob`, `--pit-penalty`, `--wumpus-penalty`, `--goal-temperature`, `--scan-cost`.

### RockSample

Gridworld with rocks of unknown quality. The agent can check rocks (distance-dependent observation accuracy), sample them, or move to the exit. Static configurations vary rock placement and quality.

```bash
uv run python run_rocksample.py --grid-size 5 --n-rocks 3 --n-configs 8 --episodes 100 \
    --planning-method bp --planning-horizon 10
```

Environment-specific arguments: `--n-rocks`, `--n-configs`, `--half-eff-dist`, `--pos-noise`, `--slip-prob`, `--good-reward`, `--bad-penalty`, `--exit-reward`, `--goal-temperature`, `--scan-cost`, `--sample-cost`, `--terminal-goal-only`.

### MiniGrid DoorKey

Partially observable gridworld with a key-door puzzle. The agent has a limited field of view and must pick up a key to unlock a door. This environment is not part of the DVC pipeline.

```bash
uv run python run_minigrid.py --grid-size 3 --episodes 100 \
    --planning-method region-extended --fov-size 7
```

Environment-specific arguments: `--fov-size`, `--no-orientation`, `--inference-iterations`, `--obs-alpha`, `--full-tensors`, `--record`, `--video-dir`.

## Common CLI Arguments

All environments share these arguments:

| Flag | Description |
|---|---|
| `--grid-size N` | Grid size |
| `--episodes N` | Number of episodes |
| `--max-steps N` | Maximum steps per episode |
| `--planning-horizon N` | Lookahead depth |
| `--planning-iterations N` | Number of message-passing iterations |
| `--planning-method METHOD` | Planning algorithm (see table above) |
| `--damping F` | Channel update damping (1.0 = no damping) |
| `--receding-horizon` | Decrease horizon as episode time runs out |
| `--seed N` | Random seed |
| `--verbose` | Print per-step details |
| `--output FILE` | Save results to JSON |

Use `--help` on any script for all parameters. Per-method iteration counts and damping are configured in `params.yaml`.

## Diagnostics

Each environment has a `run_*_diagnostics.py` script that runs a single episode and prints full internal state at every step: beliefs, observations, timing, action distributions, and entropy.

```bash
uv run python run_frozen_lake_diagnostics.py --planning-method region-extended
uv run python run_wumpus_world_diagnostics.py --planning-method loopy
uv run python run_rocksample_diagnostics.py --planning-method bp
uv run python run_minigrid_diagnostics.py --grid-size 3 --planning-method bp
```

## Project Structure

```
agents/              # Agent implementations per environment
environments/        # Tensor generation & environment wrappers
inference/           # Planning algorithms & state inference
utils/               # Index conversion utilities
tests/               # Unit & integration tests
```

## Testing

```bash
uv run python run_tests.py
```

Tests cover tensor generation, inference algorithms, and agent integration for all four environments.
