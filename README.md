# MinigridMP-AIF-JAX

Message passing implementation for planning in MiniGrid's DoorKey environment using JAX.

## Overview

This project implements discrete factor graph inference for goal-directed planning in partially observable grid worlds. Features:

- **Loopy Belief Propagation** for state inference
- **Multiple message-passing schemes** for action planning (see below)
- **Memory-efficient indexed tensor representation** (100-400x smaller than full tensors)

## Planning Methods

| Flag | Method | Description |
|------|--------|-------------|
| `bp` | Standard BP | Marginalizes static parameter θ once; forward-backward on temporal graph |
| `loopy` | Loopy BP | Treats θ as a variable node in the factor graph |
| `region-extended` | Region-extended loopy BP | Adds observation factors to the planning graph |
| `reduced-aif` | Reduced region-extended | Fixed θ with kernel reparameterization + observation factors |
| `nuijten` | Nuijten MP | Region beliefs without kernels, θ inferred |
| `reduced-nuijten` | Reduced Nuijten MP | Region beliefs without kernels, θ fixed |

Select a method with `--planning-method <flag>`.

## Quick Start

```bash
# Install dependencies
uv sync

# Run basic experiment (5x5 grid, 100 episodes, standard BP)
uv run python run_experiment.py --grid-size 3 --episodes 100

# Run with a different planning method
uv run python run_experiment.py --grid-size 3 --episodes 100 --planning-method region-extended

# Run all tests
uv run python run_tests.py
```

## Experiments

Experiments are managed with [DVC](https://dvc.org). All shared parameters live in `params.yaml` and the pipeline is defined in `dvc.yaml`.

```bash
# Install dev dependencies (includes DVC)
uv sync --group dev

# Run all 6 planning methods
uv run dvc repro

# Compare results
uv run dvc metrics show
uv run dvc metrics diff
```

To tweak parameters, edit `params.yaml` and re-run `uv run dvc repro` — DVC only re-runs stages whose parameters or code changed. Results (JSON) are git-tracked in `data/results/`; videos are DVC-cached in `data/videos/`.

### Running a single method

```bash
uv run python run_experiment.py --grid-size 3 --episodes 100 \
    --planning-method bp \
    --planning-horizon 15 \
    --inference-iterations 10 \
    --planning-iterations 10 \
    --fov-size 7 \
    --seed 0
```

### Key options

- `--grid-size N` — Internal grid size (MiniGrid size = N+2)
- `--planning-horizon N` — Lookahead depth
- `--receding-horizon` — Decrease horizon as episode time runs out
- `--fov-size N` — Field-of-view size (odd, >= 3)
- `--no-orientation` — Replace orientation observation with uniform (agent must infer orientation)
- `--record first,last` — Record episodes to video
- `--output results.json` — Save results to JSON

## Diagnostics

`run_diagnostics.py` runs a single episode and prints full internal state at every step: beliefs, observations, inference/planning timing, action distributions, and entropy.

```bash
uv run python run_diagnostics.py --grid-size 3 --seed 0 --planning-method bp
```

## Key Features

- **JAX-based**: JIT-compiled inference for GPU acceleration
- **Two representations**: Full tensors (debugging) and indexed tensors (production)
- **Validated**: Tests compare against MiniGrid ground truth

## Project Structure

```
agents/         # Agent implementations
environments/   # MiniGrid tensor generation & gym wrapper
inference/      # State inference & planning algorithms
tests/          # Unit & integration tests
```

See [CLAUDE.md](CLAUDE.md) for detailed technical documentation.

## Requirements

- Python 3.10+
- JAX
- MiniGrid
- uv (recommended) or pip
