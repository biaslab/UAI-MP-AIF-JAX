# MinigridMP-AIF-JAX

Message passing implementation for planning in MiniGrid's DoorKey environment using JAX.

## Overview

This project implements discrete factor graph inference for goal-directed planning in partially observable grid worlds. Currently features:

- **Loopy Belief Propagation** for state inference
- **Forward-Backward Message Passing** for action planning
- **Memory-efficient indexed tensor representation** (100-400x smaller than full tensors)

**In Progress:** Implementing full Active Inference with region-extended Bethe approximation (see [.claude/skills/](.claude/skills/) for mathematical framework).

## Quick Start

```bash
# Install dependencies
uv sync

# Run basic experiment (5x5 grid, 100 episodes)
uv run python run_experiment.py --grid-size 3 --episodes 100

# Run all tests
uv run python run_tests.py
```

## Key Features

- **JAX-based**: JIT-compiled inference for GPU acceleration
- **Two representations**: Full tensors (debugging) and indexed tensors (production)
- **Validated**: Tests compare against MiniGrid ground truth

## Project Structure

```
agents/         # Agent implementations (FlatTensor & IndexedTensor)
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
