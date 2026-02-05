# MinigridMP-AIF-JAX

Active Inference agents for MiniGrid using JAX. Implements loopy belief propagation for state inference and message passing for goal-directed planning.

## Project Structure

```
agents/              # FlatTensorAgent & IndexedTensorAgent implementations
environments/        # Tensor generation & MiniGrid wrapper
inference/           # State inference (loopy BP) & planning algorithms
utils/              # Index conversion utilities
tests/              # Unit & integration tests
```

## Key Files

- [inference/state_inference.py](inference/state_inference.py) - Loopy BP for belief updates
- [inference/planning.py](inference/planning.py) - Forward-backward planning on temporal factor graph
- [agents/flat_tensor_agent.py](agents/flat_tensor_agent.py) - Both agent implementations
- [environments/minigrid.py](environments/minigrid.py) - Transition & observation tensor generation

## Running

```bash
# Run experiments
python run_experiment.py --grid-size 3 --episodes 100

# Run tests
python run_tests.py
```

Use `--help` for all parameters. Common: `--planning-horizon`, `--inference-iterations`, `--planning-iterations`, `--receding-horizon`.

## Important Details

**Two representations:**
- `IndexedTensorAgent` (default) - Memory-efficient, stores indices not full tensors
- `FlatTensorAgent` (debugging) - Full tensors, use only for grid_size ≤ 3

Both produce identical results. Always use indexed for production.

**JAX requirements:**
- All inference functions are JIT-compiled
- First run is slow (compilation), subsequent runs are fast
- Horizon and iteration counts must be compile-time constants
- Use `jax.lax.fori_loop` inside JIT functions

**State representation:**
- Dynamic: (location, orientation, door_key_state) → flat index
- Static: (key_position, door_position) → flat index
- See `flatten_state_index()` in [utils/tensors.py](utils/tensors.py)

**Coordinates:** Our (x,y) is y-flipped vs MiniGrid. Use conversions in [tests/test_minigrid_groundtruth.py](tests/test_minigrid_groundtruth.py).

## Testing

Run `python run_tests.py` before committing. Tests validate message passing, tensor generation against MiniGrid, and agent integration.
