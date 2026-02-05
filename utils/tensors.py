"""Tensor utilities and index conversions."""

import jax.numpy as jnp
import numpy as np
from pathlib import Path


def flatten_state_index(
    location: int,
    orientation: int,
    door_key_state: int,
    n_locations: int,
    n_orientations: int,
    n_door_key_states: int,
) -> int:
    """
    Flatten (location, orientation, door_key_state) to single index.
    
    Order: door_key_state changes fastest, then orientation, then location.
    Matches the tensor generation in environments/minigrid.py.
    """
    return (
        location * (n_orientations * n_door_key_states)
        + orientation * n_door_key_states
        + door_key_state
    )


def unflatten_state_index(
    flat_idx: int,
    n_locations: int,
    n_orientations: int,
    n_door_key_states: int,
) -> tuple[int, int, int]:
    """
    Unflatten single index to (location, orientation, door_key_state).
    
    Inverse of flatten_state_index.
    """
    door_key_state = flat_idx % n_door_key_states
    remainder = flat_idx // n_door_key_states
    orientation = remainder % n_orientations
    location = remainder // n_orientations
    return location, orientation, door_key_state


def flatten_static_index(
    key_pos: int, door_pos: int, n_key_positions: int, n_door_positions: int
) -> int:
    """Flatten (key_position, door_position) to single static index."""
    return key_pos + n_key_positions * door_pos


def unflatten_static_index(
    static_idx: int, n_key_positions: int, n_door_positions: int
) -> tuple[int, int]:
    """Unflatten static index to (key_position, door_position)."""
    key_pos = static_idx % n_key_positions
    door_pos = static_idx // n_key_positions
    return key_pos, door_pos


def coords_to_location(x: int, y: int, grid_size: int) -> int:
    """Convert (x, y) grid coordinates to location index."""
    return (x - 1) * grid_size + (y - 1)


def location_to_coords(location: int, grid_size: int) -> tuple[int, int]:
    """Convert location index to (x, y) grid coordinates."""
    x = location // grid_size + 1
    y = location % grid_size + 1
    return x, y


def create_onehot(index: int, size: int) -> jnp.ndarray:
    """Create a one-hot vector."""
    return jnp.eye(size)[index]


def load_tensors_from_julia(
    data_dir: Path,
) -> dict[str, jnp.ndarray]:
    """
    Load pre-computed tensors exported from Julia.
    
    Expected files in data_dir:
    - transition_tensor.npy
    - observation_tensors.npy
    - orientation_observation_tensor.npy
    
    Returns dict with JAX arrays.
    """
    tensors = {}
    
    transition_path = data_dir / "transition_tensor.npy"
    if transition_path.exists():
        tensors["transition_tensor"] = jnp.array(np.load(transition_path))
    
    obs_path = data_dir / "observation_tensors.npy"
    if obs_path.exists():
        tensors["observation_tensors"] = jnp.array(np.load(obs_path))
    
    ori_path = data_dir / "orientation_observation_tensor.npy"
    if ori_path.exists():
        tensors["orientation_observation_tensor"] = jnp.array(np.load(ori_path))
    
    return tensors


def get_dimensions(grid_size: int) -> dict[str, int]:
    """Get all dimension sizes for a given grid size."""
    n_locations = grid_size * grid_size
    n_orientations = 4
    n_door_key_states = 3
    n_key_positions = n_locations - 2 * grid_size
    n_door_positions = n_locations - 2 * grid_size
    n_states = n_locations * n_orientations * n_door_key_states
    n_static = n_key_positions * n_door_positions
    n_actions = 7
    
    return {
        "grid_size": grid_size,
        "n_locations": n_locations,
        "n_orientations": n_orientations,
        "n_door_key_states": n_door_key_states,
        "n_key_positions": n_key_positions,
        "n_door_positions": n_door_positions,
        "n_states": n_states,
        "n_static": n_static,
        "n_actions": n_actions,
    }
