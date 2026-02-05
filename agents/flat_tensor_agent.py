"""Flat tensor agent using JAX inference."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from inference.state_inference import state_inference_step, state_inference_step_indexed
from inference.planning import planning, planning_indexed
from utils.tensors import create_onehot, get_dimensions, flatten_state_index


@dataclass
class FlatTensorAgent:
    """
    Agent using flattened state representation.
    
    State = (location, orientation, door_key_state) combined into single index.
    Static = (key_position, door_position) combined into single index.
    """
    
    grid_size: int
    dims: dict[str, int]
    
    transition_tensor: jnp.ndarray
    observation_tensors: jnp.ndarray
    orientation_tensor: jnp.ndarray
    
    q_state: jnp.ndarray
    q_static: jnp.ndarray
    goal: jnp.ndarray
    
    planning_horizon: int
    n_inference_iterations: int
    n_planning_iterations: int
    
    last_action: int
    
    @classmethod
    def create(
        cls,
        grid_size: int,
        transition_tensor: jnp.ndarray,
        observation_tensors: jnp.ndarray,
        orientation_tensor: jnp.ndarray,
        goal: jnp.ndarray,
        planning_horizon: int = 10,
        n_inference_iterations: int = 10,
        n_planning_iterations: int = 10,
    ) -> "FlatTensorAgent":
        """Create a new agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        
        # Initial state: agent can be in any valid location, any orientation,
        # but door_key_state=0 (hasn't picked up key yet)
        n_valid_locations = dims["n_locations"] - 2 * grid_size
        state_probs = jnp.zeros(dims["n_states"])
        for loc in range(n_valid_locations):
            for ori in range(dims["n_orientations"]):
                idx = flatten_state_index(
                    loc, ori, 0,  # door_key_state=0
                    dims["n_locations"],
                    dims["n_orientations"],
                    dims["n_door_key_states"],
                )
                state_probs = state_probs.at[idx].set(1.0)
        state_probs = state_probs / state_probs.sum()
        
        static_probs = jnp.ones(dims["n_static"]) / dims["n_static"]
        
        return cls(
            grid_size=grid_size,
            dims=dims,
            transition_tensor=transition_tensor,
            observation_tensors=observation_tensors,
            orientation_tensor=orientation_tensor,
            q_state=state_probs,
            q_static=static_probs,
            goal=goal,
            planning_horizon=planning_horizon,
            n_inference_iterations=n_inference_iterations,
            n_planning_iterations=n_planning_iterations,
            last_action=0,
        )
    
    def reset(self) -> "FlatTensorAgent":
        """Reset beliefs to initial state."""
        dims = self.dims
        n_valid_locations = dims["n_locations"] - 2 * self.grid_size
        
        state_probs = jnp.zeros(dims["n_states"])
        for loc in range(n_valid_locations):
            for ori in range(dims["n_orientations"]):
                idx = flatten_state_index(
                    loc, ori, 0,
                    dims["n_locations"],
                    dims["n_orientations"],
                    dims["n_door_key_states"],
                )
                state_probs = state_probs.at[idx].set(1.0)
        state_probs = state_probs / state_probs.sum()
        
        static_probs = jnp.ones(dims["n_static"]) / dims["n_static"]
        
        return FlatTensorAgent(
            grid_size=self.grid_size,
            dims=self.dims,
            transition_tensor=self.transition_tensor,
            observation_tensors=self.observation_tensors,
            orientation_tensor=self.orientation_tensor,
            q_state=state_probs,
            q_static=static_probs,
            goal=self.goal,
            planning_horizon=self.planning_horizon,
            n_inference_iterations=self.n_inference_iterations,
            n_planning_iterations=self.n_planning_iterations,
            last_action=0,
        )
    
    def step(
        self,
        vision_obs: jnp.ndarray,
        orientation_obs: jnp.ndarray,
        time_remaining: int,
    ) -> tuple[int, "FlatTensorAgent"]:
        """
        Execute one agent step: perceive, infer, plan, act.
        
        Args:
            vision_obs: (7, 7, 11) one-hot vision observations
            orientation_obs: (4,) one-hot orientation
            time_remaining: steps remaining in episode
            
        Returns:
            action: selected action index
            new_agent: updated agent state
        """
        action_onehot = create_onehot(self.last_action, self.dims["n_actions"])
        
        q_current, q_static = state_inference_step(
            q_old_state=self.q_state,
            q_static_state=self.q_static,
            transition_tensor=self.transition_tensor,
            obs_tensors=self.observation_tensors,
            ori_tensor=self.orientation_tensor,
            vision_obs=vision_obs,
            ori_obs=orientation_obs,
            action_onehot=action_onehot,
            n_iterations=self.n_inference_iterations,
        )
        
        horizon = min(time_remaining, self.planning_horizon)
        action_dist = planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=self.transition_tensor,
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
        )
        
        action = int(jnp.argmax(action_dist))
        
        new_agent = FlatTensorAgent(
            grid_size=self.grid_size,
            dims=self.dims,
            transition_tensor=self.transition_tensor,
            observation_tensors=self.observation_tensors,
            orientation_tensor=self.orientation_tensor,
            q_state=q_current,
            q_static=q_static,
            goal=self.goal,
            planning_horizon=self.planning_horizon,
            n_inference_iterations=self.n_inference_iterations,
            n_planning_iterations=self.n_planning_iterations,
            last_action=action,
        )
        
        return action, new_agent


@dataclass
class IndexedTensorAgent:
    """
    Memory-efficient agent using index-based tensor representation.
    
    Instead of storing full one-hot tensors, stores index arrays that map
    (state, static, action) -> next_state directly. This reduces memory
    by ~100-400x while maintaining identical behavior.
    """
    
    grid_size: int
    dims: dict[str, int]
    
    # Index-based tensors (much smaller than full tensors)
    transition_idx: jnp.ndarray  # (n_states, n_static, n_actions) -> next_state
    observation_idx: jnp.ndarray  # (7, 7, n_states, n_static) -> cell_type
    orientation_idx: jnp.ndarray  # (n_states,) -> orientation
    
    q_state: jnp.ndarray
    q_static: jnp.ndarray
    goal: jnp.ndarray
    
    planning_horizon: int
    n_inference_iterations: int
    n_planning_iterations: int
    
    last_action: int
    
    @classmethod
    def create(
        cls,
        grid_size: int,
        transition_idx: jnp.ndarray,
        observation_idx: jnp.ndarray,
        orientation_idx: jnp.ndarray,
        goal: jnp.ndarray,
        planning_horizon: int = 10,
        n_inference_iterations: int = 10,
        n_planning_iterations: int = 10,
    ) -> "IndexedTensorAgent":
        """Create a new agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        
        # Initial state: agent can be in any valid location, any orientation,
        # but door_key_state=0 (hasn't picked up key yet)
        n_valid_locations = dims["n_locations"] - 2 * grid_size
        state_probs = jnp.zeros(dims["n_states"])
        for loc in range(n_valid_locations):
            for ori in range(dims["n_orientations"]):
                idx = flatten_state_index(
                    loc, ori, 0,  # door_key_state=0
                    dims["n_locations"],
                    dims["n_orientations"],
                    dims["n_door_key_states"],
                )
                state_probs = state_probs.at[idx].set(1.0)
        state_probs = state_probs / state_probs.sum()
        
        static_probs = jnp.ones(dims["n_static"]) / dims["n_static"]
        
        return cls(
            grid_size=grid_size,
            dims=dims,
            transition_idx=transition_idx,
            observation_idx=observation_idx,
            orientation_idx=orientation_idx,
            q_state=state_probs,
            q_static=static_probs,
            goal=goal,
            planning_horizon=planning_horizon,
            n_inference_iterations=n_inference_iterations,
            n_planning_iterations=n_planning_iterations,
            last_action=0,
        )
    
    def reset(self) -> "IndexedTensorAgent":
        """Reset beliefs to initial state."""
        dims = self.dims
        n_valid_locations = dims["n_locations"] - 2 * self.grid_size
        
        state_probs = jnp.zeros(dims["n_states"])
        for loc in range(n_valid_locations):
            for ori in range(dims["n_orientations"]):
                idx = flatten_state_index(
                    loc, ori, 0,
                    dims["n_locations"],
                    dims["n_orientations"],
                    dims["n_door_key_states"],
                )
                state_probs = state_probs.at[idx].set(1.0)
        state_probs = state_probs / state_probs.sum()
        
        static_probs = jnp.ones(dims["n_static"]) / dims["n_static"]
        
        return IndexedTensorAgent(
            grid_size=self.grid_size,
            dims=self.dims,
            transition_idx=self.transition_idx,
            observation_idx=self.observation_idx,
            orientation_idx=self.orientation_idx,
            q_state=state_probs,
            q_static=static_probs,
            goal=self.goal,
            planning_horizon=self.planning_horizon,
            n_inference_iterations=self.n_inference_iterations,
            n_planning_iterations=self.n_planning_iterations,
            last_action=0,
        )
    
    def step(
        self,
        vision_obs: jnp.ndarray,
        orientation_obs: jnp.ndarray,
        time_remaining: int,
    ) -> tuple[int, "IndexedTensorAgent"]:
        """
        Execute one agent step: perceive, infer, plan, act.
        
        Args:
            vision_obs: (7, 7, 11) soft vision observations
            orientation_obs: (4,) soft orientation observation
            time_remaining: steps remaining in episode
            
        Returns:
            action: selected action index
            new_agent: updated agent state
        """
        q_current, q_static = state_inference_step_indexed(
            q_old_state=self.q_state,
            q_static_state=self.q_static,
            transition_idx=self.transition_idx,
            obs_idx=self.observation_idx,
            ori_idx=self.orientation_idx,
            vision_obs=vision_obs,
            ori_obs=orientation_obs,
            action_idx=self.last_action,
            n_iterations=self.n_inference_iterations,
        )
        
        horizon = min(time_remaining, self.planning_horizon)
        action_dist = planning_indexed(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_idx=self.transition_idx,
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
        )
        
        action = int(jnp.argmax(action_dist))
        
        new_agent = IndexedTensorAgent(
            grid_size=self.grid_size,
            dims=self.dims,
            transition_idx=self.transition_idx,
            observation_idx=self.observation_idx,
            orientation_idx=self.orientation_idx,
            q_state=q_current,
            q_static=q_static,
            goal=self.goal,
            planning_horizon=self.planning_horizon,
            n_inference_iterations=self.n_inference_iterations,
            n_planning_iterations=self.n_planning_iterations,
            last_action=action,
        )
        
        return action, new_agent
