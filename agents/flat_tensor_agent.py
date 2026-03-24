"""Flat tensor agent using JAX inference."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from inference.state_inference import state_inference_step
from inference.loopy_vbp import loopy_vbp_planning
from inference.loopy_bp import loopy_bp_planning
from inference.region_extended_loopy_bp import region_extended_loopy_bp_planning
from inference.dyn_channel_loopy_bp import dyn_channel_loopy_bp_planning
from inference.nuijten_mp import nuijten_mp_planning
from inference.vbp_channel import vbp_channel_planning
from inference.precise_info_seeking import precise_info_seeking_planning
from inference.active_inference import active_inference_planning
from utils.tensors import create_onehot, get_dimensions, flatten_state_index


def _flatten_obs_tensor(obs_tensors):
    """Flatten 5D obs tensor (fov_w, fov_h, n_types, n_states, n_static) to 4D."""
    if obs_tensors.ndim == 5:
        fov_w, fov_h = obs_tensors.shape[0], obs_tensors.shape[1]
        return obs_tensors.reshape(fov_w * fov_h, *obs_tensors.shape[2:])
    return obs_tensors  # already 4D


@dataclass
class LoopyVBPAgent:
    """
    Agent using loopy VBP planning with θ as a variable node.

    Like LoopyBPAgent but uses VBP (ε→0): max over actions in all messages
    instead of logsumexp.
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
    ) -> "LoopyVBPAgent":
        """Create a new loopy VBP agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        dims["n_static"] = transition_tensor.shape[2]

        n_valid_locations = dims["n_locations"] - 2 * grid_size
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

    def reset(self) -> "LoopyVBPAgent":
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

        return LoopyVBPAgent(
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
    ) -> tuple[int, "LoopyVBPAgent"]:
        """
        Execute one agent step: perceive (standard BP), plan (loopy VBP), act.
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
        action_dist = loopy_vbp_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=self.transition_tensor,
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
        )

        action = int(jnp.argmax(action_dist))

        new_agent = LoopyVBPAgent(
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
class LoopyBPAgent:
    """
    Agent using loopy BP planning with θ as a variable node.

    Uses loopy_bp_planning which keeps θ as a variable node and iterates
    cavity messages.
    """

    grid_size: int
    dims: dict[str, int]

    transition_tensor: jnp.ndarray  # (n_states, n_states, n_static, n_actions)
    observation_tensors: jnp.ndarray  # (fov_w, fov_h, N_CELL_TYPES, n_states, n_static)
    orientation_tensor: jnp.ndarray  # (4, n_states)

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
    ) -> "LoopyBPAgent":
        """Create a new loopy BP agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        dims["n_static"] = transition_tensor.shape[2]

        n_valid_locations = dims["n_locations"] - 2 * grid_size
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

    def reset(self) -> "LoopyBPAgent":
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

        return LoopyBPAgent(
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
    ) -> tuple[int, "LoopyBPAgent"]:
        """
        Execute one agent step: perceive (standard BP), plan (loopy BP), act.
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
        action_dist = loopy_bp_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=self.transition_tensor,
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
        )

        action = int(jnp.argmax(action_dist))

        new_agent = LoopyBPAgent(
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
class RegionExtendedAgent:
    """
    Agent using region-extended loopy BP planning with observation factors.
    """

    grid_size: int
    dims: dict[str, int]

    transition_tensor: jnp.ndarray  # (n_states, n_states, n_static, n_actions)
    observation_tensors: jnp.ndarray  # (fov_w, fov_h, N_CELL_TYPES, n_states, n_static)
    orientation_tensor: jnp.ndarray  # (4, n_states)

    q_state: jnp.ndarray
    q_static: jnp.ndarray
    goal: jnp.ndarray

    planning_horizon: int
    n_inference_iterations: int
    n_planning_iterations: int

    last_action: int

    damping: float

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
        damping: float = 1.0,
    ) -> "RegionExtendedAgent":
        """Create a new region-extended agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        dims["n_static"] = transition_tensor.shape[2]

        n_valid_locations = dims["n_locations"] - 2 * grid_size
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
            damping=damping,
        )

    def reset(self) -> "RegionExtendedAgent":
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

        return RegionExtendedAgent(
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
            damping=self.damping,
        )

    def step(
        self,
        vision_obs: jnp.ndarray,
        orientation_obs: jnp.ndarray,
        time_remaining: int,
    ) -> tuple[int, "RegionExtendedAgent"]:
        """
        Execute one agent step: perceive (standard BP), plan (region-extended), act.
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
        action_dist, _, _ = region_extended_loopy_bp_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=self.transition_tensor,
            observation_tensor=_flatten_obs_tensor(self.observation_tensors),
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
            damping=self.damping,
        )

        action = int(jnp.argmax(action_dist))

        new_agent = RegionExtendedAgent(
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
            damping=self.damping,
        )

        return action, new_agent


@dataclass
class DynChannelLoopyBPAgent:
    """
    Agent using dyn-channel loopy BP planning with observation factors.
    Only dynamics factors get channel reparameterization; obs use raw B.
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

    damping: float

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
        damping: float = 1.0,
    ) -> "DynChannelLoopyBPAgent":
        """Create a new dyn-channel loopy BP agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        dims["n_static"] = transition_tensor.shape[2]

        n_valid_locations = dims["n_locations"] - 2 * grid_size
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
            damping=damping,
        )

    def reset(self) -> "DynChannelLoopyBPAgent":
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

        return DynChannelLoopyBPAgent(
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
            damping=self.damping,
        )

    def step(
        self,
        vision_obs: jnp.ndarray,
        orientation_obs: jnp.ndarray,
        time_remaining: int,
    ) -> tuple[int, "DynChannelLoopyBPAgent"]:
        """
        Execute one agent step: perceive (standard BP), plan (dyn-channel), act.
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
        action_dist, _ = dyn_channel_loopy_bp_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=self.transition_tensor,
            observation_tensor=_flatten_obs_tensor(self.observation_tensors),
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
            damping=self.damping,
        )

        action = int(jnp.argmax(action_dist))

        new_agent = DynChannelLoopyBPAgent(
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
            damping=self.damping,
        )

        return action, new_agent


@dataclass
class VBPChannelAgent:
    """
    Agent using VBP channel planning with action channel reparameterization.
    Theta is inferred via cavity messages. Obs use raw B.
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

    damping: float

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
        damping: float = 1.0,
    ) -> "VBPChannelAgent":
        """Create a new VBP channel agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        dims["n_static"] = transition_tensor.shape[2]

        n_valid_locations = dims["n_locations"] - 2 * grid_size
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
            damping=damping,
        )

    def reset(self) -> "VBPChannelAgent":
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

        return VBPChannelAgent(
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
            damping=self.damping,
        )

    def step(
        self,
        vision_obs: jnp.ndarray,
        orientation_obs: jnp.ndarray,
        time_remaining: int,
    ) -> tuple[int, "VBPChannelAgent"]:
        """
        Execute one agent step: perceive (standard BP), plan (VBP channel), act.
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
        action_dist, _ = vbp_channel_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=self.transition_tensor,
            observation_tensor=_flatten_obs_tensor(self.observation_tensors),
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
            damping=self.damping,
        )

        action = int(jnp.argmax(action_dist))

        new_agent = VBPChannelAgent(
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
            damping=self.damping,
        )

        return action, new_agent


@dataclass
class PreciseInfoSeekingAgent:
    """
    Agent using precise info-seeking planning: VBP action channels + obs channels.
    Combines VBP-style action channel reparameterization with region-extended
    observation channel reparameterization.
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

    damping: float

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
        damping: float = 1.0,
    ) -> "PreciseInfoSeekingAgent":
        """Create a new precise info-seeking agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        dims["n_static"] = transition_tensor.shape[2]

        n_valid_locations = dims["n_locations"] - 2 * grid_size
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
            damping=damping,
        )

    def reset(self) -> "PreciseInfoSeekingAgent":
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

        return PreciseInfoSeekingAgent(
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
            damping=self.damping,
        )

    def step(
        self,
        vision_obs: jnp.ndarray,
        orientation_obs: jnp.ndarray,
        time_remaining: int,
    ) -> tuple[int, "PreciseInfoSeekingAgent"]:
        """
        Execute one agent step: perceive (standard BP), plan (precise info-seeking), act.
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
        action_dist, _, _ = precise_info_seeking_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=self.transition_tensor,
            observation_tensor=_flatten_obs_tensor(self.observation_tensors),
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
            damping=self.damping,
        )

        action = int(jnp.argmax(action_dist))

        new_agent = PreciseInfoSeekingAgent(
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
            damping=self.damping,
        )

        return action, new_agent


@dataclass
class ActiveInferenceAgent:
    """
    Agent using Active Inference planning: VBP action channels + dyn channels + obs channels.
    Combines VBP-style action channel, region-extended dynamics channel, and
    observation channel reparameterization.
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

    damping: float

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
        damping: float = 1.0,
    ) -> "ActiveInferenceAgent":
        """Create a new Active Inference agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        dims["n_static"] = transition_tensor.shape[2]

        n_valid_locations = dims["n_locations"] - 2 * grid_size
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
            damping=damping,
        )

    def reset(self) -> "ActiveInferenceAgent":
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

        return ActiveInferenceAgent(
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
            damping=self.damping,
        )

    def step(
        self,
        vision_obs: jnp.ndarray,
        orientation_obs: jnp.ndarray,
        time_remaining: int,
    ) -> tuple[int, "ActiveInferenceAgent"]:
        """
        Execute one agent step: perceive (standard BP), plan (active inference), act.
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
        action_dist, _, _ = active_inference_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=self.transition_tensor,
            observation_tensor=_flatten_obs_tensor(self.observation_tensors),
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
            damping=self.damping,
        )

        action = int(jnp.argmax(action_dist))

        new_agent = ActiveInferenceAgent(
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
            damping=self.damping,
        )

        return action, new_agent


@dataclass
class NuijtenMPAgent:
    """
    Agent using Nuijten MP planning with θ as a variable node.
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
    ) -> "NuijtenMPAgent":
        """Create a new Nuijten MP agent with uniform initial beliefs."""
        dims = get_dimensions(grid_size)
        dims["n_static"] = transition_tensor.shape[2]

        n_valid_locations = dims["n_locations"] - 2 * grid_size
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

    def reset(self) -> "NuijtenMPAgent":
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

        return NuijtenMPAgent(
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
    ) -> tuple[int, "NuijtenMPAgent"]:
        """
        Execute one agent step: perceive (standard BP), plan (Nuijten MP), act.
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
        action_dist, _, _ = nuijten_mp_planning(
            q_current_state=q_current,
            q_static_state=q_static,
            transition_tensor=self.transition_tensor,
            observation_tensor=_flatten_obs_tensor(self.observation_tensors),
            goal=self.goal,
            horizon=horizon,
            n_iterations=self.n_planning_iterations,
        )

        action = int(jnp.argmax(action_dist))

        new_agent = NuijtenMPAgent(
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
