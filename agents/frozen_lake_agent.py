"""Frozen Lake agents using Active Inference planning.

State inference uses noisy binary sensor observations to update beliefs over
position and hole configuration via Bayesian updates.

Multiple planning methods are supported via separate agent classes.
"""

import jax.numpy as jnp
from jax import nn
from dataclasses import dataclass, replace

from inference.loopy_bp import loopy_bp_planning
from inference.region_extended_loopy_bp import region_extended_loopy_bp_planning
from inference.dyn_channel_loopy_bp import dyn_channel_loopy_bp_planning
from inference.nuijten_mp import nuijten_mp_planning
from inference.loopy_vbp import loopy_vbp_planning
from inference.vbp_channel import vbp_channel_planning
from inference.precise_info_seeking import precise_info_seeking_planning

EPSILON = 1e-12


def _infer_state(
    q_current: jnp.ndarray,
    q_static: jnp.ndarray,
    transition_tensor: jnp.ndarray,
    observation_tensor: jnp.ndarray,
    obs: jnp.ndarray,
    action_onehot: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bayesian state and static-state update.

    1. Predict x_new from x_old via transition.
    2. Incorporate observation likelihood to update both x and θ.

    Args:
        q_current: (n_states,) prior belief over position
        q_static: (n_static,) belief over configurations
        transition_tensor: (n_states, n_states, n_static, n_actions)
        observation_tensor: (n_channels, 2, n_states, n_static)
        obs: (n_channels,) binary sensor readings
        action_onehot: (n_actions,) one-hot previous action

    Returns:
        q_new_state: (n_states,) posterior over position
        q_new_static: (n_static,) posterior over configuration
    """
    n_states = q_current.shape[0]
    n_static = q_static.shape[0]
    n_channels = observation_tensor.shape[0]

    # Predict: P(x_new) = Σ_{x_old, θ, a} T(x_new|x_old,θ,a) · q(x_old) · q(θ) · a(a)
    predicted = jnp.einsum(
        "ijkl,j,k,l->i",
        transition_tensor, q_current, q_static, action_onehot,
    )
    predicted = predicted / (predicted.sum() + EPSILON)

    # Observation log-likelihood: Σ_c log P(obs_c | x, θ)
    # obs[c] ∈ {0, 1}, B[c, obs_c, x, θ] = P(obs_c | x, θ)
    obs_int = jnp.round(obs).astype(jnp.int32)  # (n_channels,)

    # For each channel c: P(obs_c | x, θ) = B[c, obs_int[c], x, θ]
    # Shape: (n_channels, n_states, n_static)
    log_lik_per_channel = jnp.log(
        observation_tensor[jnp.arange(n_channels), obs_int, :, :] + EPSILON
    )
    # Sum over channels: (n_states, n_static)
    log_likelihood = log_lik_per_channel.sum(axis=0)

    # Joint posterior: P(x, θ | obs) ∝ P(obs | x, θ) · P(x) · P(θ)
    log_joint = (
        log_likelihood
        + jnp.log(predicted + EPSILON)[:, None]
        + jnp.log(q_static + EPSILON)[None, :]
    )

    # Marginals
    log_q_x = jnp.logaddexp.reduce(log_joint, axis=1)
    q_new_state = nn.softmax(log_q_x)

    log_q_theta = jnp.logaddexp.reduce(log_joint, axis=0)
    q_new_static = nn.softmax(log_q_theta)

    return q_new_state, q_new_static


# ---------------------------------------------------------------------------
# Base agent with shared logic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FrozenLakeAgentBase:
    transition_tensor: jnp.ndarray   # (n_states, n_states, n_static, n_actions)
    observation_tensor: jnp.ndarray  # (n_states + n_pos, 2, n_states, n_static) position + grid cell sensors
    goal: jnp.ndarray               # (n_states,) or (n_states, n_static)
    holes: jnp.ndarray              # (n_static, n_states)
    q_current_state: jnp.ndarray    # (n_states,)
    q_static_state: jnp.ndarray     # (n_static,)
    planning_horizon: int
    planning_iterations: int
    action_prior: jnp.ndarray | None
    last_action: int
    damping: float
    momentum: float

    @property
    def _directional_obs(self) -> jnp.ndarray:
        """Directional-only observation tensor (last 4 channels) for planning.

        Planners with observation factors only need the θ-dependent
        directional channels, not the θ-independent position channels.
        """
        n_states = self.transition_tensor.shape[0]
        return self.observation_tensor[n_states:]

    def reset(self):
        n_states = self.transition_tensor.shape[0]
        n_static = self.holes.shape[0]
        # Agent always starts at position 0 (known)
        q_start = jnp.zeros(n_states, dtype=jnp.float32).at[0].set(1.0)
        return replace(self,
            q_current_state=q_start,
            q_static_state=jnp.ones(n_static) / n_static,
            last_action=-1,
        )

    def _plan(self, q_current, q_static, horizon):
        raise NotImplementedError

    def step(self, obs: jnp.ndarray, time_remaining: int):
        """Take a step given binary sensor observations.

        Args:
            obs: (n_states,) binary sensor readings
            time_remaining: steps left (used for receding horizon)

        Returns:
            action: int
            updated_agent: agent with updated beliefs
        """
        n_actions = self.transition_tensor.shape[3]

        # State inference
        if self.last_action >= 0:
            action_onehot = jnp.zeros(n_actions).at[self.last_action].set(1.0)
        else:
            action_onehot = jnp.ones(n_actions) / n_actions

        q_current, q_static = _infer_state(
            self.q_current_state, self.q_static_state,
            self.transition_tensor, self.observation_tensor,
            obs, action_onehot,
        )

        # Planning
        horizon = min(time_remaining, self.planning_horizon)
        action_dist = self._plan(q_current, q_static, horizon)
        action = int(jnp.argmax(action_dist))

        return action, replace(self,
            q_current_state=q_current,
            q_static_state=q_static,
            last_action=action,
        )


# ---------------------------------------------------------------------------
# Method-specific agents
# ---------------------------------------------------------------------------

def _create(cls, transition_tensor, observation_tensor, goal, holes,
            planning_horizon, planning_iterations, action_prior, damping=1.0,
            momentum=0.0):
    """Shared factory for all Frozen Lake agents."""
    n_states = transition_tensor.shape[0]
    n_static = holes.shape[0]
    return cls(
        transition_tensor=jnp.array(transition_tensor, dtype=jnp.float32),
        observation_tensor=jnp.array(observation_tensor, dtype=jnp.float32),
        goal=jnp.array(goal, dtype=jnp.float32),
        holes=jnp.array(holes, dtype=jnp.float32),
        q_current_state=jnp.ones(n_states, dtype=jnp.float32) / n_states,
        q_static_state=jnp.ones(n_static, dtype=jnp.float32) / n_static,
        planning_horizon=planning_horizon,
        planning_iterations=planning_iterations,
        action_prior=jnp.array(action_prior, dtype=jnp.float32) if action_prior is not None else None,
        last_action=-1,
        damping=damping,
        momentum=momentum,
    )


@dataclass(frozen=True)
class FrozenLakeLoopyBPAgent(_FrozenLakeAgentBase):
    """Loopy BP with θ as variable node."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal, holes,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(FrozenLakeLoopyBPAgent, transition_tensor, observation_tensor,
                       goal, holes, planning_horizon, planning_iterations, action_prior,
                       damping=damping, momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        return loopy_bp_planning(
            q_current, q_static, self.transition_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior,
        )


@dataclass(frozen=True)
class FrozenLakeLoopyVBPAgent(_FrozenLakeAgentBase):
    """Loopy VBP (value iteration, θ as variable node)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal, holes,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(FrozenLakeLoopyVBPAgent, transition_tensor, observation_tensor,
                       goal, holes, planning_horizon, planning_iterations, action_prior,
                       damping=damping, momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        return loopy_vbp_planning(
            q_current, q_static, self.transition_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
        )


@dataclass(frozen=True)
class FrozenLakeRegionExtendedAgent(_FrozenLakeAgentBase):
    """Region-extended loopy BP with observation factors."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal, holes,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(FrozenLakeRegionExtendedAgent, transition_tensor,
                       observation_tensor, goal, holes, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = region_extended_loopy_bp_planning(
            q_current, q_static, self.transition_tensor,
            self._directional_obs, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
            momentum=self.momentum,
        )
        return action_dist


@dataclass(frozen=True)
class FrozenLakeDynChannelAgent(_FrozenLakeAgentBase):
    """Dyn-channel loopy BP (obs raw B + dyn channels, θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal, holes,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(FrozenLakeDynChannelAgent, transition_tensor,
                       observation_tensor, goal, holes, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _ = dyn_channel_loopy_bp_planning(
            q_current, q_static, self.transition_tensor,
            self._directional_obs, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
            momentum=self.momentum,
        )
        return action_dist


@dataclass(frozen=True)
class FrozenLakeNuijtenMPAgent(_FrozenLakeAgentBase):
    """Nuijten MP (region beliefs + EFE, θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal, holes,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(FrozenLakeNuijtenMPAgent, transition_tensor,
                       observation_tensor, goal, holes, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = nuijten_mp_planning(
            q_current, q_static, self.transition_tensor,
            self._directional_obs, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior,
        )
        return action_dist


@dataclass(frozen=True)
class FrozenLakeVBPChannelAgent(_FrozenLakeAgentBase):
    """VBP channel (action channel reparameterization, θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal, holes,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(FrozenLakeVBPChannelAgent, transition_tensor,
                       observation_tensor, goal, holes, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _ = vbp_channel_planning(
            q_current, q_static, self.transition_tensor,
            self._directional_obs, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
            momentum=self.momentum,
        )
        return action_dist


@dataclass(frozen=True)
class FrozenLakePreciseInfoSeekingAgent(_FrozenLakeAgentBase):
    """Precise info-seeking (VBP action channels + obs channels)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal, holes,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(FrozenLakePreciseInfoSeekingAgent, transition_tensor,
                       observation_tensor, goal, holes, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = precise_info_seeking_planning(
            q_current, q_static, self.transition_tensor,
            self._directional_obs, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
            momentum=self.momentum,
        )
        return action_dist


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_CLASSES = {
    "loopy_bp": FrozenLakeLoopyBPAgent,
    "loopy_vbp": FrozenLakeLoopyVBPAgent,
    "region_extended": FrozenLakeRegionExtendedAgent,
    "dyn_channel": FrozenLakeDynChannelAgent,
    "nuijten": FrozenLakeNuijtenMPAgent,
    "vbp_channel": FrozenLakeVBPChannelAgent,
    "precise_info_seeking": FrozenLakePreciseInfoSeekingAgent,
}


def create_agent(
    method: str,
    transition_tensor,
    observation_tensor,
    goal,
    holes,
    planning_horizon: int = 5,
    planning_iterations: int = 3,
    action_prior=None,
    damping: float = 1.0,
    momentum: float = 0.0,
):
    """Create a Frozen Lake agent for the given planning method."""
    if method not in AGENT_CLASSES:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(AGENT_CLASSES.keys())}")

    cls = AGENT_CLASSES[method]
    return cls.create(
        transition_tensor, observation_tensor, goal, holes,
        planning_horizon, planning_iterations, action_prior,
        damping=damping, momentum=momentum,
    )
