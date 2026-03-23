"""Wumpus World agents using Active Inference planning.

State inference uses observations (breeze, stench, glitter) to update beliefs
over position and hole/wumpus configuration. Since observations are indirect
(agent doesn't see its exact position), full Bayesian inference is needed.

For simplicity, we use a lightweight inference approach:
- q_current_state is maintained via transition + observation updates
- q_static_state is updated by incorporating observation likelihood

Multiple planning methods are supported via separate agent classes.
"""

import jax.numpy as jnp
from dataclasses import dataclass, replace
from jax import nn

from inference.messages import safe_log
from inference.loopy_bp import loopy_bp_planning
from inference.region_extended_loopy_bp import region_extended_loopy_bp_planning
from inference.dyn_channel_loopy_bp import dyn_channel_loopy_bp_planning
from inference.nuijten_mp import nuijten_mp_planning
from inference.loopy_vbp import loopy_vbp_planning
from inference.vbp_channel import vbp_channel_planning
from inference.precise_info_seeking import precise_info_seeking_planning
from inference.active_inference import active_inference_planning

EPSILON = 1e-12


def _infer_state(
    q_current: jnp.ndarray,
    q_static: jnp.ndarray,
    transition_tensor: jnp.ndarray,
    observation_tensor: jnp.ndarray,
    obs_onehot: jnp.ndarray,
    action_onehot: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bayesian state and static-state update for Wumpus World.

    1. Predict x_new from x_old via transition (marginalize θ with current belief).
    2. Incorporate observation likelihood to update both x and θ.

    Args:
        q_current: (n_states,) prior belief over position (before action)
        q_static: (n_static,) belief over configurations
        transition_tensor: (n_states, n_states, n_static, n_actions)
        observation_tensor: (n_channels, n_obs_types, n_states, n_static)
        obs_onehot: (n_channels, n_obs_types) one-hot observations per channel
        action_onehot: (n_actions,) one-hot previous action

    Returns:
        q_new_state: (n_states,) posterior over current position
        q_new_static: (n_static,) posterior over configuration
    """
    n_states = q_current.shape[0]
    n_static = q_static.shape[0]
    n_channels = observation_tensor.shape[0]

    # Predict: P(x_new) = Σ_{x_old, θ, a} T(x_new|x_old,θ,a) · q(x_old) · q(θ) · a_onehot(a)
    # T shape: (x_new, x_old, θ, a)
    predicted = jnp.einsum(
        "ijkl,j,k,l->i",
        transition_tensor, q_current, q_static, action_onehot,
    )
    predicted = predicted / (predicted.sum() + EPSILON)

    # Observation likelihood: for each channel, L(x, θ) = Σ_o B(c, o, x, θ) · obs(c, o)
    # obs_onehot shape: (n_channels, n_obs_types)
    # B shape: (n_channels, n_obs_types, n_states, n_static)
    log_likelihood = jnp.zeros((n_states, n_static))
    for c in range(n_channels):
        # P(obs_c | x, θ) = Σ_o B(c, o, x, θ) · obs_onehot(c, o)
        channel_lik = jnp.einsum("ox,o->x", observation_tensor[c, :, :, :].reshape(observation_tensor.shape[1], -1), obs_onehot[c])
        channel_lik = channel_lik.reshape(n_states, n_static)
        log_likelihood = log_likelihood + jnp.log(channel_lik + EPSILON)

    # Joint posterior: P(x, θ | obs) ∝ P(obs | x, θ) · P(x) · P(θ)
    log_joint = log_likelihood + jnp.log(predicted + EPSILON)[:, None] + jnp.log(q_static + EPSILON)[None, :]

    # Marginals
    log_q_x = jnp.logaddexp.reduce(log_joint, axis=1)
    q_new_state = nn.softmax(log_q_x)

    log_q_theta = jnp.logaddexp.reduce(log_joint, axis=0)
    q_new_static = nn.softmax(log_q_theta)

    return q_new_state, q_new_static


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _WumpusAgentBase:
    transition_tensor: jnp.ndarray   # (n_states, n_states, n_static, n_actions)
    observation_tensor: jnp.ndarray  # (3, 2, n_states, n_static)
    goal: jnp.ndarray               # (n_states,) or (n_states, n_static)
    q_current_state: jnp.ndarray    # (n_states,)
    q_static_state: jnp.ndarray     # (n_static,)
    planning_horizon: int
    planning_iterations: int
    action_prior: jnp.ndarray | None
    last_action: int
    damping: float
    momentum: float

    def reset(self):
        n_states = self.goal.shape[0]
        n_static = self.q_static_state.shape[0]
        return replace(self,
            q_current_state=jnp.ones(n_states) / n_states,
            q_static_state=jnp.ones(n_static) / n_static,
            last_action=-1,
        )

    def _plan(self, q_current, q_static, horizon):
        raise NotImplementedError

    def step(self, obs: jnp.ndarray, time_remaining: int):
        """Take a step given binary observations.

        Args:
            obs: (3,) binary observations [breeze, stench, glitter]
            time_remaining: steps left

        Returns:
            action: int
            updated_agent: agent with updated beliefs
        """
        n_actions = self.transition_tensor.shape[3]

        # Convert obs to one-hot: (n_channels, 2)
        n_channels = self.observation_tensor.shape[0]
        obs_onehot = jnp.zeros((n_channels, 2))
        for c in range(n_channels):
            obs_onehot = obs_onehot.at[c, jnp.round(obs[c]).astype(int)].set(1.0)

        # State inference
        if self.last_action >= 0:
            action_onehot = jnp.zeros(n_actions).at[self.last_action].set(1.0)
            q_current, q_static = _infer_state(
                self.q_current_state, self.q_static_state,
                self.transition_tensor, self.observation_tensor,
                obs_onehot, action_onehot,
            )
        else:
            # First step: just use observation likelihood with uniform prior
            action_onehot = jnp.ones(n_actions) / n_actions
            q_current, q_static = _infer_state(
                self.q_current_state, self.q_static_state,
                self.transition_tensor, self.observation_tensor,
                obs_onehot, action_onehot,
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

def _create(cls, transition_tensor, observation_tensor, goal,
            planning_horizon, planning_iterations, action_prior, damping=1.0,
            momentum=0.0):
    n_states = goal.shape[0]
    n_static = observation_tensor.shape[3]
    return cls(
        transition_tensor=jnp.array(transition_tensor, dtype=jnp.float32),
        observation_tensor=jnp.array(observation_tensor, dtype=jnp.float32),
        goal=jnp.array(goal, dtype=jnp.float32),
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
class WumpusLoopyBPAgent(_WumpusAgentBase):
    """Loopy BP with θ as variable node."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(WumpusLoopyBPAgent, transition_tensor, observation_tensor,
                       goal, planning_horizon, planning_iterations, action_prior,
                       damping=damping, momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        return loopy_bp_planning(
            q_current, q_static, self.transition_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior,
        )


@dataclass(frozen=True)
class WumpusLoopyVBPAgent(_WumpusAgentBase):
    """Loopy VBP (value iteration, θ as variable node)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(WumpusLoopyVBPAgent, transition_tensor, observation_tensor,
                       goal, planning_horizon, planning_iterations, action_prior,
                       damping=damping, momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        return loopy_vbp_planning(
            q_current, q_static, self.transition_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
        )


@dataclass(frozen=True)
class WumpusRegionExtendedAgent(_WumpusAgentBase):
    """Region-extended loopy BP with observation factors."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(WumpusRegionExtendedAgent, transition_tensor,
                       observation_tensor, goal, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = region_extended_loopy_bp_planning(
            q_current, q_static, self.transition_tensor,
            self.observation_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
            momentum=self.momentum,
        )
        return action_dist


@dataclass(frozen=True)
class WumpusDynChannelAgent(_WumpusAgentBase):
    """Dyn-channel loopy BP."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(WumpusDynChannelAgent, transition_tensor,
                       observation_tensor, goal, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _ = dyn_channel_loopy_bp_planning(
            q_current, q_static, self.transition_tensor,
            self.observation_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
            momentum=self.momentum,
        )
        return action_dist


@dataclass(frozen=True)
class WumpusNuijtenMPAgent(_WumpusAgentBase):
    """Nuijten MP (θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(WumpusNuijtenMPAgent, transition_tensor,
                       observation_tensor, goal, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = nuijten_mp_planning(
            q_current, q_static, self.transition_tensor,
            self.observation_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior,
        )
        return action_dist


@dataclass(frozen=True)
class WumpusVBPChannelAgent(_WumpusAgentBase):
    """VBP channel (action channel reparameterization, θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(WumpusVBPChannelAgent, transition_tensor,
                       observation_tensor, goal, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _ = vbp_channel_planning(
            q_current, q_static, self.transition_tensor,
            self.observation_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
            momentum=self.momentum,
        )
        return action_dist


@dataclass(frozen=True)
class WumpusPreciseInfoSeekingAgent(_WumpusAgentBase):
    """Precise info-seeking (VBP action channels + obs channels)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(WumpusPreciseInfoSeekingAgent, transition_tensor,
                       observation_tensor, goal, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = precise_info_seeking_planning(
            q_current, q_static, self.transition_tensor,
            self.observation_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
            momentum=self.momentum,
        )
        return action_dist


@dataclass(frozen=True)
class WumpusActiveInferenceAgent(_WumpusAgentBase):
    """Active Inference (VBP action channels + dyn channels + obs channels)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, momentum=0.0):
        return _create(WumpusActiveInferenceAgent, transition_tensor,
                       observation_tensor, goal, planning_horizon,
                       planning_iterations, action_prior, damping=damping,
                       momentum=momentum)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = active_inference_planning(
            q_current, q_static, self.transition_tensor,
            self.observation_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
            momentum=self.momentum,
        )
        return action_dist


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_CLASSES = {
    "loopy_bp": WumpusLoopyBPAgent,
    "loopy_vbp": WumpusLoopyVBPAgent,
    "region_extended": WumpusRegionExtendedAgent,
    "dyn_channel": WumpusDynChannelAgent,
    "nuijten": WumpusNuijtenMPAgent,
    "vbp_channel": WumpusVBPChannelAgent,
    "precise_info_seeking": WumpusPreciseInfoSeekingAgent,
    "active_inference": WumpusActiveInferenceAgent,
}


def create_agent(
    method: str,
    transition_tensor,
    observation_tensor,
    goal,
    planning_horizon: int = 5,
    planning_iterations: int = 3,
    action_prior=None,
    damping: float = 1.0,
    momentum: float = 0.0,
):
    """Create a Wumpus World agent for the given planning method."""
    if method not in AGENT_CLASSES:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(AGENT_CLASSES.keys())}")

    cls = AGENT_CLASSES[method]
    return cls.create(
        transition_tensor, observation_tensor, goal,
        planning_horizon, planning_iterations, action_prior,
        damping=damping, momentum=momentum,
    )
