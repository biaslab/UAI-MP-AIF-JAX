"""RockSample agents using Active Inference planning.

State inference uses position + rock quality observations to update beliefs
over dynamic state and rock quality configuration via Bayesian updates.

Multiple planning methods are supported via separate agent classes.
"""

import jax.numpy as jnp
from jax import nn
from jax.scipy.special import logsumexp
from dataclasses import dataclass, replace

from inference.planning import planning, safe_log
from inference.loopy_bp import loopy_bp_planning
from inference.region_extended_loopy_bp import region_extended_loopy_bp_planning
from inference.reduced_region_extended import reduced_region_extended_planning
from inference.dyn_channel_loopy_bp import dyn_channel_loopy_bp_planning
from inference.reduced_dyn_channel import reduced_dyn_channel_planning
from inference.nuijten_mp import nuijten_mp_planning, reduced_nuijten_mp_planning
from inference.loopy_vbp import loopy_vbp_planning

EPSILON = 1e-12


def _infer_state(
    q_current: jnp.ndarray,
    q_static: jnp.ndarray,
    transition_tensor: jnp.ndarray,
    observation_tensor: jnp.ndarray,
    obs: jnp.ndarray,
    action_onehot: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bayesian state and static-state update for RockSample.

    1. Predict x_new from x_old via transition.
    2. Incorporate observation likelihood to update both x and θ.

    Args:
        q_current: (n_states,) prior belief over state
        q_static: (n_static,) belief over configurations
        transition_tensor: (n_states, n_states, n_static, n_actions)
        observation_tensor: (n_channels, 2, n_states, n_static)
        obs: (n_channels,) binary sensor readings
        action_onehot: (n_actions,) one-hot previous action

    Returns:
        q_new_state: (n_states,) posterior over state
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
    obs_int = jnp.round(obs).astype(jnp.int32)

    log_lik_per_channel = jnp.log(
        observation_tensor[jnp.arange(n_channels), obs_int, :, :] + EPSILON
    )
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
class _RockSampleAgentBase:
    transition_tensor: jnp.ndarray   # (n_states, n_states, n_static, n_actions)
    observation_tensor: jnp.ndarray  # (n_pos + k, 2, n_states, n_static)
    goal: jnp.ndarray               # (n_states, n_static)
    rock_positions: jnp.ndarray     # (k,) rock positions
    qualities: jnp.ndarray          # (n_configs, k)
    n_pos: int
    q_current_state: jnp.ndarray    # (n_states,)
    q_static_state: jnp.ndarray     # (n_static,)
    planning_horizon: int
    planning_iterations: int
    action_prior: jnp.ndarray | None
    last_action: int
    damping: float
    start_state_idx: int            # flat index of start state
    terminal_goal_only: bool        # if True, goal applied only at final step

    @property
    def _rock_obs(self) -> jnp.ndarray:
        """Rock quality observation channels (θ-dependent) for planning."""
        return self.observation_tensor[self.n_pos:]

    def _planning_goal(self, q_static: jnp.ndarray) -> jnp.ndarray:
        """Get goal for planning. If terminal_goal_only, marginalize 2D→1D."""
        if self.terminal_goal_only and self.goal.ndim == 2:
            log_C = safe_log(self.goal)
            log_pref = logsumexp(log_C + safe_log(q_static)[None, :], axis=1)
            return nn.softmax(log_pref)
        return self.goal

    def reset(self):
        n_states = self.transition_tensor.shape[0]
        n_static = self.q_static_state.shape[0]
        q_start = jnp.zeros(n_states, dtype=jnp.float32).at[self.start_state_idx].set(1.0)
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
            obs: (n_channels,) binary sensor readings
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

def _create(cls, transition_tensor, observation_tensor, goal,
            rock_positions, qualities, n_pos, start_state_idx,
            planning_horizon, planning_iterations, action_prior, damping=1.0,
            terminal_goal_only=False):
    """Shared factory for all RockSample agents."""
    n_states = transition_tensor.shape[0]
    n_static = transition_tensor.shape[2]
    return cls(
        transition_tensor=jnp.array(transition_tensor, dtype=jnp.float32),
        observation_tensor=jnp.array(observation_tensor, dtype=jnp.float32),
        goal=jnp.array(goal, dtype=jnp.float32),
        rock_positions=jnp.array(rock_positions, dtype=jnp.int32),
        qualities=jnp.array(qualities, dtype=jnp.float32),
        n_pos=n_pos,
        q_current_state=jnp.ones(n_states, dtype=jnp.float32) / n_states,
        q_static_state=jnp.ones(n_static, dtype=jnp.float32) / n_static,
        planning_horizon=planning_horizon,
        planning_iterations=planning_iterations,
        action_prior=jnp.array(action_prior, dtype=jnp.float32) if action_prior is not None else None,
        last_action=-1,
        damping=damping,
        start_state_idx=start_state_idx,
        terminal_goal_only=terminal_goal_only,
    )


@dataclass(frozen=True)
class RockSampleBPAgent(_RockSampleAgentBase):
    """Standard BP (marginalizes θ once)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               rock_positions, qualities, n_pos, start_state_idx,
               planning_horizon=5, planning_iterations=1, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(RockSampleBPAgent, transition_tensor, observation_tensor,
                       goal, rock_positions, qualities, n_pos, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        return planning(
            q_current, q_static, self.transition_tensor, self._planning_goal(q_static),
            horizon=horizon, action_prior=self.action_prior,
        )


@dataclass(frozen=True)
class RockSampleLoopyBPAgent(_RockSampleAgentBase):
    """Loopy BP with θ as variable node."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               rock_positions, qualities, n_pos, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(RockSampleLoopyBPAgent, transition_tensor, observation_tensor,
                       goal, rock_positions, qualities, n_pos, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        return loopy_bp_planning(
            q_current, q_static, self.transition_tensor, self.goal,
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior,
        )


@dataclass(frozen=True)
class RockSampleLoopyVBPAgent(_RockSampleAgentBase):
    """Loopy VBP (value iteration, θ as variable node)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               rock_positions, qualities, n_pos, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(RockSampleLoopyVBPAgent, transition_tensor, observation_tensor,
                       goal, rock_positions, qualities, n_pos, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        return loopy_vbp_planning(
            q_current, q_static, self.transition_tensor,
            self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
        )


@dataclass(frozen=True)
class RockSampleRegionExtendedAgent(_RockSampleAgentBase):
    """Region-extended loopy BP with observation factors."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               rock_positions, qualities, n_pos, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(RockSampleRegionExtendedAgent, transition_tensor,
                       observation_tensor, goal, rock_positions, qualities,
                       n_pos, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = region_extended_loopy_bp_planning(
            q_current, q_static, self.transition_tensor,
            self._rock_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
        )
        return action_dist


@dataclass(frozen=True)
class RockSampleReducedRegionExtendedAgent(_RockSampleAgentBase):
    """Reduced region-extended (fixed θ)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               rock_positions, qualities, n_pos, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(RockSampleReducedRegionExtendedAgent, transition_tensor,
                       observation_tensor, goal, rock_positions, qualities,
                       n_pos, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = reduced_region_extended_planning(
            q_current, q_static, self.transition_tensor,
            self._rock_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
        )
        return action_dist


@dataclass(frozen=True)
class RockSampleDynChannelAgent(_RockSampleAgentBase):
    """Dyn-channel loopy BP (obs raw B + dyn channels, θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               rock_positions, qualities, n_pos, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(RockSampleDynChannelAgent, transition_tensor,
                       observation_tensor, goal, rock_positions, qualities,
                       n_pos, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _ = dyn_channel_loopy_bp_planning(
            q_current, q_static, self.transition_tensor,
            self._rock_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
        )
        return action_dist


@dataclass(frozen=True)
class RockSampleReducedDynChannelAgent(_RockSampleAgentBase):
    """Reduced dyn-channel (fixed θ)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               rock_positions, qualities, n_pos, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(RockSampleReducedDynChannelAgent, transition_tensor,
                       observation_tensor, goal, rock_positions, qualities,
                       n_pos, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _ = reduced_dyn_channel_planning(
            q_current, q_static, self.transition_tensor,
            self._rock_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
        )
        return action_dist


@dataclass(frozen=True)
class RockSampleNuijtenMPAgent(_RockSampleAgentBase):
    """Nuijten MP (region beliefs + EFE, θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               rock_positions, qualities, n_pos, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(RockSampleNuijtenMPAgent, transition_tensor,
                       observation_tensor, goal, rock_positions, qualities,
                       n_pos, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = nuijten_mp_planning(
            q_current, q_static, self.transition_tensor,
            self._rock_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior,
        )
        return action_dist


@dataclass(frozen=True)
class RockSampleReducedNuijtenMPAgent(_RockSampleAgentBase):
    """Reduced Nuijten MP (fixed θ)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               rock_positions, qualities, n_pos, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(RockSampleReducedNuijtenMPAgent, transition_tensor,
                       observation_tensor, goal, rock_positions, qualities,
                       n_pos, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = reduced_nuijten_mp_planning(
            q_current, q_static, self.transition_tensor,
            self._rock_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior,
        )
        return action_dist


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_CLASSES = {
    "bp": RockSampleBPAgent,
    "loopy_bp": RockSampleLoopyBPAgent,
    "loopy_vbp": RockSampleLoopyVBPAgent,
    "region_extended": RockSampleRegionExtendedAgent,
    "reduced_region_extended": RockSampleReducedRegionExtendedAgent,
    "dyn_channel": RockSampleDynChannelAgent,
    "reduced_dyn_channel": RockSampleReducedDynChannelAgent,
    "nuijten": RockSampleNuijtenMPAgent,
    "reduced_nuijten": RockSampleReducedNuijtenMPAgent,
}


def create_agent(
    method: str,
    transition_tensor,
    observation_tensor,
    goal,
    rock_positions,
    qualities,
    n_pos: int,
    start_state_idx: int,
    planning_horizon: int = 5,
    planning_iterations: int = 3,
    action_prior=None,
    damping: float = 1.0,
    terminal_goal_only: bool = False,
):
    """Create a RockSample agent for the given planning method."""
    if method not in AGENT_CLASSES:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(AGENT_CLASSES.keys())}")

    cls = AGENT_CLASSES[method]
    return cls.create(
        transition_tensor, observation_tensor, goal,
        rock_positions, qualities, n_pos, start_state_idx,
        planning_horizon, planning_iterations, action_prior,
        damping=damping, terminal_goal_only=terminal_goal_only,
    )
