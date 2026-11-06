"""Teacher-centred residual policy with separate tracking and cost values."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

CONSTRAINT_NAMES = ("boundary", "actuator", "solver", "fallback")
DEFAULT_TEACHER = (1., 1., 1., 0.)


class SRPPOActorCritic(nn.Module):
    """Gaussian residual actor, risk head, scalar reward and vector cost critics.

    PPO probabilities refer to the sampled Gaussian ``raw_z`` before the
    tanh, gate bounds and safety shield. Cost critic columns are positive
    discounted costs in ``CONSTRAINT_NAMES`` order.
    """

    def __init__(self, observation_dim: int = 200, action_dim: int = 4,
                 hidden: int = 128, residual_limit: float = .25,
                 cost_dim: int = 4):
        super().__init__()
        if observation_dim <= 0 or action_dim != 4 or hidden <= 0 or cost_dim != 4:
            raise ValueError("positive dimensions and exactly four actions/constraints required")
        if not math.isfinite(residual_limit) or not 0. < residual_limit <= 1.:
            raise ValueError("residual_limit must be in (0, 1]")
        self.variant = "srppo"
        self.observation_dim, self.action_dim = int(observation_dim), int(action_dim)
        self.hidden, self.cost_dim = int(hidden), int(cost_dim)
        self.residual_limit = float(residual_limit)
        self.backbone = nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(),
                                      nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -1.2))
        self.risk_head = nn.Linear(hidden, 1)
        self.value_track = nn.Linear(hidden, 1)
        self.value_safe = nn.Linear(hidden, cost_dim)
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)

    def _features(self, observations: torch.Tensor,
                  mask: torch.Tensor | None = None) -> torch.Tensor:
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(f"observations must end in {self.observation_dim}")
        if not torch.isfinite(observations).all():
            raise ValueError("observations must be finite")
        if mask is not None:
            if mask.shape != observations.shape or not torch.isfinite(mask).all():
                raise ValueError("finite observation mask must match observations")
            observations = observations * mask
        return self.backbone(observations)

    def distribution(self, observations: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.distributions.Normal:
        hidden = self._features(observations, mask)
        mean = self.actor_mean(hidden)
        std = self.actor_log_std.clamp(-5., 2.).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def values(self, observations: torch.Tensor,
               mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self._features(observations, mask)
        return {"value_track": self.value_track(hidden).squeeze(-1),
                "value_safe": self.value_safe(hidden),
                "risk": self.risk_head(hidden).squeeze(-1).sigmoid()}

    def residual_to_gate(self, raw_z: torch.Tensor,
                         teacher_gate: torch.Tensor | None = None,
                         residual_scale: float | torch.Tensor | None = None) -> torch.Tensor:
        if raw_z.shape[-1] != self.action_dim or not torch.isfinite(raw_z).all():
            raise ValueError("raw_z must be finite [..., 4]")
        teacher = torch.as_tensor(DEFAULT_TEACHER if teacher_gate is None else teacher_gate,
                                  device=raw_z.device, dtype=raw_z.dtype)
        if teacher.shape[-1:] != (4,) or not torch.isfinite(teacher).all():
            raise ValueError("teacher_gate must be finite [..., 4]")
        if ((teacher < 0.) | (teacher > 1.)).any():
            raise ValueError("teacher_gate must lie in [0, 1]")
        try:
            teacher = torch.broadcast_to(teacher, raw_z.shape)
        except RuntimeError as exc:
            raise ValueError("teacher_gate cannot broadcast to raw_z") from exc
        scale = torch.as_tensor(self.residual_limit if residual_scale is None else residual_scale,
                                device=raw_z.device, dtype=raw_z.dtype)
        if not torch.isfinite(scale).all() or ((scale < 0.) | (scale > 1.)).any():
            raise ValueError("residual scale must be finite in [0, 1]")
        if scale.shape == raw_z.shape[:-1]:
            scale = scale.unsqueeze(-1)
        return (teacher + torch.tanh(raw_z) * scale).clamp(0., 1.)

    def act(self, observations: torch.Tensor, teacher_gate: torch.Tensor | None = None,
            deterministic: bool = False, mask: torch.Tensor | None = None,
            residual_scale: float | torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        dist = self.distribution(observations, mask)
        raw_z = dist.mean if deterministic else dist.sample()
        return {"raw_z": raw_z, "log_prob": dist.log_prob(raw_z).sum(-1),
                "entropy": dist.entropy().sum(-1),
                "alpha_raw": self.residual_to_gate(raw_z, teacher_gate, residual_scale),
                **self.values(observations, mask)}

    def evaluate(self, observations: torch.Tensor, raw_z: torch.Tensor,
                 teacher_gate: torch.Tensor | None = None,
                 mask: torch.Tensor | None = None,
                 residual_scale: float | torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if raw_z.shape != (*observations.shape[:-1], self.action_dim) or not torch.isfinite(raw_z).all():
            raise ValueError("raw_z must be finite with the policy action shape")
        dist = self.distribution(observations, mask)
        return {"log_prob": dist.log_prob(raw_z).sum(-1),
                "entropy": dist.entropy().sum(-1),
                "alpha_raw": self.residual_to_gate(raw_z, teacher_gate, residual_scale),
                **self.values(observations, mask)}

    def log_prob(self, observations: torch.Tensor, raw_z: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.evaluate(observations, raw_z, mask=mask)["log_prob"]

    def behavior_cloning_loss(self, observations: torch.Tensor,
                              teacher_gate: torch.Tensor | None,
                              target_gate: torch.Tensor,
                              feasible: torch.Tensor | None = None,
                              mask: torch.Tensor | None = None,
                              residual_scale: float | torch.Tensor | None = None) -> torch.Tensor:
        """Fit candidate targets around a separate fixed deployable anchor.

        ``teacher_gate`` is the anchor (or None for [1,1,1,0]), not the target
        label. Passing each label as its own anchor would give a zero-loss
        identity warm start and teach the actor nothing.
        """
        # Do not clamp the differentiable BC prediction at the teacher's
        # boundary (the default anchor has three channels at 1 and one at 0,
        # where a hard clamp would make the initial Jacobian exactly zero).
        raw_prediction = self.actor_mean(self._features(observations, mask))
        teacher = torch.as_tensor(DEFAULT_TEACHER if teacher_gate is None else teacher_gate,
                                  device=raw_prediction.device, dtype=raw_prediction.dtype)
        teacher = torch.broadcast_to(teacher, raw_prediction.shape)
        scale = torch.as_tensor(self.residual_limit if residual_scale is None else residual_scale,
                                device=raw_prediction.device, dtype=raw_prediction.dtype)
        if scale.shape == raw_prediction.shape[:-1]:
            scale = scale.unsqueeze(-1)
        prediction = teacher + torch.tanh(raw_prediction) * scale
        target = torch.as_tensor(target_gate, device=prediction.device, dtype=prediction.dtype).detach()
        if target.shape != prediction.shape or not torch.isfinite(target).all():
            raise ValueError("finite BC targets must match the gate batch")
        if ((target < 0.) | (target > 1.)).any():
            raise ValueError("BC gate targets must lie in [0, 1]")
        loss = nn.functional.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
        if feasible is not None:
            if feasible.shape != loss.shape:
                raise ValueError("feasibility shape differs from BC batch")
            loss = loss * (.25 + .75 * feasible.to(loss.dtype))
        return loss.mean()

    def checkpoint_config(self) -> dict[str, Any]:
        return {"observation_dim": self.observation_dim, "action_dim": self.action_dim,
                "hidden": self.hidden, "cost_dim": self.cost_dim,
                "residual_limit": self.residual_limit}

    @classmethod
    def from_checkpoint_config(cls, config: dict[str, Any]) -> "SRPPOActorCritic":
        """Reconstruct architecture exactly; unknown/missing fields fail closed."""
        expected = {"observation_dim", "action_dim", "hidden", "cost_dim", "residual_limit"}
        if set(config) != expected:
            raise ValueError("checkpoint policy configuration schema mismatch")
        return cls(**config)
