"""Deterministic gate safety projection (the shield never calls MPC)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


EMERGENCY_GATE = np.zeros(4, dtype=np.float64)


@dataclass(frozen=True)
class SafetyConfig:
    """Limits used by :func:`project_gate`.

    ``slew_limit`` is applied per 500 ms policy decision. The episode adapter
    does not call the normal projection on intervening control periods. The optional margin and
    risk thresholds shrink residual actions towards the teacher; they do not
    alter the number of MPC solves.
    """

    slew_limit: float = 0.25
    residual_limit: float = 0.25
    min_margin: float = 0.08
    risk_threshold: float = 0.35
    risk_shrink: float = 0.8
    emergency_gate: tuple[float, float, float, float] = (0., 0., 0., 0.)

    def __post_init__(self) -> None:
        values = (self.slew_limit, self.residual_limit, self.min_margin,
                  self.risk_threshold, self.risk_shrink)
        if not np.isfinite(values).all() or self.slew_limit < 0. or self.slew_limit > 1.:
            raise ValueError("invalid safety limits")
        if self.residual_limit < 0. or self.min_margin < 0. or self.risk_threshold < 0.:
            raise ValueError("invalid safety thresholds")
        if self.risk_shrink < 0. or self.risk_shrink > 1.:
            raise ValueError("risk_shrink must be in [0, 1]")
        _vec(self.emergency_gate, "emergency_gate")


@dataclass(frozen=True)
class ProjectionResult:
    alpha: np.ndarray
    raw_alpha: np.ndarray
    teacher_alpha: np.ndarray
    overridden: bool
    reason: str
    clip_fraction: float
    risk: float
    emergency: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _vec(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite with shape [4]")
    return arr


def _context(context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return {} if context is None else context


def project_gate(previous: Any, target: Any, context: Mapping[str, Any] | None = None,
                config: SafetyConfig | None = None) -> ProjectionResult:
    """Project a policy gate onto safe bounds and a teacher-centred slew set.

    Non-finite policy output, observations, or a failed QP use the emergency
    gate and are explicitly recorded.  A QP status is supplied by the episode
    adapter in ``context``; no solver is invoked here.
    """
    cfg = config or SafetyConfig()
    ctx = _context(context)
    try:
        previous_arr = _vec(previous, "previous")
        target_arr = _vec(target, "target")
        teacher = _vec(ctx.get("teacher_gate", (1., 1., 1., 0.)), "teacher_gate")
    except (TypeError, ValueError):
        emergency = _vec(cfg.emergency_gate, "emergency_gate")
        return ProjectionResult(emergency.copy(), np.full(4, np.nan), emergency,
                                True, "nonfinite_input", 1., 1., True)

    def emergency_result(reason: str, raw: np.ndarray = target_arr) -> ProjectionResult:
        emergency = np.clip(_vec(cfg.emergency_gate, "emergency_gate"), 0., 1.)
        return ProjectionResult(emergency.copy(), raw.copy(), teacher.copy(), True,
                                reason, 1., 1., True)

    # A failed/non-finite upstream QP must be visible to both the evaluator
    # and the learner.  ``qp_success`` defaults to true for pure policy tests.
    try:
        qp_status = float(ctx.get("qp_success", True))
        if not np.isfinite(qp_status) or qp_status not in (0., 1.):
            return emergency_result("nonfinite_qp_status")
        qp_success = bool(qp_status)
        streak = int(ctx.get("saturation_streak", 0))
        if not np.isfinite(float(ctx.get("saturation_streak", 0))):
            return emergency_result("nonfinite_saturation_streak")
    except (TypeError, ValueError, OverflowError):
        return emergency_result("nonfinite_qp_status")
    if not qp_success:
        return emergency_result("qp_failure")
    for key in ("observation", "qp_input"):
        if key in ctx:
            value = np.asarray(ctx[key])
            if not np.isfinite(value).all():
                return emergency_result(f"nonfinite_{key}")

    raw = np.clip(target_arr, 0., 1.)
    projected = np.clip(raw, 0., 1.)
    reasons: list[str] = []
    if not np.allclose(raw, target_arr):
        reasons.append("bounds")

    # Slew is always enforced, including teacher changes at an episode reset.
    step = np.clip(projected - previous_arr, -abs(cfg.slew_limit),
                   abs(cfg.slew_limit))
    projected = np.clip(previous_arr + step, 0., 1.)
    if not np.allclose(projected, raw):
        reasons.append("slew")

    margin = float(ctx.get("min_margin", 1.0))
    risk = float(ctx.get("risk", 0.0))
    if not np.isfinite(margin) or not np.isfinite(risk):
        return emergency_result("nonfinite_risk")
    if margin < cfg.min_margin:
        projected = teacher + (projected - teacher) * max(0., margin / max(cfg.min_margin, 1e-9))
        reasons.append("margin")
    if bool(ctx.get("saturation", False)) or streak > 0:
        projected = teacher + (projected - teacher) * 0.5
        reasons.append("saturation")
    if risk > cfg.risk_threshold:
        shrink = np.clip(cfg.risk_shrink * cfg.risk_threshold / max(risk, 1e-9), 0., 1.)
        projected = teacher + (projected - teacher) * shrink
        reasons.append("risk")
    boundary_ratio = float(ctx.get("boundary_ratio", 0.))
    if not np.isfinite(boundary_ratio):
        return emergency_result("nonfinite_boundary")
    if boundary_ratio > .7:
        # Remove learned deviation as a physical stop boundary approaches.
        # This is a conservative heuristic, not a proof of invariance.
        shrink = np.clip((1. - boundary_ratio) / .3, 0., 1.)
        projected = teacher + (projected - teacher) * shrink
        reasons.append("boundary")
    if bool(ctx.get("estimator_fallback", False)):
        projected[0] = 0.
        reasons.append("estimator_fallback")
    if bool(ctx.get("residual_clip", False)):
        projected[1:] *= .5
        reasons.append("residual_clip")
    projected = np.clip(projected, 0., 1.)
    # Re-apply slew after a shrink towards the teacher so every returned action
    # obeys the contract regardless of context values.
    projected = np.clip(previous_arr + np.clip(projected - previous_arr,
                                               -abs(cfg.slew_limit), abs(cfg.slew_limit)), 0., 1.)
    reason = "+".join(dict.fromkeys(reasons)) if reasons else "none"
    return ProjectionResult(projected, raw, teacher, bool(reasons), reason,
                            float(np.mean(np.abs(projected - target_arr) > 1e-12)),
                            float(np.clip(risk, 0., 1.)), False,
                            {"margin": margin, "qp_success": qp_success})


def shield_action(previous: Any, target: Any, context: Mapping[str, Any] | None = None,
                  config: SafetyConfig | None = None) -> ProjectionResult:
    """Compatibility alias used by episode adapters."""
    return project_gate(previous, target, context, config)
