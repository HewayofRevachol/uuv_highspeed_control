"""Vector-valued tracking and safety costs for SR-PPO."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class StepCosts:
    """Costs for one control period, retained before scalarisation."""

    track: float = 0.0
    boundary: float = 0.0
    saturation: float = 0.0
    margin: float = 0.0
    qp: float = 0.0
    fallback: float = 0.0
    deadline: float = 0.0
    slew: float = 0.0
    failed: float = 0.0
    terminal: float = 0.0

    def vector(self) -> np.ndarray:
        return np.asarray([self.track, self.boundary, self.saturation, self.margin,
                           self.qp, self.fallback, self.deadline, self.slew,
                           self.failed, self.terminal], dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _hinge(value: np.ndarray, limit: np.ndarray, onset: float = .7) -> float:
    ratio = np.abs(value) / np.maximum(np.abs(limit), 1e-9)
    # Start before the stop envelope. At the boundary a component costs 1;
    # unlike an outside-only hinge, the learner gets a signal while recovery
    # is still possible.
    return float(np.mean((np.maximum(ratio - onset, 0.) / (1. - onset)) ** 2))


def compute_cost(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None,
                alpha: Any, previous_alpha: Any, command: Any,
                info: Mapping[str, Any] | None = None,
                limits: Mapping[str, Any] | None = None) -> StepCosts:
    """Compute finite non-negative component costs.

    ``before`` and ``after`` are adapter measurements.  Error and actuator
    values may be supplied directly in ``info`` for lightweight testing.  A
    physical failure is terminal and receives its remaining-horizon cost;
    time-limit truncation is recorded separately and does not receive it.
    """
    info = {} if info is None else info
    limits = {} if limits is None else limits
    alpha = np.asarray(alpha, dtype=float); previous_alpha = np.asarray(previous_alpha, dtype=float)
    command = np.asarray(command, dtype=float)
    if alpha.shape != (4,) or previous_alpha.shape != (4,) or command.ndim != 1:
        raise ValueError("alpha/previous_alpha must be [4] and command must be a vector")
    if not np.isfinite(alpha).all() or not np.isfinite(previous_alpha).all() or not np.isfinite(command).all():
        raise ValueError("cost inputs must be finite")

    errors = np.asarray(info.get("errors", (0., 0., 0., 0.)), dtype=float)
    error_limits = np.asarray(limits.get("error", (8., 5., 2., 8.)), dtype=float)
    if errors.shape != error_limits.shape or not np.isfinite(errors).all():
        raise ValueError("errors and error limits must be finite with matching shape")
    track = float(info.get("tracking_cost", np.mean((errors / np.maximum(np.abs(error_limits), 1e-9)) ** 2)))
    boundary_errors = np.asarray(info.get("boundary_errors", errors), dtype=float)
    boundary_limits = np.asarray(limits.get("boundary", error_limits), dtype=float)
    if boundary_errors.shape != boundary_limits.shape or not np.isfinite(boundary_errors).all():
        raise ValueError("boundary errors/limits must be finite with matching shape")
    boundary = _hinge(boundary_errors, boundary_limits)

    actual = np.asarray(info.get("actual", command), dtype=float)
    actuator_limits = np.asarray(limits.get("actuator", np.ones_like(actual)), dtype=float)
    if actual.shape != actuator_limits.shape or not np.isfinite(actual).all():
        raise ValueError("actual and actuator limits must be finite with matching shape")
    saturation = float(np.mean(np.maximum(np.abs(actual) / np.maximum(np.abs(actuator_limits), 1e-9) - .98, 0.) ** 2))
    margin_value = info.get("margin", None)
    if margin_value is None:
        margin = float(np.mean(np.maximum(0., .1 - (1. - np.abs(actual) / np.maximum(np.abs(actuator_limits), 1e-9)))))
    else:
        margin_array = np.asarray(margin_value, dtype=float)
        if not np.isfinite(margin_array).all():
            raise ValueError("margin must be finite")
        margin = float(np.mean(np.maximum(0., .1 - margin_array)))
    margin = max(0., _finite_scalar(margin, "margin"))

    slew = float(np.mean(np.abs(alpha - previous_alpha) ** 2))
    max_slew = float(limits.get("slew", .25))
    slew += float(np.mean(np.maximum(np.abs(alpha - previous_alpha) - max_slew, 0.) ** 2))
    qp = float(bool(info.get("qp_failure", False)))
    fallback = float(bool(info.get("fallback", False))) + float(bool(info.get("residual_clip", False)))
    deadline = float(bool(info.get("deadline", False)))
    failed = bool(info.get("failed", False))
    terminated = bool(info.get("terminated", failed))
    truncated = bool(info.get("truncated", False))
    remaining = max(0., float(info.get("remaining_s", 0.)))
    terminal = (20. + 100. * remaining) if failed and terminated and not truncated else 0.0
    # ``failed`` is an event marker; terminal carries the absorbing penalty so
    # the physical failure is charged exactly once by scalarization.
    failed_cost = float(failed)
    values = [track, boundary, saturation, margin, qp, fallback, deadline, slew,
              failed_cost, terminal]
    if not np.isfinite(values).all() or np.any(np.asarray(values) < 0.):
        raise ValueError("computed cost is not finite and non-negative")
    return StepCosts(*map(float, values))


def safety_vector(cost: StepCosts, dt: float) -> np.ndarray:
    """Integrated positive costs in boundary/actuator/solver/fallback order.

    Failure absorption belongs to the reward once, outside this vector. The
    failure marker is for auditing and is never an additional objective term.
    """
    if not np.isfinite(dt) or dt <= 0.:
        raise ValueError("dt must be positive and finite")
    return dt * np.array([cost.boundary, cost.saturation + cost.margin,
                          cost.qp + cost.deadline, cost.fallback], dtype=np.float64)


def scalarize_cost(cost: StepCosts | Any, lagrange: Any = None,
                   weights: Any = None) -> float:
    """Convert a vector cost to a minimisation scalar.

    ``lagrange`` multiplies safety components (all fields after ``track``),
    which makes the constrained objective explicit and checkpointable.
    """
    vector = cost.vector() if isinstance(cost, StepCosts) else np.asarray(cost, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.any(vector < 0.):
        raise ValueError("cost vector must be finite and non-negative")
    if weights is None:
        weights = np.ones_like(vector)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != vector.shape:
        raise ValueError("weights shape mismatch")
    if lagrange is None:
        multipliers = np.r_[1., np.ones(len(vector) - 1)]
        if len(multipliers) >= 10:
            # ``failed`` is an event bit; the absorbing penalty is in
            # ``terminal`` and must not be charged twice.
            multipliers[8] = 0.
    else:
        lam = np.asarray(lagrange, dtype=float).reshape(-1)
        if len(lam) == len(vector) - 1:
            multipliers = np.r_[1., 1. + np.maximum(lam, 0.)]
        elif len(lam) == len(vector):
            multipliers = 1. + np.maximum(lam, 0.)
        else:
            raise ValueError("lagrange shape mismatch")
    if isinstance(cost, StepCosts):
        multipliers[8] = 0.  # event bit, including when multipliers are supplied
    return float(np.dot(vector, weights * multipliers))


def aggregate_cost(steps: list[StepCosts], truncated: bool = False) -> dict[str, float | bool]:
    """Aggregate up to five 100 ms costs into one 500 ms decision record."""
    if not steps or len(steps) > 5:
        raise ValueError("one decision must contain one to five control steps")
    vectors = np.stack([step.vector() for step in steps])
    result = {"track": float(vectors[:, 0].mean()),
              "boundary": float(vectors[:, 1].max()),
              "saturation": float(vectors[:, 2].sum()),
              "margin": float(vectors[:, 3].max()),
              "qp": float(vectors[:, 4].sum()),
              "fallback": float(vectors[:, 5].sum()),
              "deadline": float(vectors[:, 6].sum()),
              "slew": float(vectors[:, 7].sum()),
              "failed": float(vectors[:, 8].max()),
              "terminal": float(vectors[:, 9].max()),
              "truncated": bool(truncated),
              "n_steps": len(steps)}
    return result
