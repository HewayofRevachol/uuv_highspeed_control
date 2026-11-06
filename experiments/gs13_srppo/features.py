"""Small, causal feature and reward adapter for GS-12 episode integration."""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .costs import StepCosts, scalarize_cost


OBSERVATION_DIM = 200
FRAME_DIM = 18
CURRENT_DIM = 20


class CausalObservation:
    """Stack ten completed frames and a current causal context."""

    def __init__(self, scale: Any | None = None):
        self.scale = np.ones(OBSERVATION_DIM, dtype=float) if scale is None else np.asarray(scale, dtype=float)
        if self.scale.shape != (OBSERVATION_DIM,) or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("scale must be finite positive [200]")
        self.frames: deque[np.ndarray] = deque(maxlen=10)

    def reset(self) -> None:
        self.frames.clear()

    def append(self, frame: Any) -> None:
        value = np.asarray(frame, dtype=float)
        if value.shape != (FRAME_DIM,) or not np.isfinite(value).all():
            raise ValueError("frame must be finite [18]")
        self.frames.append(value.copy())

    def observe(self, current: Any) -> np.ndarray:
        value = np.asarray(current, dtype=float)
        if value.shape != (CURRENT_DIM,) or not np.isfinite(value).all() or not self.frames:
            raise ValueError("current must be finite [20] and a frame must exist")
        frames = [self.frames[0]] * (10 - len(self.frames)) + list(self.frames)
        result = np.r_[np.asarray(frames).reshape(-1), value]
        result[-1] = len(self.frames) / 10.
        return np.clip(result / self.scale, -10., 10.).astype(np.float32)


def reward_components(cost: StepCosts, lagrange: Any = None) -> dict[str, float]:
    """Expose positive cost components and the negative learner reward."""
    values = cost.as_dict()
    return {"reward_tracking": -float(values["track"]),
            "reward_safety": -float(sum(values[key] for key in ("boundary", "saturation", "margin", "qp", "fallback", "deadline"))),
            "reward_slew": -float(values["slew"]),
            "failure_cost": -float(values["terminal"]),
            "reward": -scalarize_cost(cost, lagrange)}


def schema() -> dict[str, Any]:
    return {"version": "gs13_observation_200_v1", "dimension": OBSERVATION_DIM,
            "frame_dimension": FRAME_DIM, "current_dimension": CURRENT_DIM,
            "causal": True, "scale_source": "train_calibration"}
