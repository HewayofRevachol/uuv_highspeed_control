"""Frozen wrapper around the existing causal sliding-window estimator."""
import json
from pathlib import Path

from experiments.h15_mainline.response_estimator import SlidingWindowResponseEstimator


THETA0 = (0.85, 2.0)


class FrozenResponseAdapter:
    def __init__(self, normalization_path=None, calibration_path=None,
                 parameters=None):
        self.estimator = None
        if normalization_path is not None:
            normalization = json.loads(Path(normalization_path).read_text())
            calibration = (json.loads(Path(calibration_path).read_text())
                           if calibration_path is not None else None)
            self.estimator = SlidingWindowResponseEstimator(
                normalization, parameters, calibration)
        self.last = None

    def reset(self):
        if self.estimator is not None:
            self.estimator.reset()
        self.last = None

    def update(self, completed_period_row):
        """Consume only a completed-period row; absent artifacts mean E0."""
        if self.estimator is None:
            return {"theta_hat": THETA0, "source": "E0",
                    "fallback_reason": "frozen_artifact_missing",
                    "radius": (0.25, 2.5), "forward_ms": 0.}
        result = self.estimator.update(completed_period_row)
        self.last = result
        return {"theta_hat": (result.fin_gain, result.fin_tau_scale),
                "source": result.source,
                "fallback_reason": result.fallback_reason,
                "radius": (result.gain_radius, result.tau_radius),
                "forward_ms": result.forward_ms}
