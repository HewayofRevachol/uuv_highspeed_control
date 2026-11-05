"""Small causal AR(1) residual forecaster and persistence baseline."""
import json
from pathlib import Path

import numpy as np


class ResidualAR1:
    def __init__(self, phi=None, mean=None, version="persistence"):
        self.phi = np.ones(5) if phi is None else np.asarray(phi, dtype=float)
        self.mean = np.zeros(5) if mean is None else np.asarray(mean, dtype=float)
        self.version = version
        if self.phi.shape != (5,) or self.mean.shape != (5,):
            raise ValueError("AR(1) requires five residual dimensions")

    @classmethod
    def fit(cls, train_residuals):
        values = np.asarray(train_residuals, dtype=float)
        if values.ndim != 2 or values.shape[1] != 5 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("train residuals must be finite [T,5]")
        mean = values.mean(0)
        x, y = values[:-1]-mean, values[1:]-mean
        denom = np.sum(x*x, axis=0)
        phi = np.divide(np.sum(x*y, axis=0), denom,
                        out=np.zeros(5), where=denom > 1e-12)
        return cls(np.clip(phi, -.98, .98), mean, "train_ar1")

    @classmethod
    def fit_episodes(cls, episodes):
        values = [np.asarray(x, dtype=float) for x in episodes]
        if not values or any(x.ndim != 2 or x.shape[1] != 5 or
                             not np.isfinite(x).all() for x in values):
            raise ValueError("episodes must be finite residual arrays [T,5]")
        joined = np.concatenate(values, axis=0)
        mean = joined.mean(0)
        pairs = [(x[:-1]-mean, x[1:]-mean) for x in values if len(x) > 1]
        x, y = np.concatenate([pair[0] for pair in pairs]), np.concatenate(
            [pair[1] for pair in pairs])
        denom = np.sum(x*x, axis=0)
        phi = np.divide(np.sum(x*y, axis=0), denom,
                        out=np.zeros(5), where=denom > 1e-12)
        return cls(np.clip(phi, -.98, .98), mean, "train_ar1")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        return cls(value["phi"], value["mean"], value["version"])

    def predict(self, history, horizon=35):
        history = np.asarray(history, dtype=float)
        if history.ndim != 2 or history.shape[1] != 5 or not np.isfinite(history).all():
            raise ValueError("residual history must be finite [T,5]")
        last = history[-1] if len(history) else self.mean
        value = last.copy()
        forecast = np.empty((horizon, 5), dtype=float)
        for i in range(horizon):
            value = self.mean + self.phi*(value-self.mean)
            forecast[i] = value
        return forecast

    def save(self, path, manifest=None):
        value = {"schema": 1, "version": self.version,
                 "phi": self.phi.tolist(), "mean": self.mean.tolist(),
                 "manifest": manifest or {}}
        Path(path).write_text(json.dumps(value, indent=2)+"\n")
