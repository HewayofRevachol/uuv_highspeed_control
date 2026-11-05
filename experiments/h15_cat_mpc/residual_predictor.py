"""Small causal AR(1) residual forecaster and persistence baseline."""
import json
from pathlib import Path

import numpy as np
import torch
from simulator.model import MISMATCHED_PLANT


RESIDUAL_LABEL_VERSION = "nominal_5x20ms_issued_tau_once_v3"


def nominal_delay_tau(device="cpu", dtype=torch.float64, substep_s=.02,
                      fin_tau_scale=1.):
    """Return nominal delay and *effective* tau, with fin scaling applied once.

    The returned tau is ready for either actuator model; rollout callers must
    not rescale it. These values never use randomized plant delay/tau.
    """
    if not np.isfinite(fin_tau_scale) or fin_tau_scale <= 0:
        raise ValueError("fin_tau_scale must be positive and finite")
    if not np.isfinite(substep_s) or substep_s <= 0:
        raise ValueError("substep_s must be positive and finite")
    delays_s = (MISMATCHED_PLANT.thrust_delay_s,
                MISMATCHED_PLANT.fin_delay_s,
                MISMATCHED_PLANT.fin_delay_s)
    taus_s = (MISMATCHED_PLANT.thrust_tau_s,
              MISMATCHED_PLANT.fin_tau_s*fin_tau_scale,
              MISMATCHED_PLANT.fin_tau_s*fin_tau_scale)
    delays = torch.ceil(torch.tensor(delays_s, device=device, dtype=dtype)/substep_s-1e-12)
    return delays.to(torch.long)[None], torch.tensor(taus_s, device=device,
                                                     dtype=dtype)[None]


def nominal_period_rollout(model, state, actuator, actuator_rate, queue,
                           delay_steps, tau, issued_command, parameters,
                           second_order=True, integrator=True,
                           period_s=.1, substep_s=.02):
    """Roll out a period using effective tau in seconds, without further scaling.

    Use ``nominal_delay_tau(fin_tau_scale=theta[1])`` to construct tau. Keeping
    scaling outside this function makes first- and second-order models obey
    the same contract and prevents applying theta twice.
    """
    steps = round(period_s/substep_s)
    if steps < 1 or abs(steps*substep_s-period_s) > 1e-12:
        raise ValueError("period must be an integer number of actuator substeps")
    state, actuator = state.clone(), actuator.clone()
    actuator_rate, queue = actuator_rate.clone(), queue.clone()
    delays = torch.as_tensor(delay_steps, device=state.device, dtype=torch.long)
    issued = issued_command
    tau_used = torch.as_tensor(tau, device=state.device, dtype=state.dtype)
    if not torch.isfinite(tau_used).all() or (tau_used <= 0).any():
        raise ValueError("effective actuator tau must be positive and finite")
    for _ in range(steps):
        delayed = queue.gather(1, (delays-1).clamp_min(0)[:, None]).squeeze(1)
        target = torch.where(delays > 0, delayed, issued)
        queue = torch.cat((issued[:, None], queue[:, :-1]), 1)
        if second_order:
            omega = 1./tau_used.clamp_min(.03)
            acceleration = omega.square()*(target-actuator)-2.*omega*actuator_rate
            actuator_rate = actuator_rate+substep_s*acceleration
            actuator = actuator+substep_s*actuator_rate
        else:
            gain = -torch.expm1(-substep_s/tau_used.clamp_min(1e-12))
            actuator = actuator+gain*(target-actuator)
        if integrator:
            from aav_mpc.h15_target_domain import midpoint_two_substeps
            state = midpoint_two_substeps(model, state, actuator, parameters,
                torch.zeros((state.shape[0], 5), device=state.device, dtype=state.dtype),
                current=None, dt=substep_s)
        else:
            state = model.rk4(state, actuator, parameters=parameters,
                              dt=substep_s)
    return state, actuator, actuator_rate, queue


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
    def load(cls, path, expected_label_version=None,
             expected_compensation_mode=None,
             expected_normalization_sha256=None):
        value = json.loads(Path(path).read_text())
        label_version = value.get("manifest", {}).get("label_version")
        if expected_label_version is not None and label_version != expected_label_version:
            raise ValueError(f"predictor label version {label_version!r} is incompatible; "
                             f"expected {expected_label_version!r}")
        comp_mode = value.get("manifest", {}).get("compensation_mode")
        if (expected_compensation_mode is not None and
                comp_mode != expected_compensation_mode):
            raise ValueError(f"predictor compensation mode {comp_mode!r} is incompatible; "
                             f"expected {expected_compensation_mode!r}")
        normalization_sha = value.get("manifest", {}).get("normalization_sha256")
        if (expected_normalization_sha256 is not None and
                normalization_sha != expected_normalization_sha256):
            raise ValueError("predictor normalization hash is incompatible with adapter")
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
