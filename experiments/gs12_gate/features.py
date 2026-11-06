"""Causal GS-12 observations, residuals, gate reward and policy."""
from __future__ import annotations

from collections import deque
import math

import numpy as np
import torch
from torch import nn

from experiments.gs12_gate.model_bridge import THETA_SAFE, NU_INDEX
from simulator.batch_mpc import CONTROL_SCALE

FRAME_NAMES = ("east_error", "depth_error", "north_velocity_error", "north_error",
               "pitch", "q", "r", "actual_thrust", "actual_elevator", "actual_rudder",
               "issued_thrust", "issued_elevator", "issued_rudder",
               "innovation_u", "innovation_v", "innovation_w", "innovation_q", "innovation_r")
CURRENT_NAMES = ("theta_gain", "theta_tau", "radius_gain", "radius_tau",
                 "previous_alpha_theta", "previous_alpha_u", "previous_alpha_vertical",
                 "previous_alpha_lateral", "margin_thrust", "margin_elevator",
                 "margin_rudder", "qp_success", "solve_time",
                 "preview_east_0p5", "preview_depth_0p5", "preview_vn_0p5",
                 "preview_east_1p0", "preview_depth_1p0", "preview_vn_1p0",
                 "valid_history_fraction")
FRAME_SCALE = np.array([8., 5., 2., 8., np.deg2rad(40), 1., 1.,
                        *CONTROL_SCALE, *CONTROL_SCALE, 1., 1., 1., .1, .1])
CURRENT_SCALE = np.array([1., 2., .5, 2.5, 1., 1., 1., 1.,
                          1., 1., 1., 1., 100., 8., 5., 2., 8., 5., 2., 1.])


def schema():
    names = [f"frame_{lag:02d}.{name}" for lag in range(9, -1, -1) for name in FRAME_NAMES]
    names += list(CURRENT_NAMES)
    units = (["1"]*4 + ["rad", "rad/s", "rad/s"] +
             ["N", "rad", "rad"]*2 + ["m/s2"]*3 + ["rad/s2"]*2)*10
    units[:4] = ["m", "m", "m/s", "m"]
    for start in range(18, 180, 18):
        units[start:start+4] = ["m", "m", "m/s", "m"]
    units += ["1"]*12 + ["ms"] + ["m", "m", "m/s"]*2 + ["1"]
    scales = np.r_[np.tile(FRAME_SCALE, 10), CURRENT_SCALE]
    assert len(names) == len(units) == len(scales) == 200
    frame_sources = (["measured_state_and_known_reference"]*4+
                     ["measured_hull_state"]*3+
                     ["measured_actuator"]*3+
                     ["issued_command"]*3+
                     ["completed_transition_model_innovation"]*5)
    return {"feature_names": names, "units": units, "scale": scales.tolist(),
            "source": frame_sources*10+
                      ["causal_estimate"]*4+["prior_gate"]*4+
                      ["measured_margin"]*3+["prior_qp"]*2+
                      ["known_reference_preview"]*6+["history_count"],
            "timestamp": [f"t-{lag*.1:.1f}s" for lag in range(9, -1, -1)
                          for _ in FRAME_NAMES]+["t"]*20}


class ObservationHistory:
    def __init__(self, scale=None):
        self.frames = deque(maxlen=10)
        self.scale = np.asarray(scale if scale is not None else schema()["scale"], float)
        if self.scale.shape != (200,) or np.any(self.scale <= 0):
            raise ValueError("invalid observation scale")

    def reset(self):
        self.frames.clear()

    def append(self, frame):
        value = np.asarray(frame, float)
        if value.shape != (18,) or not np.isfinite(value).all():
            raise ValueError("frame must be finite [18]")
        self.frames.append(value.copy())

    def observe(self, current):
        current = np.asarray(current, float)
        if not self.frames or current.shape != (20,) or not np.isfinite(current).all():
            raise ValueError("observation requires frame [18] and current [20]")
        frames = [self.frames[0]]*(10-len(self.frames)) + list(self.frames)
        value = np.r_[np.asarray(frames).ravel(), current]
        value[-1] = len(self.frames)/10
        return np.clip(value/self.scale, -10., 10.).astype(np.float32)


class CausalResidual:
    """Ten completed .1 s transitions, re-predicted under current theta."""
    def __init__(self, dynamics, limits=(1., 1., 1., .1, .1)):
        self.dynamics = dynamics
        self.limits = np.asarray(limits, float)
        self.history = deque(maxlen=10)

    def reset(self):
        self.history.clear()

    def append(self, z_before, issued_normalized, nu_after, actual_dt=.1):
        if abs(actual_dt-.1) > 1e-9:
            return False
        z = np.asarray(z_before, float)
        u = np.asarray(issued_normalized, float)
        nu = np.asarray(nu_after, float)
        if z.shape != (25,) or u.shape != (3,) or nu.shape != (5,):
            raise ValueError("residual transition shape mismatch")
        self.history.append((z.copy(), u.copy(), nu.copy()))
        return True

    def estimate(self, theta):
        if not self.history:
            return np.zeros(5), np.zeros(5), np.zeros(5), False
        theta = np.asarray(theta, float)
        labels = []
        for z, u, nu_after in reversed(self.history):
            f_ref = self.dynamics.evaluate(z[None, None], u[None, None], THETA_SAFE)[0][0, 0]
            f_theta = self.dynamics.evaluate(z[None, None], u[None, None], theta)[0][0, 0]
            d_ref = (nu_after-f_ref[NU_INDEX])/.1
            labels.append((d_ref, d_ref-(f_theta[NU_INDEX]-f_ref[NU_INDEX])/.1))
        weights = .8**np.arange(len(labels)); weights /= weights.sum()
        ref = weights@np.asarray([row[0] for row in labels])
        raw = weights@np.asarray([row[1] for row in labels])
        clipped = np.clip(raw, -self.limits, self.limits)
        return ref, raw, clipped, bool(np.any(clipped != raw))

    def latest_ref(self):
        if not self.history:
            return np.zeros(5)
        z, u, nu_after = self.history[-1]
        predicted = self.dynamics.evaluate(z[None, None], u[None, None], THETA_SAFE)[0][0, 0]
        return (nu_after-predicted[NU_INDEX])/.1


class GateActorCritic(nn.Module):
    VARIANTS = {
        "standard": (200, 4),
        "tied_residual": (200, 2),
        "no_history": (38, 4),
    }

    def __init__(self, variant="standard"):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown gate policy variant: {variant}")
        self.variant = variant
        input_dim, self.action_dim = self.VARIANTS[variant]
        def network(output):
            return nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64),
                                 nn.Tanh(), nn.Linear(64, output))
        self.actor, self.critic = network(self.action_dim), network(1)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def _input(self, observation):
        if observation.shape[-1] != 200:
            raise ValueError("policy requires canonical 200-column observation")
        # Same information types, but only the latest 18-column frame plus
        # the current 20 columns for the no-history retraining ablation.
        return observation[..., 162:] if self.variant == "no_history" else observation

    def distribution(self, observation):
        mean = self.actor(self._input(observation))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def log_prob(self, observation, raw_z):
        return self.distribution(observation).log_prob(raw_z).sum(-1)

    def value(self, observation):
        return self.critic(self._input(observation)).squeeze(-1)

    def act(self, observation, deterministic=False):
        dist = self.distribution(observation)
        raw = dist.mean if deterministic else dist.sample()
        return {"raw_z": raw, "log_prob": dist.log_prob(raw).sum(-1),
                "target": torch.sigmoid(raw), "value": self.value(observation)}


def fast_reward(errors, pitch, delta_command_norm, saturation_fraction, dt=.1):
    e_e, e_d, e_vn, e_n = map(float, errors)
    cost = (e_e/8)**2 + (e_d/5)**2 + (e_vn/2)**2 + .25*(e_n/8)**2
    cost += .1*(pitch/np.deg2rad(40))**2
    cost += .01*float(np.dot(delta_command_norm, delta_command_norm))
    cost += .02*float(saturation_fraction)
    return -dt*min(max(cost, 0.), 50.), cost


def failure_cost(first_failure_s, planned_duration_s):
    return -(20. + 100.*max(0., planned_duration_s-first_failure_s))
