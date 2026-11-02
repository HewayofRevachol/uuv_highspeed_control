"""Filtered Smith-style state preview for the delayed UUV actuators.

The plant state is measured without sensor delay.  Here the predictor moves
that state to the time at which a new fin command has a useful effect.  The
filtered historical velocity prediction error corrects model mismatch, while
the command history and first-order actuator model handle actuation delay.
"""

import numpy as np
import torch

from simulator.model import DT, MISMATCHED_PLANT


class FilteredActuationPredictor:
    def __init__(self, model, batch_size, control_dt=.1, filter_time=.4,
                 preview_time=None):
        if not np.isfinite(filter_time) or filter_time <= 0:
            raise ValueError("filter_time must be finite and positive")
        if (not np.isfinite(control_dt) or control_dt <= 0 or
                not np.isclose(control_dt/DT, round(control_dt/DT))):
            raise ValueError("control_dt must be a positive multiple of actuator DT")
        nominal = MISMATCHED_PLANT
        self.model = model
        self.control_dt = control_dt
        self.control_steps = round(control_dt/DT)
        self.delay_steps = model.tensor([
            np.ceil(nominal.thrust_delay_s/DT - 1e-12),
            np.ceil(nominal.fin_delay_s/DT - 1e-12),
            np.ceil(nominal.fin_delay_s/DT - 1e-12),
        ]).long()
        self.tau = model.tensor([nominal.thrust_tau_s, nominal.fin_tau_s,
                                  nominal.fin_tau_s])
        effective_time = nominal.fin_delay_s + nominal.fin_tau_s
        if preview_time is None:
            preview_time = effective_time
        if not np.isfinite(preview_time) or preview_time < 0:
            raise ValueError("preview_time must be finite and nonnegative")
        self.preview_steps = int(np.ceil(preview_time/DT - 1e-12))
        self.preview_time = self.preview_steps*DT
        self.filter_gain = -np.expm1(-control_dt/filter_time)
        self.velocity_error = torch.zeros(batch_size, 5, device=model.device,
                                          dtype=model.dtype)
        self.expected_next = torch.zeros(batch_size, 10, device=model.device,
                                         dtype=model.dtype)
        self.expected_valid = torch.zeros(batch_size, device=model.device,
                                          dtype=torch.bool)

    def reset(self, mask):
        self.velocity_error[mask] = 0
        self.expected_next[mask] = 0
        self.expected_valid[mask] = False

    def observe(self, measured_state, mask=None):
        if mask is None:
            mask = torch.ones(measured_state.shape[0], device=self.model.device,
                              dtype=torch.bool)
        innovation = measured_state[:, 5:] - self.expected_next[:, 5:]
        innovation = innovation.clamp(min=self.model.tensor([-1., -1., -1., -.2, -.2]),
                                      max=self.model.tensor([1., 1., 1., .2, .2]))
        update = self.velocity_error + self.filter_gain*(innovation-self.velocity_error)
        self.velocity_error = torch.where((mask & self.expected_valid)[:, None],
                                          update, self.velocity_error)
        self.expected_valid[mask] = False

    def _rollout(self, state, actuator, history, command, steps, current):
        predicted = state.clone()
        actual = actuator.clone()
        queue = history.clone()
        gain = torch.where(self.tau > 0,
                           -torch.expm1(-DT/self.tau.clamp_min(1e-12)),
                           torch.ones_like(self.tau))
        delays = self.delay_steps.expand(state.shape[0], -1)
        for _ in range(steps):
            delayed = queue.gather(1, (delays-1).clamp_min(0)[:, None]).squeeze(1)
            target = torch.where(delays > 0, delayed, command)
            queue = torch.cat((command[:, None], queue[:, :-1]), 1)
            actual = actual + gain*(target-actual)
            predicted = self.model.rk4(predicted, actual, current=current, dt=DT)
        return predicted

    def expect_next(self, state, actuator, history, command, current, mask=None):
        predicted = self._rollout(state, actuator, history, command,
                                  self.control_steps, current)
        if mask is None:
            mask = torch.ones(state.shape[0], device=self.model.device,
                              dtype=torch.bool)
        self.expected_next = torch.where(mask[:, None], predicted, self.expected_next)
        self.expected_valid |= mask

    def preview(self, state, actuator, history, command, current):
        predicted = self._rollout(state, actuator, history, command,
                                  self.preview_steps, current)
        if self.preview_steps:
            # A historical one-control-period velocity error estimates the
            # velocity drift over the forthcoming preview interval.
            error_scale = self.preview_time/self.control_dt
            predicted[:, 5:] += error_scale*self.velocity_error
            predicted[:, 3] += .5*self.preview_time*error_scale*self.velocity_error[:, 3]
            predicted[:, 4] += .5*self.preview_time*error_scale*self.velocity_error[:, 4]
        return predicted
