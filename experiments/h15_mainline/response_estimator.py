"""Bounded sliding-window equivalent fin-response estimator for H15.

Only measured states, measured actuators and issued/queued commands enter the
fit. The result is an effective short-window model, not true plant parameters.
State replay radii come from a separate dev-only calibration artifact. Parameter
radii span the full configured projection range and are deliberately coarse.
"""

from collections import deque
from dataclasses import dataclass
import json
import math
import time

import numpy as np

from simulator.model import (MASS, M_EFF, D_COEFF, ELEVATOR_COEFF,
                             RUDDER_COEFF, MISMATCHED_PLANT)
from experiments.h15_mainline.observation_features import FEATURE_NAMES, from_period_row


@dataclass(frozen=True)
class ResponseEstimate:
    fin_gain: float
    fin_tau_scale: float
    gain_radius: float
    tau_radius: float
    source: str
    fallback_reason: str | None
    forward_ms: float
    fallback_ms: float
    samples: int
    constraint_radii: dict | None = None


class SlidingWindowResponseEstimator:
    """Estimate common fin gain and lag from the last 10 observed periods."""

    DEFAULTS = {"window": 10, "minimum_samples": 8, "ridge_gain": .02,
                "ridge_actuator": 1e-5, "minimum_gain_information": 1e-4,
                "minimum_actuator_information": 2e-4,
                "recency_weight": .9,
                "max_age_s": .15, "max_abs_normalized_feature": 8.,
                "gain_bounds": [.6, 1.1], "tau_scale_bounds": [.5, 3.],
                "max_gain_change_per_period": .05,
                "max_tau_change_per_period": .2,
                "e0_gain": .85, "e0_tau_scale": 2.}

    def __init__(self, normalization, parameters=None, calibration=None):
        if tuple(normalization["feature_names"]) != FEATURE_NAMES or normalization["split"] != "train":
            raise ValueError("expected frozen train observation statistics")
        self.normalization = normalization
        self.parameters = dict(self.DEFAULTS if parameters is None else parameters)
        if calibration is not None:
            if (calibration.get("method") != "empirical_dev_replay_residual_quantile" or
                    calibration.get("horizons_control_periods") != [1, 5, 25, 35]):
                raise ValueError("expected frozen dev replay calibration")
            self.constraint_radii = calibration["radii"]
        else:
            self.constraint_radii = None
        self.history = deque(maxlen=int(self.parameters["window"]))
        self.last = None
        p = MISMATCHED_PLANT
        self.inverse_mass = np.linalg.inv(p.mass_scale * M_EFF)

    def reset(self):
        self.history.clear()
        self.last = None

    def _fallback(self, reason, started):
        tick = time.perf_counter()
        p = self.parameters
        gain_radius, tau_radius = self._parameter_radii(
            p["e0_gain"], p["e0_tau_scale"])
        result = ResponseEstimate(p["e0_gain"], p["e0_tau_scale"],
                                  gain_radius, tau_radius, "E0", reason,
                                  (tick-started)*1000.,
                                  (time.perf_counter()-tick)*1000.,
                                  len(self.history), self.constraint_radii)
        self.last = result
        return result

    def _parameter_radii(self, gain, tau):
        p = self.parameters
        gain_radius = max(gain-p["gain_bounds"][0], p["gain_bounds"][1]-gain)
        tau_radius = max(tau-p["tau_scale_bounds"][0], p["tau_scale_bounds"][1]-tau)
        return float(gain_radius), float(tau_radius)

    @staticmethod
    def _read_sample(row):
        state_before = np.asarray(json.loads(row["state_before_json"]), dtype=float)
        state_after = np.asarray(json.loads(row["state_after_json"]), dtype=float)
        actuator_before = np.asarray(json.loads(row["actuator_before_json"]), dtype=float)
        actuator_after = np.asarray([
            float(row["thrust_actual_n"]),
            math.radians(float(row["elevator_actual_deg"])),
            math.radians(float(row["rudder_actual_deg"]))])
        queue = np.asarray(json.loads(row["queue_before_json"]), dtype=float)
        command = np.asarray(json.loads(row["command_json"]), dtype=float)
        mpc_first_command = np.asarray(json.loads(row["mpc_base_command_json"]),
                                       dtype=float)
        mpc_plan = np.asarray(json.loads(row["mpc_plan_commands_json"]),
                              dtype=float)
        if (state_before.shape != (10,) or state_after.shape != (10,) or
                actuator_before.shape != (3,) or actuator_after.shape != (3,) or
                queue.shape != (5, 3) or command.shape != (3,) or
                mpc_first_command.shape != (3,) or mpc_plan.shape != (35, 3)):
            raise ValueError("malformed observed state, actuator, queue or MPC plan")
        values = np.concatenate((state_before, state_after, actuator_before,
                                 actuator_after, queue.ravel(), command,
                                 mpc_first_command, mpc_plan.ravel()))
        if not np.isfinite(values).all():
            raise ValueError("nonfinite history input")
        if not np.allclose(mpc_plan[0], mpc_first_command, rtol=1e-6, atol=1e-3):
            raise ValueError("MPC plan first command differs from logged base command")
        return (float(row["time_s"]), state_before, state_after,
                actuator_before, actuator_after, queue, command,
                mpc_plan)

    def update(self, row):
        """Consume an observed period and return a bounded estimate or E0."""
        started = time.perf_counter()
        try:
            features = from_period_row(row)
            sample = self._read_sample(row)
            if self.history and sample[0] <= self.history[-1][0]:
                self.reset()
            if not math.isfinite(sample[0]):
                return self._fallback("nonfinite_timestamp", started)
            self.history.append(sample)
            if len(self.history) < self.parameters["minimum_samples"]:
                return self._fallback("short_history", started)
            if abs(self.history[-1][0] - self.history[-2][0] - .1) > self.parameters["max_age_s"]:
                return self._fallback("stale_history", started)
            z = (np.asarray(features) - self.normalization["mean"]) / self.normalization["scale"]
            if not np.isfinite(z).all() or np.max(np.abs(z)) > self.parameters["max_abs_normalized_feature"]:
                return self._fallback("ood_observation", started)
            result = self._fit(started)
        except (ValueError, KeyError, TypeError, IndexError, OverflowError,
                ZeroDivisionError, np.linalg.LinAlgError) as exc:
            return self._fallback("invalid_history:" + type(exc).__name__, started)
        self.last = result
        return result

    def _fit(self, started):
        p = self.parameters
        gain_design, gain_targets, gain_weights = [], [], []
        actuator_x, actuator_y, actuator_weights = [[], []], [[], []], [[], []]
        nominal = MISMATCHED_PLANT
        for index, (_, before, after, actual_before, actual_after, queue,
                    command, _) in enumerate(self.history):
            weight = p["recency_weight"] ** (len(self.history)-index-1)
            state = (before + after) / 2.
            actual = (actual_before + actual_after) / 2.
            u, v, w, q, r = state[5:10]
            nu = state[5:10]
            coriolis = nominal.mass_scale * MASS * np.array(
                [-r*v+q*w, r*u, -q*u, 0., 0.])
            base_force = np.array([nominal.thrust_gain * actual[0], 0., 0., 0., 0.])
            base_acc = self.inverse_mass @ (
                base_force - coriolis - nominal.damping_scale*u*D_COEFF@nu)
            elevator_acc = self.inverse_mass @ (
                u*u*nominal.elevator_gain*ELEVATOR_COEFF*actual[1])
            rudder_acc = self.inverse_mass @ (
                u*u*nominal.rudder_gain*RUDDER_COEFF*actual[2])
            gain_design.extend(zip(elevator_acc[3:5], rudder_acc[3:5]))
            gain_targets.extend((after[8:10]-before[8:10])/.1 - base_acc[3:5])
            gain_weights.extend((weight, weight))
            for j, channel in enumerate((1, 2)):
                # Nominal three-substep fin delay: three queued values, then
                # two current commands. True delay is deliberately not read.
                target = (queue[2, channel] + queue[1, channel] +
                          queue[0, channel] + 2*command[channel]) / 5.
                actuator_x[j].append(target - actual_before[channel])
                actuator_y[j].append(actual_after[channel] - actual_before[channel])
                actuator_weights[j].append(weight)
        design = np.asarray(gain_design) * np.sqrt(np.asarray(gain_weights))[:, None]
        targets = np.asarray(gain_targets) * np.sqrt(np.asarray(gain_weights))
        information = np.linalg.eigvalsh(design.T@design)
        if information[0] < p["minimum_gain_information"]:
            return self._fallback("low_gain_excitation", started)
        alpha = []
        for x, y, weights in zip(actuator_x, actuator_y, actuator_weights):
            root_weight = np.sqrt(np.asarray(weights))
            x, y = np.asarray(x)*root_weight, np.asarray(y)*root_weight
            information = float(x@x)
            if information < p["minimum_actuator_information"]:
                return self._fallback("low_actuator_excitation", started)
            value = float((x@y + p["ridge_actuator"]*.46) /
                          (information + p["ridge_actuator"]))
            alpha.append(np.clip(value, .05, .95))
        lag = float(np.median([-0.1/math.log1p(-a)/nominal.fin_tau_s for a in alpha]))
        ridge = p["ridge_gain"]
        gains = np.linalg.solve(design.T@design + ridge*np.eye(2),
                                design.T@targets + ridge*np.full(2, p["e0_gain"]))
        gain = float(np.median(gains))
        gain = float(np.clip(gain, *p["gain_bounds"]))
        lag = float(np.clip(lag, *p["tau_scale_bounds"]))
        if self.last is not None and self.last.source == "E1":
            gain = float(np.clip(gain,
                                 self.last.fin_gain-p["max_gain_change_per_period"],
                                 self.last.fin_gain+p["max_gain_change_per_period"]))
            lag = float(np.clip(lag,
                                self.last.fin_tau_scale-p["max_tau_change_per_period"],
                                self.last.fin_tau_scale+p["max_tau_change_per_period"]))
        if not np.isfinite([gain, lag]).all():
            return self._fallback("nonfinite_fit", started)
        gain_radius, tau_radius = self._parameter_radii(gain, lag)
        return ResponseEstimate(gain, lag, gain_radius,
                                tau_radius, "E1", None,
                                (time.perf_counter()-started)*1000., 0.,
                                len(self.history), self.constraint_radii)
