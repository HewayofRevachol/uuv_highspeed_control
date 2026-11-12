"""Eight-decision beta episode over the unmodified GS-12 C4 plant and MPC."""
from __future__ import annotations

import numpy as np

from experiments.gs12_joint_route.joint_reference import joint_task_reference
from experiments.gs12_weight_route.runner_core import ClosedLoopSession, SCALE
from .features import observation
from .reward import segment_reward, tracking_error_squared

ACTION_BETAS = (0., .25, .5, .75)
CYCLES_PER_DECISION = 80
SLEW_PER_CYCLE = .02


class BetaEpisode:
    """One task, one continuous ClosedLoopSession, eight possible decisions.

    Pass the frozen task's full F75 J_track as j_f75 for the training reward.
    For an interface smoke test, a positive placeholder is acceptable because
    that reward is not used for model selection.
    """

    def __init__(self, task, j_f75, initial_beta=.75):
        if initial_beta not in ACTION_BETAS:
            raise ValueError("initial_beta must be one of the pilot actions")
        if not np.isfinite(j_f75) or j_f75 < 0:
            raise ValueError("j_f75 must be finite and nonnegative")
        self.session = ClosedLoopSession(
            task, controller="C4", a_depth=1.4, a_rudder_rate=.7,
            duration_s=64., reference_fn=joint_task_reference,
            predictor_params=(1., 1.), dynamic_predictor=True, preview_s=16.)
        self.j_f75 = float(j_f75)
        self.beta_actual = float(initial_beta)
        self.beta_prev_target = float(initial_beta)
        self.previous_j = 0.
        self.previous_slew = 0.
        self.decisions = 0
        self.done = False

    def observe(self):
        if self.done:
            raise RuntimeError("terminal episode has no next decision")
        return observation(self.session, self.beta_actual, self.beta_prev_target,
                           self.previous_j, self.previous_slew)

    def step(self, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTION_BETAS):
            raise ValueError("action must be an integer in [0, 3]")
        return self.step_target(ACTION_BETAS[int(action)])

    def step_target(self, target):
        """Apply a continuous beta target; retain the existing 0.02/cycle slew."""
        if self.done:
            raise RuntimeError("episode already finished")
        target = float(target)
        if not np.isfinite(target) or not 0. <= target <= .75:
            raise ValueError("beta target must be finite and in [0, .75]")
        previous_target = self.beta_prev_target
        start_k = self.session.k
        start_sub = len(self.session.fields["time_substep_s"])
        start_cmd = max(0, len(self.session.fields["command"])-1)
        beta_path = []
        failure = None
        for _ in range(CYCLES_PER_DECISION):
            self.beta_actual += float(np.clip(target-self.beta_actual,
                                              -SLEW_PER_CYCLE, SLEW_PER_CYCLE))
            self.beta_actual = float(np.clip(self.beta_actual, 0., .75))
            beta_path.append(self.beta_actual)
            event = self.session.advance_control_cycle(self.beta_actual)
            if not event["qp_ok"]:
                failure = "qp_failure"
            elif event["first_violation"] is not None:
                failure = "early_violation"
            if failure or event["done"]:
                break
        fields = self.session.fields
        out = np.asarray(fields["output_substep"][start_sub:])
        ref = np.asarray(fields["reference_substep"][start_sub:])
        squared = tracking_error_squared(out, ref)
        j_segment = float(np.mean(squared))
        physical_error = ref[:, [1, 2, 5]] - out[:, [1, 2, 5]]
        commands = np.asarray(fields["command"][start_cmd:])/SCALE
        actual = np.asarray(fields["actual_substep"][start_sub:])/SCALE
        command_slew = float(np.max(np.abs(np.diff(commands, axis=0)))) if len(commands) > 1 else 0.
        actual_slew = float(np.max(np.abs(np.diff(actual, axis=0)))) if len(actual) > 1 else 0.
        lower = np.array([0., -1., -1.])
        command_sat = bool(np.any((commands >= 1.-1e-6) | (commands <= lower+1e-6)))
        actual_sat = bool(np.any((actual >= 1.-1e-6) | (actual <= lower+1e-6)))
        solves = np.asarray(fields["solve_ms"][start_k:self.session.k])
        self.decisions += 1
        self.beta_prev_target = float(target)
        self.previous_j = j_segment
        self.previous_slew = command_slew
        self.done = failure is not None or self.session.done
        reward = segment_reward(j_segment, self.j_f75, target, previous_target,
                                self.decisions, failed=failure is not None)
        info = {
            "segment_index": self.decisions,
            "start_s": start_k*.1,
            "end_s": float(fields["time_substep_s"][-1]),
            "beta_target": target,
            "beta_actual": beta_path,
            "J_seg": j_segment,
            "rmse_E_D_VN": np.sqrt(np.mean(physical_error**2, axis=0)).tolist(),
            "peak_E_D_VN": np.max(np.abs(physical_error), axis=0).tolist(),
            "qp_failures": int(np.sum(~np.asarray(fields["qp_ok"][start_k:self.session.k]))),
            "qp_primal_max": float(np.max(fields["qp_primal"][start_k:self.session.k])),
            "qp_dual_max": float(np.max(fields["qp_dual"][start_k:self.session.k])),
            "command_saturation": command_sat,
            "actual_saturation": actual_sat,
            "max_normalized_command_slew": command_slew,
            "max_normalized_actual_slew": actual_slew,
            "solve_p50_ms": float(np.percentile(solves, 50)),
            "solve_p95_ms": float(np.percentile(solves, 95)),
            "solve_p99_ms": float(np.percentile(solves, 99)),
            "terminal_reason": failure or ("duration" if self.done else None),
        }
        next_observation = (np.zeros(19, dtype=np.float32) if self.done
                            else self.observe())
        return next_observation, reward, self.done, info

    def result(self):
        return self.session.result()
