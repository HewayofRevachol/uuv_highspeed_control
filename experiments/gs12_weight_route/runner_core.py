"""Minimal GS-12 closed loop with a per-solve weight hook."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "gs12_analaysis" / "frequency_extension_src"
os.environ["GS12_REPO"] = str(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import plant_harness as harness
from simulator.batch_mpc import BatchedUUVModel, CONTROL_SCALE
from simulator.rti_symbolic import ActuatorAwareCpuDynamics
from aav_mpc.condensed_mpc import CondensedActuatorMPC
from experiments.gs12_weight_route.reference_tasks import task_reference

SCALE = np.asarray(CONTROL_SCALE, dtype=float)


class ClosedLoopSession:
    """The existing run_one plant/MPC loop, exposed one control cycle at a time."""

    def __init__(self, task, controller="C4", a_depth=1., a_rudder_rate=1.,
                 duration_s=64., reference_fn=None, predictor_params=None,
                 dynamic_predictor=False, preview_s=0.):
        torch.set_num_threads(1)
        self.task = task
        self.a_depth = a_depth
        self.a_rudder_rate = a_rudder_rate
        self.dynamic_predictor = dynamic_predictor
        config = harness.PlantConfig(elevator=.9, rudder=.9, order="second_all")
        self.model = BatchedUUVModel("cpu")
        self.mpc = CondensedActuatorMPC(self.model, 1, horizon=35, dt=.1)
        model_gain, model_tau = predictor_params if predictor_params is not None else (
            (1., 1.) if controller == "C4" else (.85, 2.))
        if dynamic_predictor:
            # Dynamic predictor path keeps the P0 base model at every solve.
            model_gain, model_tau = 1., 1.
        if not np.isfinite([model_gain, model_tau]).all() or model_gain <= 0 or model_tau <= 0:
            raise ValueError("predictor_params must contain finite positive gain and tau")
        self.mpc.dynamics = ActuatorAwareCpuDynamics(.1, model_gain, model_tau)
        self.plant = harness.Plant(task.get("speed_mps", 15.), config)
        self.mpc.previous[:] = self.model.tensor(self.plant.actual/SCALE)
        self.mpc.plan[:] = self.mpc.previous[:, None]
        self.nsteps = round(duration_s/.1)
        ref_fn = task_reference if reference_fn is None else reference_fn
        self.ref_grid = ref_fn(
            task, np.arange(round((duration_s+preview_s)/.02)+5*36+1)*.02)
        self.fields = {k: [] for k in (
            "time_substep_s", "state_substep", "output_substep",
            "reference_substep", "diagnostic_substep", "actual_substep",
            "command_substep", "time_control_s", "command", "qp_ok",
            "qp_primal", "qp_dual", "a_depth", "a_rudder_rate", "solve_ms")}
        if dynamic_predictor:
            self.fields.update({k: [] for k in ("beta_t", "model_gain_t", "model_tau_t")})
        self.started = time.perf_counter()
        self.first_violation = None
        self.k = 0

    @property
    def done(self):
        return self.k >= self.nsteps or self.first_violation is not None

    def advance_control_cycle(self, beta=None):
        """Advance exactly one MPC solve and up to five original plant substeps."""
        if self.done:
            raise RuntimeError("closed-loop session already finished")
        if self.dynamic_predictor != (beta is not None):
            raise ValueError("beta is required exactly when dynamic_predictor is enabled")
        k = self.k
        fields = self.fields
        mpc = self.mpc
        model = self.model
        plant = self.plant
        ref_grid = self.ref_grid
        scales = (float(self.a_depth), float(self.a_rudder_rate))
        mpc.set_tracking_weight_scales(*scales)
        state = model.tensor(plant.controller_state())[None]
        refs = ref_grid[(5*(k+1)):(5*(k+36)):5]
        tic = time.perf_counter()
        if beta is None:
            command_t, ok = mpc.solve(state, model.tensor(refs)[None])
        else:
            beta = float(beta)
            command_t, ok = mpc.solve(state, model.tensor(refs)[None],
                                      alpha=[beta, 0.], theta_hat=[.85, 2.], d_hat=None)
            fields["beta_t"].append(beta)
            fields["model_gain_t"].append(1.-.15*beta)
            fields["model_tau_t"].append(1.+beta)
        fields["solve_ms"].append((time.perf_counter()-tic)*1000.)
        command = command_t[0].numpy()
        fields["time_control_s"].append(k*.1)
        fields["command"].append(command.copy())
        fields["qp_ok"].append(bool(ok[0]))
        fields["qp_primal"].append(float(mpc.last_primal_residual[0]))
        fields["qp_dual"].append(float(mpc.last_dual_residual[0]))
        fields["a_depth"].append(scales[0])
        fields["a_rudder_rate"].append(scales[1])
        for j in range(5):
            plant.step(command)
            now = (5*k+j+1)*.02
            out = plant.output()
            ref = ref_grid[5*k+j+1]
            diag = harness.diagnostic_values(out, ref)
            fields["time_substep_s"].append(now)
            fields["state_substep"].append(plant.x.copy())
            fields["output_substep"].append(out)
            fields["reference_substep"].append(ref.copy())
            fields["diagnostic_substep"].append(diag)
            fields["actual_substep"].append(plant.actual.copy())
            fields["command_substep"].append(command.copy())
            exceeded = np.abs(diag) > [8., 5., 2., np.deg2rad(40.)]
            nonfinite = not np.isfinite(plant.x).all()
            if self.first_violation is None and (np.any(exceeded) or nonfinite):
                names = ["lateral_8m", "depth_5m", "tangent_speed_2mps", "pitch_40deg"]
                self.first_violation = {"time_s": now,
                    "kinds": [n for n, b in zip(names, exceeded) if b] +
                             (["nonfinite"] if nonfinite else []),
                    "diagnostic_values": diag.tolist()}
            if nonfinite or self.first_violation is not None:
                break
        self.k += 1
        return {"qp_ok": bool(ok[0]), "first_violation": self.first_violation,
                "time_s": float(fields["time_substep_s"][-1]), "done": self.done}

    def result(self):
        """Return the original run_one result schema for completed or partial runs."""
        if not self.fields["command"]:
            raise RuntimeError("no control cycle has run")
        arrays = {k: np.asarray(v) for k, v in self.fields.items()}
        command_sat = np.mean((arrays["command"]/SCALE >= 1-1e-6) |
                              (arrays["command"]/SCALE <= np.array([0., -1., -1.])+1e-6), axis=0)
        actual_norm = arrays["actual_substep"]/SCALE
        actual_sat = np.mean((actual_norm >= 1-1e-6) |
                             (actual_norm <= np.array([0., -1., -1.])+1e-6), axis=0)
        status = "early_violation" if self.first_violation else (
                 "qp_failure" if np.any(~arrays["qp_ok"]) else (
                 "complete_constrained" if np.any(command_sat) or np.any(actual_sat) else "complete_valid"))
        meta = {"status": status, "duration_observed_s": float(arrays["time_substep_s"][-1]),
                "first_violation": self.first_violation,
                "qp_failures": int(np.sum(~arrays["qp_ok"])),
                "command_saturation_fraction": command_sat.tolist(),
                "actual_saturation_fraction": actual_sat.tolist(),
                "solve_p50_ms": float(np.percentile(arrays["solve_ms"], 50)),
                "solve_p95_ms": float(np.percentile(arrays["solve_ms"], 95)),
                "solve_p99_ms": float(np.percentile(arrays["solve_ms"], 99)),
                "wall_s": time.perf_counter()-self.started}
        return {"meta": meta, "arrays": arrays}


def run_one(task, controller="C4", a_depth=1., a_rudder_rate=1., duration_s=64.,
            reference_fn=None, predictor_params=None, beta_policy=None,
            preview_s=0.):
    """Run one serial causal trajectory; weight scales update before each QP."""
    session = ClosedLoopSession(
        task, controller, a_depth, a_rudder_rate, duration_s, reference_fn,
        predictor_params, dynamic_predictor=beta_policy is not None,
        preview_s=preview_s)
    while not session.done:
        beta = None if beta_policy is None else float(
            beta_policy(session.k*.1, session.ref_grid, task))
        session.advance_control_cycle(beta)
    return session.result()
