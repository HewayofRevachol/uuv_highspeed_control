#!/usr/bin/env python3
"""Required route gates: reference derivatives, QP wiring, and old-loop parity."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
os.environ["GS12_REPO"] = str(ROOT)

from reference_tasks import mixed_reference, single_frequency
from runner_core import run_one, SCALE
from experiments.gs12_analaysis.frequency_extension_src.plant_harness import PlantConfig, Plant, run_closed_loop
from simulator.batch_mpc import BatchedUUVModel
from simulator.rti_symbolic import ActuatorAwareCpuDynamics
from aav_mpc.condensed_mpc import CondensedActuatorMPC
import osqp


def derivative_check():
    t = np.linspace(1., 63., 20001)
    dt = t[1]-t[0]
    worst = 0.
    for fn in (lambda x: single_frequency(x, 8, "depth", .4),
               lambda x: single_frequency(x, 32, "east", -.7),
               lambda x: mixed_reference(x, .3)):
        r = fn(t)
        deriv = np.gradient(r[:, :3], dt, axis=0, edge_order=2)
        worst = max(worst, float(np.max(np.abs(deriv-r[:, 5:8]))))
    if worst > .03:
        raise AssertionError(f"reference derivative mismatch {worst}")
    return worst


def qp_check():
    model = BatchedUUVModel("cpu")
    mpc = CondensedActuatorMPC(model, 1, horizon=35, dt=.1)
    plant = Plant(15., PlantConfig(elevator=.9, rudder=.9, order="second_all"))
    mpc.previous[:] = model.tensor(plant.actual/SCALE); mpc.plan[:] = mpc.previous[:, None]
    x = plant.controller_state()
    ref = single_frequency(np.arange(1, 36)*.1, 8, "depth")
    original = osqp.OSQP.setup
    observed = []
    def capture(self, *args, **kwargs):
        P = kwargs.get("P", args[0] if args else None)
        q = kwargs.get("q", args[1] if len(args)>1 else None)
        observed.append((P.toarray().copy(), np.asarray(q).copy()))
        return original(self, *args, **kwargs)
    outputs = []
    with patch.object(osqp.OSQP, "setup", capture):
        for scales in ((1., 1.), (1.4, 1.), (1., 1.4)):
            mpc.set_tracking_weight_scales(*scales)
            plan = mpc.plan[0].numpy().copy()
            shifted = np.concatenate((plan[1:], plan[-1:]), axis=0)
            mpc._one(x, ref, shifted, mpc.previous[0].numpy(),
                     np.array([1., 1.]), np.zeros((35, 5)))
            outputs.append(observed[-1])
    p0, q0 = outputs[0]; pd, qd = outputs[1]; pr, qr = outputs[2]
    if not np.any(np.abs(pd-p0)>1e-10) or not np.any(np.abs(qd-q0)>1e-10):
        raise AssertionError("depth multiplier did not change QP objective")
    if not np.any(np.abs(pr-p0)>1e-10):
        raise AssertionError("fin-rate multiplier did not change QP objective")
    return {"depth_hessian_delta": float(np.max(np.abs(pd-p0))),
            "depth_gradient_delta": float(np.max(np.abs(qd-q0))),
            "rate_hessian_delta": float(np.max(np.abs(pr-p0))),
            "rate_gradient_delta": float(np.max(np.abs(qr-q0)))}


def parity_check():
    task = {"task_id": "parity", "kind": "single", "axis": "depth", "k": 8,
            "phase_rad": .3, "amplitude": .006, "duration_s": 24., "speed_mps": 15.}
    new = run_one(task, "C4", 1., 1., 24.)
    old = run_closed_loop(PlantConfig(elevator=.9, rudder=.9, order="second_all"), 15.,
                          lambda t: single_frequency(t, 8, "depth", .3), 24., 1., 1., None, True)
    na, oa = new["arrays"], old["arrays"]
    comparisons = {
        "command": float(np.max(np.abs(na["command"]-oa["command"]))),
        "hull_state": float(np.max(np.abs(na["state_substep"]-oa["state"]))),
        "output": float(np.max(np.abs(na["output_substep"]-oa["output"]))),
        "diagnostic": float(np.max(np.abs(na["diagnostic_substep"]-oa["diagnostic"]))),
        "qp_valid_mismatch_count": int(np.count_nonzero(na["qp_ok"] != oa["qp_ok"])),
    }
    if max(comparisons[k] for k in ("command", "hull_state", "output", "diagnostic")) > 2e-7 or comparisons["qp_valid_mismatch_count"]:
        raise AssertionError(f"old/new runner parity failed: {comparisons}")
    return comparisons


if __name__ == "__main__":
    result = {"reference_derivative_max_abs": derivative_check(),
              "qp_wiring": qp_check(), "old_runner_parity_max_abs": parity_check()}
    print(result)
