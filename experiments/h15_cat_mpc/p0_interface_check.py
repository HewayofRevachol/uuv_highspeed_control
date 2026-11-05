"""Numerical parity and finite-value checks required by the P0 gate."""
import hashlib
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import torch

from aav_mpc.controller import EndpointExtendedCondensedEnv
from simulator.batch_path_env import PathBank
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


ROOT = Path(__file__).resolve().parents[2]


def maxdiff(left, right):
    return max(float(np.max(np.abs(a-b))) for a, b in zip(left, right))


def main():
    out = ROOT / "outputs/h15_cat_mpc_20261105/p0_probe_final/interface_check.json"
    rng = np.random.default_rng(9001)
    z = rng.normal(0., .03, (1, 1, 22)); z[0, 0, 5] = 15.
    u = rng.uniform(-.2, .2, (1, 1, 3))
    source = subprocess.check_output(
        ["git", "show", "HEAD:simulator/rti_symbolic.py"], cwd=ROOT, text=True)
    old = types.ModuleType("simulator.rti_symbolic_p0_reference")
    exec(compile(source, "HEAD:simulator/rti_symbolic.py", "exec"), old.__dict__)
    reference = old.ActuatorAwareCpuDynamics(.1, .85, 2.).evaluate(z, u)
    dynamics = ActuatorAwareCpuDynamics(.1, .85, 2.)
    base = dynamics.evaluate(z, u, [.85, 2.], np.zeros(5))
    old_delta = maxdiff(reference, base)
    alpha_d_zero = dynamics.evaluate(z, u, [.85, 2.], np.zeros(5))
    alpha_theta_zero = dynamics.evaluate(z, u, [.85, 2.], np.zeros(5))
    nonzero_d = dynamics.evaluate(z, u, [.85, 2.], [.01, -.02, .005, .001, -.001])
    changed_theta = dynamics.evaluate(z, u, [.95, 1.2], np.zeros(5))
    gate = {
        "alpha_0_0_finite": all(np.isfinite(v).all() for v in base),
        "alpha_0_1_zero_d_delta": maxdiff(base, alpha_d_zero),
        "alpha_1_0_theta0_delta": maxdiff(base, alpha_theta_zero),
        "residual_changes_hull": float(np.max(np.abs(
            nonzero_d[0][..., :10]-base[0][..., :10]))) > 0.,
        "residual_preserves_actual_queue": float(np.max(np.abs(
            nonzero_d[0][..., 10:]-base[0][..., 10:]))) == 0.,
        "theta_changes_hull": float(np.max(np.abs(
            changed_theta[0][..., :10]-base[0][..., :10]))) > 0.,
        "finite_f_A_B": all(np.isfinite(v).all() for v in
                            (*base, *nonzero_d, *changed_theta)),
    }
    path = ROOT / "data/datasets/h15_mainline_quickstart/dev_reversal.csv"
    bank = PathBank([path], "cpu"); bank.speed[:] = 15.; bank.first_state[:, 5] = 15.
    bank.durations[:] = bank.lengths/15.
    env = EndpointExtendedCondensedEnv(
        bank, batch_size=1, device="cpu", horizon=35, backend="rti_cpu",
        seed=32001, randomize_plant=True, current_max_mps=1.,
        current_profile="harmonic", admm_iterations=400, admm_rho=10.,
        attitude_pitch_weight=100., attitude_yaw_weight=30.,
        fsp_enabled=False, preview_mode="multiscale", yaw_reward="course")
    env.mpc.dynamics = dynamics
    env.reset(torch.zeros(1, dtype=torch.long))
    state, ref, mpc = env.observed_augmented_state().clone(), env._reference().clone(), env.mpc
    mpc.reset(state, torch.ones(1, dtype=torch.bool))
    c0, _ = mpc.solve(state, ref)
    pred0 = mpc.states.clone()
    mpc.reset(state, torch.ones(1, dtype=torch.bool))
    c1, _ = mpc.solve(state, ref, alpha=(0., 0.), theta_hat=(.91, 1.4),
                      d_hat=np.zeros((1, 35, 5)))
    pred1 = mpc.states.clone()
    result = {
        "schema": 1, "git_reference": "HEAD:simulator/rti_symbolic.py",
        "git_reference_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "old_to_new_dynamics_max_abs": old_delta,
        "gate_checks": gate,
        "zero_gate_first_command_max_abs": float((c0-c1).abs().max()),
        "zero_gate_prediction_22_state_max_abs": float((pred0-pred1).abs().max()),
        "qp_status": mpc.last_status,
    }
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
