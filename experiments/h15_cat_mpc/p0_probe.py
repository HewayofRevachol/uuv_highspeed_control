"""Short mechanism probe for the P0 CAT-MPC model interface."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from aav_mpc.controller import EndpointExtendedCondensedEnv
from aav_mpc.h15_target_domain import install_target_domain
from simulator.batch_path_env import PathBank
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "simple": {"path": "data/datasets/h15_mainline_quickstart/train_late_entry.csv",
               "seed": 31001, "target": False, "current": 0., "noise": False},
    "coupled": {"path": "data/datasets/h15_mainline_quickstart/train_turn_wide.csv",
                 "seed": 31002, "target": False, "current": 0., "noise": False},
    "mismatch": {"path": "data/datasets/h15_mainline_quickstart/train_coupled_shallow.csv",
                 "seed": 31003, "target": True, "current": 1., "noise": True},
}


def run_one(case, gated, cycles):
    random.seed(case["seed"])
    np.random.seed(case["seed"])
    torch.manual_seed(case["seed"])
    bank = PathBank([ROOT / case["path"]], "cpu")
    bank.speed[:] = 15.
    bank.first_state[:, 5] = 15.
    bank.durations[:] = bank.lengths / 15.
    env = EndpointExtendedCondensedEnv(
        bank, batch_size=1, device="cpu", horizon=35, backend="rti_cpu",
        seed=case["seed"], randomize_plant=case["target"],
        current_max_mps=case["current"],
        current_profile="harmonic" if case["current"] else "steady",
        admm_iterations=400, admm_rho=10., attitude_pitch_weight=100.,
        attitude_yaw_weight=30., fsp_enabled=False,
        preview_mode="multiscale", yaw_reward="course")
    env.mpc.dynamics = ActuatorAwareCpuDynamics(.1, .85, 2.)
    if not case["noise"]:
        env.disturbance_std.zero_()
    alpha = np.array([1., 1.]) if gated else np.zeros(2)
    theta = np.array([.90, 1.20]) if gated else np.array([.85, 2.])
    # Deliberately small nonzero bias exercises the numerical d[5] path only;
    # it is a wiring probe, not a trained or evaluated residual forecast.
    d = np.tile(np.array([.002, -.001, 0., .0002, -.0002]), (35, 1)) if gated else np.zeros((35, 5))
    env.mpc.set_continuous_model(alpha, theta, d)
    env.reset(torch.zeros(1, dtype=torch.long))
    env.capture_strict_substeps = True
    rows = []
    for step in range(cycles):
        command = env.base_command[0].detach().cpu().numpy().copy()
        state = env.state[0].detach().cpu().numpy().copy()
        margins = np.array([8., 5., 40., 2.]) - np.array([
            abs(float(env._errors()[1][0])), abs(float(env._errors()[2][0])),
            abs(float(np.rad2deg(state[3]))), abs(float(env._errors()[5][0]))])
        started = time.perf_counter()
        _, _, done, info = env.step(torch.zeros(1, 3))
        period_ms = (time.perf_counter()-started)*1000.
        substeps = []
        for sample in env.strict_substeps:
            lateral = abs(float(sample["lateral_error_m"][0]))
            depth = abs(float(sample["depth_error_m"][0]))
            speed_error = abs(float(sample["speed_error_mps"][0]))
            pitch_deg = abs(float(np.rad2deg(sample["pitch_rad"][0])))
            substeps.append({"substep": sample["substep"],
                             "physical_flags": {
                                 "lateral": lateral > 8., "depth": depth > 5.,
                                 "speed": speed_error > 2., "pitch": pitch_deg > 40.}})
        rows.append({"step": step, "alpha_theta": float(alpha[0]),
                     "alpha_d": float(alpha[1]), "command": command.tolist(),
                     "state": state.tolist(), "margins": margins.tolist(),
                     "solve_period_ms": period_ms,
                     "done": bool(done[0]), "backup": False,
                     "deadline_miss": period_ms > 100.,
                     "substeps": substeps, "qp_ok": bool(env.last_mpc_ok[0])})
        if bool(done[0]):
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=40)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 40:
        parser.error("cycles must be in [1, 40]")
    args.out.mkdir(parents=True, exist_ok=False)
    results = {"schema": 1, "purpose": "P0 interface probe; not a performance evaluation",
               "cases": {}}
    for name, case in CASES.items():
        base = run_one(case, False, args.cycles)
        gated = run_one(case, True, args.cycles)
        count = min(len(base), len(gated))
        cmd_delta = [float(np.linalg.norm(np.asarray(gated[i]["command"])-
                                           np.asarray(base[i]["command"])))
                     for i in range(count)]
        state_delta = [float(np.max(np.abs(np.asarray(gated[i]["state"])-
                                             np.asarray(base[i]["state"]))))
                       for i in range(count)]
        results["cases"][name] = {
            "path": case["path"], "seed": case["seed"],
            "base_cycles": len(base), "gated_cycles": len(gated),
            "command_delta_nonzero_cycles": sum(value > 1e-9 for value in cmd_delta),
            "max_command_delta_physical_units": max(cmd_delta, default=0.),
            "max_state_delta": max(state_delta, default=0.),
            "base": base, "gated": gated,
        }
    raw = json.dumps(results, indent=2) + "\n"
    (args.out / "p0_probe.json").write_text(raw)
    manifest = {"config_sha256": hashlib.sha256(
                    (ROOT / "experiments/h15_cat_mpc/config.json").read_bytes()).hexdigest(),
                "probe_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "cycles_requested": args.cycles,
                "scope": "mechanism check only; no success-rate claim"}
    (args.out / "run.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k not in ("base", "gated")}
                      for key, value in results["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
