"""GS-12: controlled finite-record analogue of Badwe et al. Eqs. (20)-(21).

Run with --repo pointing to uuv_highspeed_control at the recorded commit.
Uses its real condensed MPC; supplies an exogenous, shared time reference.
This is a diagnostic harness, not the deployed progress-based H15 runner.
No RL, residual compensation, current, noise, or randomized parameters.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def reference(t, speed, amplitude, omega):
    t = np.asarray(t)
    east = amplitude * (1 - np.cos(omega * t))
    depth = 20 + .4 * amplitude * (1 - np.cos(1.2 * omega * t))
    ve = amplitude * omega * np.sin(omega * t)
    vd = .48 * amplitude * omega * np.sin(1.2 * omega * t)
    pitch = -np.arctan2(vd, np.sqrt(speed**2 + ve**2))
    yaw = np.arctan2(ve, speed)
    return np.stack((speed*t, east, depth, pitch, yaw,
                     np.full_like(t, speed), ve, vd), axis=-1)


def run_case(name, model_gain, model_tau, plant_gain, plant_tau, second_order,
             amplitude, omega, duration, speed):
    from simulator.batch_mpc import BatchedUUVModel, CONTROL_SCALE
    from simulator.model import D_COEFF, MISMATCHED_PLANT as p
    from simulator.rti_symbolic import ActuatorAwareCpuDynamics
    from aav_mpc.condensed_mpc import CondensedActuatorMPC
    from aav_mpc.h15_target_domain import SecondOrderActuator

    model = BatchedUUVModel("cpu")
    mpc = CondensedActuatorMPC(model, 1, horizon=35, dt=.1)
    mpc.dynamics = ActuatorAwareCpuDynamics(.1, model_gain, model_tau)
    x = model.tensor([[0., 0., 20., 0., 0., speed, 0., 0., 0., 0.]])
    trim = p.damping_scale * D_COEFF[0, 0] * speed**2 / p.thrust_gain
    initial = model.tensor([[trim, 0., 0.]])
    plant = SimpleNamespace(actuator=initial.clone(),
        tau=model.tensor([[p.thrust_tau_s, p.fin_tau_s*plant_tau,
                           p.fin_tau_s*plant_tau]]))
    updater = SecondOrderActuator(plant) if second_order else None
    queue = initial[:, None].repeat(1, 3, 1)
    parameters = model.tensor([[p.mass_scale, p.damping_scale, p.thrust_gain,
        p.elevator_gain*plant_gain, p.rudder_gain*plant_gain]])
    mpc.previous[:] = initial/model.control_scale
    mpc.plan[:] = mpc.previous[:, None]
    running = torch.ones(1, dtype=torch.bool)
    records, first_violation, max_parity = [], None, 0.
    started = time.perf_counter()
    for k in range(round(duration/.1)):
        z = torch.cat((x, plant.actuator/model.control_scale,
                       queue.reshape(1, 9)/model.control_scale.repeat(3)), -1)
        refs = reference((k+np.arange(1, 36))*.1, speed, amplitude, omega)
        command, ok = mpc.solve(z, model.tensor(refs)[None])
        predicted = mpc.dynamics.evaluate(z.numpy()[:, None],
            (command/model.control_scale).numpy()[:, None])[0][0, 0]
        for j in range(5):
            target = queue[:, 2].clone()
            queue = torch.cat((command[:, None], queue[:, :-1]), 1)
            if updater is None:
                plant.actuator += -torch.expm1(-.02/plant.tau)*(target-plant.actuator)
            else:
                plant.actuator = updater(target, .02, running)
            x = model.rk4(x, plant.actuator, parameters=parameters, dt=.02)
            now = k*.1+(j+1)*.02
            ref = reference(now, speed, amplitude, omega)
            velocity = model.derivative(x, plant.actuator, parameters=parameters)[0, :3].numpy()
            tangent = ref[5:]/np.linalg.norm(ref[5:])
            position_error = x[0, :3].numpy()-ref[:3]
            side = np.array([-np.sin(ref[4]), np.cos(ref[4]), 0.])
            violation = {"lateral": abs(float(side@position_error)) > 8.,
                "depth": abs(float(position_error[2])) > 5.,
                "tangent_speed": abs(float(tangent@velocity)-np.linalg.norm(ref[5:])) > 2.,
                "pitch": abs(float(x[0, 3])) > np.deg2rad(40)}
            if any(violation.values()) and first_violation is None:
                first_violation = {"time_s": now,
                    "kinds": [key for key, value in violation.items() if value]}
        actual_z = torch.cat((x, plant.actuator/model.control_scale,
                            queue.reshape(1, 9)/model.control_scale.repeat(3)), -1)
        parity = float(np.max(np.abs(actual_z.numpy()[0]-predicted)))
        max_parity = max(max_parity, parity)
        output = np.r_[x[0, :5].numpy(), velocity]
        error = ref-output
        error[3:5] = np.arctan2(np.sin(error[3:5]), np.cos(error[3:5]))
        records.append({"time_s": now, "reference": ref.tolist(),
            "state": x[0].tolist(), "output": output.tolist(),
            "error": error.tolist(), "command": command[0].tolist(),
            "actuator": plant.actuator[0].tolist(), "qp_ok": bool(ok[0]),
            "one_step_max_abs_state_error": parity})
        if not torch.isfinite(x).all() or first_violation is not None:
            break
    normalized_commands = np.asarray([r["command"] for r in records])/CONTROL_SCALE
    normalized_actuators = np.asarray([r["actuator"] for r in records])/CONTROL_SCALE
    return {"name": name, "model_gain": model_gain, "model_tau_scale": model_tau,
        "plant_gain": plant_gain, "plant_tau_scale": plant_tau,
        "second_order_all_channels": second_order,
        "records": records, "periods": len(records),
        "first_diagnostic_limit_violation": first_violation,
        "qp_failures": sum(not r["qp_ok"] for r in records),
        "command_saturation_fraction_by_channel": np.mean(
            (normalized_commands >= 1-1e-6) |
            (normalized_commands <= np.array([0., -1., -1.])+1e-6), axis=0).tolist(),
        "actual_saturation_fraction_by_channel": np.mean(
            (normalized_actuators >= 1-1e-6) |
            (normalized_actuators <= np.array([0., -1., -1.])+1e-6), axis=0).tolist(),
        "matched_one_step_max_abs_state_error": max_parity,
        "wall_s_not_a_realtime_benchmark": time.perf_counter()-started}


def summarize(cases):
    result = {}
    for name, case in cases.items():
        design_name = "conservative_design" if name.startswith("conservative") else "design"
        design = cases[design_name]
        n = min(len(case["records"]), len(design["records"]))
        ea = np.asarray([r["error"] for r in case["records"][:n]])
        ed = np.asarray([r["error"] for r in design["records"][:n]])
        refs_a = [r["reference"] for r in case["records"][:n]]
        refs_d = [r["reference"] for r in design["records"][:n]]
        if refs_a != refs_d:
            raise AssertionError("paired references must be identical")
        va, vd = np.var(ea, axis=0), np.var(ed, axis=0)
        ratio = [float(a/d) if d > 1e-12 else None for a, d in zip(va, vd)]
        result[name] = {key: case[key] for key in ("periods", "first_diagnostic_limit_violation",
            "qp_failures", "command_saturation_fraction_by_channel",
            "actual_saturation_fraction_by_channel", "matched_one_step_max_abs_state_error")}
        result[name].update({"design_pair": design_name, "paired_periods": n,
            "window_start_s": .1, "window_end_s": n*.1,
            "VR": ratio, "variance": va.tolist(), "design_variance": vd.tolist(),
            "RMSE": np.sqrt(np.mean(ea**2, axis=0)).tolist(),
            "mean_error": ea.mean(axis=0).tolist(),
            "max_abs_error": np.abs(ea).max(axis=0).tolist()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=24.)
    parser.add_argument("--speed", type=float, default=15.)
    args = parser.parse_args()
    if args.duration <= 0 or abs(args.duration/.1-round(args.duration/.1)) > 1e-9:
        parser.error("duration must be a positive integer multiple of 0.1 seconds")
    sys.path.insert(0, str(args.repo.resolve()))
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    sources = ["aav_mpc/condensed_mpc.py", "simulator/rti_symbolic.py",
               "simulator/batch_mpc.py", "aav_mpc/h15_target_domain.py"]
    protocol = {"purpose": "controlled nonlinear finite-record VR, not RSI or an H15 acceptance run",
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "source_sha256": {p: hashlib.sha256((args.repo/p).read_bytes()).hexdigest() for p in sources},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "torch", "scipy", "osqp", "casadi"]},
        "period_s": .1, "plant_substep_s": .02, "horizon": 35,
        "requested_duration_s": args.duration, "north_speed_mps": args.speed,
        "reference_speed_note": "north velocity fixed; tangent speed follows the analytic reference",
        "current": 0, "noise": 0, "randomization": False, "integrator_all_cases": "RK4",
        "delay_substeps_all_channels": 3, "low_level_compensation": "off",
        "initialization": "common straight trim, queue/actuator/plan prefilled with trim",
        "output_order": ["north_m", "east_m", "depth_m", "pitch_rad", "yaw_rad", "VN_mps", "VE_mps", "VD_mps"],
        "VR_window": "population centered variance on matched common time prefix, no burn-in; denominator <=1e-12 => null",
        "not_included": ["RL", "FSP", "residual predictor", "progress feedback", "target midpoint integrator"]}
    (args.out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    configurations = [("design", 1., 1., 1., 1., False),
        ("fin_0p9", 1., 1., .9, 1., False),
        ("second_order", 1., 1., 1., 1., True),
        ("combined", 1., 1., .9, 1., True),
        ("conservative_design", .85, 2., .85, 2., False),
        ("conservative_combined", .85, 2., .9, 1., True)]
    summary = {}
    for geometry, amplitude, omega in [("gentle", 2., .4), ("coupled", 6., .7)]:
        cases = {}
        for configuration in configurations:
            case = run_case(*configuration, amplitude, omega, args.duration, args.speed)
            name = configuration[0]
            cases[name] = case
            (args.out/f"{geometry}_{name}.json").write_text(json.dumps(case, indent=2)+"\n")
            print(json.dumps({"geometry": geometry, "case": name,
                "periods": case["periods"], "violation": case["first_diagnostic_limit_violation"]}), flush=True)
        for name in ("design", "conservative_design"):
            if cases[name]["matched_one_step_max_abs_state_error"] > 1e-9:
                raise AssertionError("designed plant must match actual MPC transition")
        summary[geometry] = {"amplitude_m": amplitude, "omega_radps": omega,
                             "metrics": summarize(cases)}
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
