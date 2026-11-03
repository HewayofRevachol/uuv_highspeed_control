"""Paired failure trace for the frozen T1/T3 tasks.

Run from the repository root with /bin/python3. Each row is one control
period; the paired cases share path, plant, seed, and disturbance settings.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from high_speed_experiments.reference_schedule_probe import ScheduledReferenceEnv
from aav_mpc.controller import (ActuatorAwareEnv, EndpointExtendedEnv,
                                EndpointExtendedAwareEnv, CondensedActuatorEnv,
                                EndpointExtendedCondensedEnv)
from simulator.batch_path_env import PathBank
from simulator.model import LoopPath

CONFIG = Path(__file__).parent / "configs/tasks.json"


def next_event(path, progress):
    s = path.distances
    curvature = np.gradient(path.yaw, s)
    change = np.abs(np.gradient(curvature, s))
    events = s[(change > 0.00015) | (np.abs(np.gradient(path.pitch, s)) > .0005)]
    later = events[events >= progress]
    return float(later[0] - progress) if len(later) else math.nan


def run(task, method, out, max_steps=None, randomize_plant=False,
        current_max=0., process_noise=False, nominal_body=False,
        path_override=None, speed_override=None, seed_override=None,
        speed_conditioned_cost=False, oracle_no_lag=False):
    definition = dict(json.loads(CONFIG.read_text())[task])
    if path_override is not None:
        definition["path"] = str(path_override)
    if speed_override is not None:
        definition["target_speed_mps"] = float(speed_override)
    if seed_override is not None:
        definition["seed"] = int(seed_override)
    file = ROOT / definition["path"]
    bank = PathBank([file], "cpu")
    speed = definition["target_speed_mps"]
    cost_scale = 12. / speed if speed_conditioned_cost else 1.
    bank.speed[:] = speed
    bank.first_state[:, 5] = speed
    bank.durations[:] = bank.lengths / speed
    opts = ({"preview_scale": 1.5} if method == "preview15" else
            {"curvature_accel_cap": 6.} if method == "curv6" else
            {"speed_cap": 15.} if method == "cap15" else {})
    backend = "rti_cpu" if method.startswith(("rti", "aware", "condensed")) else "osqp"
    horizon = 40 if method.endswith("40") else 25
    env_type = (EndpointExtendedCondensedEnv if method.startswith("condensed_endpoint") else
                CondensedActuatorEnv if method.startswith("condensed") else
                EndpointExtendedAwareEnv if method.startswith("aware_endpoint") else
                EndpointExtendedEnv if method.startswith("endpoint") else
                ActuatorAwareEnv if method.startswith("aware") else ScheduledReferenceEnv)
    env = env_type(bank, batch_size=1, device="cpu",
        horizon=horizon, backend=backend, seed=definition["seed"],
        randomize_plant=randomize_plant, current_max_mps=current_max,
        admm_iterations=400, admm_rho=10.,
        attitude_pitch_weight=100.*cost_scale, attitude_yaw_weight=30.*cost_scale,
        fsp_enabled=not method.startswith(("aware", "condensed")), preview_mode="multiscale", yaw_reward="course", **opts)
    env.reset(torch.zeros(1, dtype=torch.long))
    if not process_noise:
        env.disturbance_std.zero_()
    if nominal_body:
        env.parameters[:] = 1.
    if oracle_no_lag:
        env.delay_steps.zero_()
        env.tau.zero_()
        env.fsp = None
        env.mpc.reset(env.state, torch.ones(1, dtype=torch.bool))
        env.base_command, env.last_mpc_ok = env._solve_mpc()
    path = LoopPath(file)
    rows = []
    wall = time.perf_counter()
    while True:
        before = env.state[0].clone()
        actuator_before = env.actuator[0].clone()
        current_estimate_before = env.current_estimate[0].clone()
        command = env.base_command[0].clone()
        queue = env.command_history[0].clone()
        progress_before = float(env.progress[0])
        solve_ok = bool(env.last_mpc_ok[0])
        solve_status = getattr(env.mpc, "last_status", ["unknown"])[0]
        predicted = env.mpc.states[0, 1].clone() if hasattr(env.mpc, "states") else None
        tic = time.perf_counter()
        _, _, done, info = env.step(torch.zeros(1, 3))
        issued = info["command"][0].clone()
        solve_ms = (time.perf_counter() - tic) * 1000
        _, lateral, depth, _, _, speed_error, _ = env._errors()
        x = env.state[0]
        row = {"step": len(rows), "time_s": float(env.steps[0]) * .1,
            "progress_before_m": progress_before,
            "state_before_json": json.dumps(before.tolist()),
            "actuator_before_json": json.dumps(actuator_before.tolist()),
            "current_estimate_before_json": json.dumps(current_estimate_before.tolist()),
            "queue_before_json": json.dumps(queue.tolist()),
            "command_json": json.dumps(issued.tolist()),
            "mpc_base_command_json": json.dumps(command.tolist()),
            "state_after_json": json.dumps(x.tolist()),
            "current_ned_json": json.dumps(env.current_ned[0].tolist()),
            "progress_m": float(env.progress[0]),
            "distance_to_geometry_change_m": next_event(path, progress_before),
            "target_speed_mps": speed, "body_speed_mps": float(torch.linalg.vector_norm(x[5:8])),
            "speed_error_mps": float(speed_error[0]),
            "lateral_error_m": float(lateral[0]), "depth_error_m": float(depth[0]),
            "pitch_deg": math.degrees(float(x[3])), "pitch_rate_degps": math.degrees(float(x[8])),
            "yaw_rate_degps": math.degrees(float(x[9])),
            "thrust_command_n": float(issued[0]), "elevator_command_deg": math.degrees(float(issued[1])),
            "rudder_command_deg": math.degrees(float(issued[2])),
            "thrust_actual_n": float(env.actuator[0, 0]),
            "elevator_actual_deg": math.degrees(float(env.actuator[0, 1])),
            "rudder_actual_deg": math.degrees(float(env.actuator[0, 2])),
            "queue_thrust_n": json.dumps(queue[:, 0].tolist()),
            "queue_rudder_deg": json.dumps(np.rad2deg(queue[:, 2].numpy()).tolist()),
            "mpc_ok": solve_ok, "mpc_next_ok": bool(info["mpc_ok"][0]),
            "mpc_status": solve_status,
            "mpc_next_status": getattr(env.mpc, "last_status", ["unknown"])[0],
            "qp_valid": bool(getattr(env.mpc, "last_qp_valid", env.last_mpc_ok)[0]),
            "primal_residual": float(getattr(env.mpc, "last_primal_residual", torch.zeros(1))[0]),
            "dual_residual": float(getattr(env.mpc, "last_dual_residual", torch.zeros(1))[0]),
            "control_period_wall_ms": solve_ms,
            "success": bool(info["success"][0]), "failed": bool(info["failed"][0]),
            "timeout": bool(info["timeout"][0])}
        if predicted is not None:
            row["prediction_1_position_error_m"] = float(torch.linalg.vector_norm(predicted[:3] - x[:3]))
            row["prediction_1_velocity_error_mps"] = float(torch.linalg.vector_norm(predicted[5:8] - x[5:8]))
        rows.append(row)
        if bool(done[0]) or (max_steps and len(rows) >= max_steps):
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    limits = {"speed": definition["speed_tolerance_mps"],
        "lateral": definition["lateral_corridor_m"], "depth": definition["depth_corridor_m"],
        "pitch": definition["pitch_limit_deg"]}
    first = []
    for row in rows:
        for key, magnitude in (("solver", not row["mpc_ok"]),
                 ("speed", abs(row["speed_error_mps"]) > limits["speed"]),
                 ("lateral", abs(row["lateral_error_m"]) > limits["lateral"]),
                 ("depth", abs(row["depth_error_m"]) > limits["depth"]),
                 ("pitch", abs(row["pitch_deg"]) > limits["pitch"])):
            if magnitude:
                first.append((row["time_s"], key))
    summary = {"task": task, "method": method, "path": definition["path"],
        "target_speed_mps": speed,
        "seed": definition["seed"], "backend": backend, "horizon": horizon,
        "randomize_plant": randomize_plant, "current_max_mps": current_max,
        "process_noise": process_noise, "nominal_body": nominal_body,
        "speed_conditioned_cost": speed_conditioned_cost,
        "oracle_no_lag": oracle_no_lag,
        "steps": len(rows), "wall_s": time.perf_counter() - wall,
        "progress_fraction": float(env.progress[0] / bank.lengths[0]),
        "simulator_success": rows[-1]["success"], "simulator_failed": rows[-1]["failed"],
        "strict_success": rows[-1]["success"] and not first,
        "first_anomaly": min(first) if first else None,
        "fallback_count": sum(not r["mpc_ok"] for r in rows),
        "saturation_fraction": float(np.mean([r["thrust_command_n"] > 1499999 or
            abs(r["rudder_command_deg"]) > 44.99 or abs(r["elevator_command_deg"]) > 44.99 for r in rows])),
        "trace": str(out.resolve().relative_to(ROOT))}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("T1", "T3"), required=True)
    parser.add_argument("--method", choices=("osqp25", "osqp40", "rti25", "rti40",
        "preview15", "aware25", "aware40", "endpoint40", "aware_endpoint40",
        "condensed25", "condensed40", "condensed_endpoint25", "condensed_endpoint40",
        "curv6", "cap15"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--randomize-plant", action="store_true")
    parser.add_argument("--current-max", type=float, default=0.)
    parser.add_argument("--process-noise", action="store_true")
    parser.add_argument("--nominal-body", action="store_true")
    parser.add_argument("--path", type=Path)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--speed-conditioned-cost", action="store_true")
    parser.add_argument("--oracle-no-lag", action="store_true")
    args = parser.parse_args()
    run(args.task, args.method, args.out, args.max_steps, args.randomize_plant,
        args.current_max, args.process_noise, args.nominal_body,
        args.path, args.speed, args.seed, args.speed_conditioned_cost,
        args.oracle_no_lag)
