"""Replay a known feasible open-loop plan in the full stochastic H15 target.

Uses the same target environment, plant seed, harmonic current and process
noise as the paired MPC run. No MPC solve occurs after reset; commands are
scheduled directly. Success proves existence for that realized simulation,
while failure of a fixed plan does not prove infeasibility.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from aav_mpc.controller import EndpointExtendedCondensedEnv
from aav_mpc.h15_target_domain import install_target_domain
from simulator.batch_path_env import PathBank


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(offline_file, seed, out, paired_target_trace=None):
    offline = json.loads(offline_file.read_text())
    if (offline.get("sampled_plant_seed") != seed or
            not offline.get("target_deterministic") or
            not any(row["found_full_path_feasible"] for row in offline["results"])):
        raise ValueError("offline result is not a verified plan for this target plant seed")
    start = next(row["start"] for row in offline["results"]
                 if row["found_full_path_feasible"])
    command_file = offline_file.with_name(offline_file.stem + f"_start{start}_commands.npy")
    commands = np.load(command_file, allow_pickle=False)
    if commands.ndim != 2 or commands.shape[1] != 3 or not np.isfinite(commands).all():
        raise ValueError("invalid offline command sequence")
    path_file = Path(offline["task"]["path"])
    bank = PathBank([path_file], "cpu")
    speed = offline["task"]["target_speed_mps"]
    bank.speed[:] = speed
    bank.first_state[:, 5] = speed
    bank.durations[:] = bank.lengths / speed
    env = EndpointExtendedCondensedEnv(bank, batch_size=1, device="cpu",
        horizon=35, backend="rti_cpu", seed=seed, randomize_plant=True,
        current_max_mps=1., current_profile="harmonic",
        admm_iterations=400, admm_rho=10., attitude_pitch_weight=100.,
        attitude_yaw_weight=30., fsp_enabled=False, preview_mode="multiscale",
        yaw_reward="course")
    env.reset(torch.zeros(1, dtype=torch.long))
    install_target_domain(env, fin_gain_scale=.9)
    expected = offline["plant"]
    actual_parameters = env.parameters[0].tolist()
    for index, key in enumerate(("mass_scale", "damping_scale", "thrust_gain",
                                  "elevator_gain", "rudder_gain")):
        if abs(actual_parameters[index] - expected[key]) > 1e-10:
            raise ValueError(f"plant parameter mismatch: {key}")
    scale = env.model.control_scale.numpy()
    physical = torch.as_tensor(commands * scale, dtype=env.dtype)

    def scheduled_solve():
        index = min(int(env.steps[0]) // 5, len(physical)-1)
        return physical[index:index+1].clone(), torch.ones(1, dtype=torch.bool)

    def exact_low_level(_action):
        return env.base_command.clone(), env.state[:, 8:10].clone()

    env._solve_mpc = scheduled_solve
    env._low_level_command = exact_low_level
    env.base_command = physical[:1].clone()
    env.last_mpc_ok[:] = True
    rows = []
    first_anomaly = None
    task = offline["task"]
    for step in range(len(commands)*5):
        _, _, done, info = env.step(torch.zeros(1, 3))
        _, lateral, depth, _, _, speed_error, _ = env._errors()
        row = {"time_s": float(env.steps[0]) * .1,
               "progress_m": float(env.progress[0]),
               "speed_error_mps": float(speed_error[0]),
               "lateral_error_m": float(lateral[0]),
               "depth_error_m": float(depth[0]),
               "pitch_deg": float(np.rad2deg(env.state[0, 3].item())),
               "thrust_command_n": float(info["command"][0, 0]),
               "elevator_command_deg": float(np.rad2deg(info["command"][0, 1].item())),
               "rudder_command_deg": float(np.rad2deg(info["command"][0, 2].item())),
               "thrust_actual_n": float(env.actuator[0, 0]),
               "elevator_actual_deg": float(np.rad2deg(env.actuator[0, 1].item())),
               "rudder_actual_deg": float(np.rad2deg(env.actuator[0, 2].item())),
               "current_ned_json": json.dumps(env.current_ned[0].tolist()),
               "success": bool(info["success"][0]),
               "failed": bool(info["failed"][0]),
               "timeout": bool(info["timeout"][0])}
        rows.append(row)
        if first_anomaly is None:
            for name, value, limit in (("speed", row["speed_error_mps"], task["speed_tolerance_mps"]),
                                       ("lateral", row["lateral_error_m"], task["lateral_corridor_m"]),
                                       ("depth", row["depth_error_m"], task["depth_corridor_m"]),
                                       ("pitch", row["pitch_deg"], task["pitch_limit_deg"])):
                if abs(value) > limit:
                    first_anomaly = [row["time_s"], name]
                    break
        if bool(done[0]):
            break
    if paired_target_trace is not None:
        with paired_target_trace.open(newline="") as stream:
            reference_first = next(csv.DictReader(stream))
        if not np.allclose(json.loads(rows[0]["current_ned_json"]),
                           json.loads(reference_first["current_ned_json"]),
                           rtol=0., atol=1e-12):
            raise ValueError("paired target current draw does not match")
    out.parent.mkdir(parents=True, exist_ok=True)
    trace_file = out.with_suffix(".csv")
    with trace_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"path": str(path_file), "seed": seed,
               "offline_result_sha256": digest(offline_file),
               "commands_sha256": digest(command_file),
               "paired_target_trace_sha256": digest(paired_target_trace)
                   if paired_target_trace else None,
               "target_profile": "randomized_plant_harmonic_current_process_noise",
               "steps": len(rows), "strict_success": bool(rows[-1]["success"] and
                                                         first_anomaly is None),
               "simulator_success": rows[-1]["success"],
               "first_anomaly": first_anomaly,
               "progress_fraction": float(env.progress[0]/bank.lengths[0]),
               "max_abs_lateral_m": max(abs(r["lateral_error_m"]) for r in rows),
               "max_abs_depth_m": max(abs(r["depth_error_m"]) for r in rows),
               "max_abs_pitch_deg": max(abs(r["pitch_deg"]) for r in rows),
               "max_abs_speed_error_mps": max(abs(r["speed_error_mps"]) for r in rows),
               "trace": str(trace_file)}
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--paired-target-trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.offline_result, args.seed, args.out,
                         args.paired_target_trace), indent=2))


if __name__ == "__main__":
    main()
