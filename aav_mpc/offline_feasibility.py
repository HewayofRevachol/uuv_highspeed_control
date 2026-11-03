"""Long-horizon, multi-start actuator/queue-aware T1 feasibility search.

The nonlinear program uses 0.5 s command blocks and exact 0.02 s actuator
substeps. Every candidate is replayed at plant resolution before classification.
Failure to find a candidate is never interpreted as infeasibility proof.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.model import (DT, D_COEFF, ELEVATOR_COEFF, MASS, M_EFF,
                             RUDDER_COEFF, MISMATCHED_PLANT, ActuatorModel,
                             LoopPath, rk4_step)

CONFIG = Path(__file__).parent / "configs/tasks.json"
SCALE = np.array([1_500_000., np.pi / 4, np.pi / 4])
NZ = 22  # 10 hull + 3 actual actuator + 3 x 3 queued commands


def stage_function(block_s=0.5):
    """Exact fixed-parameter, no-current plant with queued command history."""
    substeps = round(block_s / DT)
    if abs(substeps * DT - block_s) > 1e-10:
        raise ValueError("block_s must be divisible by DT")
    p = MISMATCHED_PLANT
    inverse_mass = ca.DM(np.linalg.inv(p.mass_scale * M_EFF))
    damping = ca.DM(D_COEFF)
    elevator = ca.DM(ELEVATOR_COEFF)
    rudder = ca.DM(RUDDER_COEFF)
    z0 = ca.MX.sym("z", NZ)
    cmd = ca.MX.sym("cmd", 3)
    physical = ca.diag(ca.DM(SCALE)) @ cmd

    def body_derivative(x, actual):
        pitch, yaw = x[3], x[4]
        u, v, w, q_body, r_body = [x[i] for i in range(5, 10)]
        ct, st, cp, sp = ca.cos(pitch), ca.sin(pitch), ca.cos(yaw), ca.sin(yaw)
        position = ca.vertcat(cp * ct * u - sp * v + cp * st * w,
                              sp * ct * u + cp * v + sp * st * w,
                              -st * u + ct * w)
        coriolis = p.mass_scale * MASS * ca.vertcat(
            -r_body * v + q_body * w, r_body * u, -q_body * u, 0, 0)
        force = ca.vertcat(p.thrust_gain * actual[0], 0, 0, 0, 0)
        force += u * u * (p.elevator_gain * elevator * actual[1] +
                          p.rudder_gain * rudder * actual[2])
        acceleration = inverse_mass @ (force - coriolis - p.damping_scale * u * damping @ x[5:10])
        return ca.vertcat(position, q_body, r_body / ct, acceleration)

    x, a, queue = z0[:10], z0[10:13], z0[13:22]
    alpha = np.array([-np.expm1(-DT / p.thrust_tau_s),
                      -np.expm1(-DT / p.fin_tau_s),
                      -np.expm1(-DT / p.fin_tau_s)])
    snapshots = []
    for k in range(substeps):
        target = queue[6:9]  # command issued three 0.02 s substeps earlier
        queue = ca.vertcat(physical, queue[:6])
        a = a + ca.diag(ca.DM(alpha)) @ (target - a)
        h = DT
        k1 = body_derivative(x, a)
        k2 = body_derivative(x + h * k1 / 2, a)
        k3 = body_derivative(x + h * k2 / 2, a)
        k4 = body_derivative(x + h * k3, a)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if (k + 1) % 5 == 0:
            snapshots.append(x)
    end = ca.vertcat(x, a, queue)
    return ca.Function("stage", [z0, cmd], [end, ca.hcat(snapshots)])


def target(path, s):
    xyz = path.position_at(s)
    tangent = path.tangent_at(s)
    yaw = float(np.interp(s, path.distances, path.yaw))
    return xyz, tangent, yaw


def build_problem(path, task, horizon_s, block_s=0.5):
    periods = math.ceil(horizon_s / block_s)
    opti = ca.Opti()
    z = opti.variable(NZ, periods + 1)
    cmd = opti.variable(3, periods)
    slack = opti.variable(4, periods)
    opti.subject_to(ca.vec(slack) >= 0)
    opti.subject_to(opti.bounded(0, ca.vec(cmd[0, :]), 1))
    opti.subject_to(opti.bounded(-1, ca.vec(cmd[1:3, :]), 1))
    initial = np.zeros(NZ)
    initial[:3] = path.points[0]
    initial[3:5] = [path.pitch[0], path.yaw[0]]
    initial[5] = task["target_speed_mps"]
    opti.subject_to(z[:, 0] == initial)
    stage = stage_function(block_s)
    objective = 0
    for k in range(periods):
        end, snapshots = stage(z[:, k], cmd[:, k])
        opti.subject_to(z[:, k + 1] == end)
        s = min(path.length, (k + 1) * block_s * task["target_speed_mps"])
        point, tangent, yaw = target(path, s)
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        hull = z[:10, k + 1]
        lateral = ca.dot(ca.DM(normal), hull[:2] - ca.DM(point[:2]))
        depth = hull[2] - point[2]
        pitch = hull[3]
        ct, st, cp, sp = ca.cos(pitch), ca.sin(pitch), ca.cos(hull[4]), ca.sin(hull[4])
        velocity = ca.vertcat(cp * ct * hull[5] - sp * hull[6] + cp * st * hull[7],
                              sp * ct * hull[5] + cp * hull[6] + sp * st * hull[7],
                              -st * hull[5] + ct * hull[7])
        speed_error = ca.dot(ca.DM(tangent), velocity) - task["target_speed_mps"]
        quantities = [lateral, depth, pitch, speed_error]
        limits = [task["lateral_corridor_m"], task["depth_corridor_m"],
                  np.deg2rad(task["pitch_limit_deg"]), task["speed_tolerance_mps"]]
        for j in range(4):
            opti.subject_to(opti.bounded(-limits[j] - slack[j, k], quantities[j],
                                          limits[j] + slack[j, k]))
        objective += (lateral / limits[0]) ** 2 + (depth / limits[1]) ** 2
        objective += 0.25 * (speed_error / limits[3]) ** 2
        objective += 1000 * ca.sumsqr(slack[:, k] / ca.DM(limits))
        objective += .002 * ca.sumsqr(cmd[:, k])
        if k:
            objective += .02 * ca.sumsqr(cmd[:, k] - cmd[:, k - 1])
        opti.set_initial(z[:3, k + 1], point)
        opti.set_initial(z[3:5, k + 1], [path.pitch_at(s), yaw])
        opti.set_initial(z[5, k + 1], task["target_speed_mps"])
    opti.minimize(objective)
    opti.solver("ipopt", {"expand": False, "print_time": False},
                {"print_level": 0, "max_iter": 500, "tol": 1e-5})
    return opti, z, cmd, slack


def replay(path, task, normalized, block_s):
    plant = MISMATCHED_PLANT
    actuator = ActuatorModel(plant)
    state = np.zeros(10)
    state[:3] = path.points[0]
    state[3:5] = path.pitch[0], path.yaw[0]
    state[5] = task["target_speed_mps"]
    margins = {key: float("inf") for key in ("lateral_m", "depth_m", "pitch_deg", "speed_mps")}
    trace = []
    for k, control in enumerate(normalized):
        for j in range(round(block_s / DT)):
            actual = actuator.step(control * SCALE)
            state = rk4_step(state, actual, plant_params=plant)
            if (j + 1) % 5:
                continue
            path.advance(state[:3])
            s = path.progress
            point, tangent, yaw = target(path, s)
            offset = state[:3] - point
            lateral = np.dot([-np.sin(yaw), np.cos(yaw)], offset[:2])
            pitch = np.rad2deg(state[3])
            velocity = np.array([np.cos(state[4]) * np.cos(state[3]) * state[5] -
                                 np.sin(state[4]) * state[6] + np.cos(state[4]) * np.sin(state[3]) * state[7],
                                 np.sin(state[4]) * np.cos(state[3]) * state[5] +
                                 np.cos(state[4]) * state[6] + np.sin(state[4]) * np.sin(state[3]) * state[7],
                                 -np.sin(state[3]) * state[5] + np.cos(state[3]) * state[7]])
            speed_error = np.dot(velocity, tangent) - task["target_speed_mps"]
            row = {"time_s": (k * round(block_s / DT) + j + 1) * DT,
                   "progress_m": float(s), "lateral_m": float(lateral),
                   "depth_m": float(offset[2]), "pitch_deg": float(pitch),
                   "speed_error_mps": float(speed_error),
                   "actual_thrust_n": float(actual[0]),
                   "actual_elevator_deg": float(np.rad2deg(actual[1])),
                   "actual_rudder_deg": float(np.rad2deg(actual[2]))}
            trace.append(row)
            for key, val in (("lateral_m", abs(lateral)), ("depth_m", abs(offset[2])),
                             ("pitch_deg", abs(pitch)), ("speed_mps", abs(speed_error))):
                margins[key] = min(margins[key],
                    {"lateral_m": task["lateral_corridor_m"],
                     "depth_m": task["depth_corridor_m"],
                     "pitch_deg": task["pitch_limit_deg"],
                     "speed_mps": task["speed_tolerance_mps"]}[key] - val)
    found = (all(v >= 0 for v in margins.values()) and
             path.progress >= path.length - 2 and np.linalg.norm(state[:3] - path.points[-1]) < 10)
    return {"found_full_path_feasible": bool(found), "progress_fraction": float(path.progress / path.length),
            "minimum_constraint_margins": margins, "trace": trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-s", type=float, default=20.)
    parser.add_argument("--block-s", type=float, default=.5)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(CONFIG.read_text())["T1"]
    path = LoopPath(ROOT / task["path"])
    horizon = min(args.horizon_s, path.length / task["target_speed_mps"])
    opti, z, controls, slack = build_problem(path, task, horizon, args.block_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(args.starts):
        rng = np.random.default_rng(seed + 1202)
        start = np.zeros((3, controls.shape[1]))
        start[0] = .11 + .03 * rng.random(controls.shape[1])
        if seed:
            start[2] = (seed - 1) * .25 * np.sin(np.linspace(0, 4 * np.pi, controls.shape[1]))
        opti.set_initial(controls, start)
        tic = time.perf_counter()
        status = "solved"
        try:
            solution = opti.solve()
            plan = np.asarray(solution.value(controls))
            optimized_slack = float(np.max(solution.value(slack)))
        except RuntimeError as exc:
            status = str(exc).splitlines()[0]
            plan = np.asarray(opti.debug.value(controls))
            optimized_slack = float(np.max(opti.debug.value(slack)))
        replay_path = LoopPath(ROOT / task["path"])
        checked = replay(replay_path, task, plan.T, args.block_s)
        trace = checked.pop("trace")
        np.save(args.out.with_name(args.out.stem + f"_start{seed}_commands.npy"), plan.T)
        with args.out.with_name(args.out.stem + f"_start{seed}_trace.jsonl").open("w") as stream:
            for row in trace:
                stream.write(json.dumps(row) + "\n")
        row = {"start": seed, "solver_status": status, "wall_s": time.perf_counter() - tic,
               "max_optimized_slack": optimized_slack, **checked}
        results.append(row)
        args.out.write_text(json.dumps({"task": task, "horizon_s": horizon,
            "block_s": args.block_s, "results": results}, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
