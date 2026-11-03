"""Long-horizon, multi-start actuator/queue-aware path feasibility search.

The nonlinear program uses 0.5 s command blocks and exact 0.02 s actuator
substeps. Every candidate is replayed at plant resolution before classification.
Failure to find a candidate is never interpreted as infeasibility proof.
"""
import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.model import (DT, D_COEFF, ELEVATOR_COEFF, MASS, M_EFF,
                             RUDDER_COEFF, MISMATCHED_PLANT, ActuatorModel,
                             LoopPath, derivative, rk4_step)

CONFIG = Path(__file__).parent / "configs/tasks.json"
SCALE = np.array([1_500_000., np.pi / 4, np.pi / 4])
NZ = 22  # 10 hull + 3 actual actuator + 3 x 3 queued commands


def plant_delay_steps(plant):
    return [int(np.ceil(delay / DT - 1e-12))
            for delay in (plant.thrust_delay_s, plant.fin_delay_s,
                          plant.fin_delay_s)]


def queue_slots(plant):
    return max(3, *plant_delay_steps(plant))


def randomized_plant_for_seed(seed):
    """Reproduce the path environment's reset draw, before target fin scaling."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    def uniform(shape, low, high):
        return low + (high-low)*torch.rand(shape, generator=generator,
                                           dtype=torch.float64)
    uniform((1,), 0., 1.)  # steady-current amplitude draw
    uniform((1,), -np.pi, np.pi)  # current bearing draw
    base = MISMATCHED_PLANT
    factors = (uniform((1, 5), 0., 1.).numpy()[0] *
               np.array([.06, .14, .10, .20, .20]) +
               np.array([.97, .93, .95, .90, .90]))
    return replace(base,
        mass_scale=base.mass_scale*factors[0],
        damping_scale=base.damping_scale*factors[1],
        thrust_gain=base.thrust_gain*factors[2],
        elevator_gain=base.elevator_gain*factors[3],
        rudder_gain=base.rudder_gain*factors[4],
        thrust_delay_s=float(uniform((1,), .02, .08)[0]),
        fin_delay_s=float(uniform((1,), .02, .08)[0]),
        thrust_tau_s=float(uniform((1,), .06, .14)[0]),
        fin_tau_s=float(uniform((1,), .05, .12)[0]))


def realized_target_schedule(path_file, task, seed, substeps):
    """Clone the target environment's future current and AR process-noise draws.

    This deliberately gives the offline optimizer information unavailable to
    an online controller. Only path-feasibility diagnosis uses it.
    """
    import torch

    from aav_mpc.controller import EndpointExtendedCondensedEnv
    from simulator.batch_path_env import PathBank

    bank = PathBank([path_file], "cpu")
    bank.speed[:] = task["target_speed_mps"]
    bank.first_state[:, 5] = task["target_speed_mps"]
    bank.durations[:] = bank.lengths / task["target_speed_mps"]
    env = EndpointExtendedCondensedEnv(bank, batch_size=1, device="cpu",
        horizon=35, backend="rti_cpu", seed=seed, randomize_plant=True,
        current_max_mps=1., current_profile="harmonic",
        admm_iterations=400, admm_rho=10., attitude_pitch_weight=100.,
        attitude_yaw_weight=30., fsp_enabled=False, preview_mode="multiscale",
        yaw_reward="course")
    env.reset(torch.zeros(1, dtype=torch.long))
    generator = torch.Generator(device="cpu")
    generator.set_state(env.generator.get_state())
    phase = env.current_phase.clone()
    disturbance = env.disturbance.clone()
    currents = np.zeros((substeps, 3))
    disturbances = np.zeros((substeps, 5))
    for k in range(substeps):
        phase = phase + 2*np.pi*DT*env.current_frequency_hz
        current = env.current_base + (env.current_wave*phase.cos()[..., None]).sum(1)
        noise = torch.randn((1, 5), device="cpu", dtype=env.dtype,
                            generator=generator)
        disturbance = (env.decay*disturbance +
                       np.sqrt(1-env.decay**2)*env.disturbance_std*noise)
        currents[k] = current[0].numpy()
        disturbances[k] = disturbance[0].numpy()
    return currents, disturbances


def stage_function(block_s=0.5, plant=MISMATCHED_PLANT, target_domain=False,
                   exogenous=False):
    """Fixed-parameter, no-current plant with queued command history."""
    substeps = round(block_s / DT)
    if abs(substeps * DT - block_s) > 1e-10:
        raise ValueError("block_s must be divisible by DT")
    p = plant
    inverse_mass = ca.DM(np.linalg.inv(p.mass_scale * M_EFF))
    damping = ca.DM(D_COEFF)
    elevator = ca.DM(ELEVATOR_COEFF)
    rudder = ca.DM(RUDDER_COEFF)
    slots = queue_slots(p)
    nz = 13 + 3*slots + (3 if target_domain else 0)
    z0 = ca.MX.sym("z", nz)
    cmd = ca.MX.sym("cmd", 3)
    current_sequence = ca.MX.sym("current", 3, substeps) if exogenous else None
    disturbance_sequence = ca.MX.sym("disturbance", 5, substeps) if exogenous else None
    physical = ca.diag(ca.DM(SCALE)) @ cmd

    def body_derivative(x, actual, current, disturbance):
        pitch, yaw = x[3], x[4]
        u, v, w, q_body, r_body = [x[i] for i in range(5, 10)]
        ct, st, cp, sp = ca.cos(pitch), ca.sin(pitch), ca.cos(yaw), ca.sin(yaw)
        position = ca.vertcat(cp * ct * u - sp * v + cp * st * w,
                              sp * ct * u + cp * v + sp * st * w,
                              -st * u + ct * w)
        position += current
        coriolis = p.mass_scale * MASS * ca.vertcat(
            -r_body * v + q_body * w, r_body * u, -q_body * u, 0, 0)
        force = ca.vertcat(p.thrust_gain * actual[0], 0, 0, 0, 0)
        force += u * u * (p.elevator_gain * elevator * actual[1] +
                          p.rudder_gain * rudder * actual[2])
        acceleration = inverse_mass @ (force - coriolis - p.damping_scale * u * damping @ x[5:10])
        acceleration += disturbance
        return ca.vertcat(position, q_body, r_body / ct, acceleration)

    x, a, queue = z0[:10], z0[10:13], z0[13:13+3*slots]
    rate = z0[13+3*slots:] if target_domain else None
    delays = plant_delay_steps(p)
    alpha = np.array([-np.expm1(-DT / p.thrust_tau_s),
                      -np.expm1(-DT / p.fin_tau_s),
                      -np.expm1(-DT / p.fin_tau_s)])
    omega = np.array([1. / max(p.thrust_tau_s, .03),
                      1. / max(p.fin_tau_s, .03),
                      1. / max(p.fin_tau_s, .03)])
    snapshots = []
    for k in range(substeps):
        current = current_sequence[:, k] if exogenous else ca.DM.zeros(3, 1)
        disturbance = disturbance_sequence[:, k] if exogenous else ca.DM.zeros(5, 1)
        target = ca.vertcat(*[physical[j] if delays[j] == 0 else
            queue[3*(delays[j]-1)+j] for j in range(3)])
        queue = ca.vertcat(physical, queue[:-3])
        if target_domain:
            rate = rate + DT * (ca.diag(ca.DM(omega ** 2)) @ (target - a) -
                                ca.diag(ca.DM(2 * omega)) @ rate)
            a = a + DT * rate
        else:
            a = a + ca.diag(ca.DM(alpha)) @ (target - a)
        h = DT
        if target_domain:
            half = h / 2
            for _ in range(2):
                k1 = body_derivative(x, a, current, disturbance)
                k2 = body_derivative(x + half * k1 / 2, a, current, disturbance)
                x = x + half * k2
        else:
            k1 = body_derivative(x, a, current, disturbance)
            k2 = body_derivative(x + h * k1 / 2, a, current, disturbance)
            k3 = body_derivative(x + h * k2 / 2, a, current, disturbance)
            k4 = body_derivative(x + h * k3, a, current, disturbance)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if (k + 1) % 5 == 0:
            snapshots.append(x)
    end = ca.vertcat(x, a, queue, rate) if target_domain else ca.vertcat(x, a, queue)
    inputs = [z0, cmd, current_sequence, disturbance_sequence] if exogenous else [z0, cmd]
    return ca.Function("stage", inputs, [end, ca.hcat(snapshots)])


def target(path, s):
    xyz = path.position_at(s)
    tangent = path.tangent_at(s)
    yaw = float(np.interp(s, path.distances, path.yaw))
    return xyz, tangent, yaw


def build_problem(path, task, horizon_s, block_s=0.5, plant=MISMATCHED_PLANT,
                  target_domain=False, current_schedule=None,
                  disturbance_schedule=None, max_solve_s=None,
                  free_progress=False):
    periods = math.ceil(horizon_s / block_s)
    opti = ca.Opti()
    z = opti.variable(13 + 3*queue_slots(plant) + (3 if target_domain else 0),
                      periods + 1)
    cmd = opti.variable(3, periods)
    slack = opti.variable(4, periods)
    opti.subject_to(ca.vec(slack) >= 0)
    opti.subject_to(opti.bounded(0, ca.vec(cmd[0, :]), 1))
    opti.subject_to(opti.bounded(-1, ca.vec(cmd[1:3, :]), 1))
    if free_progress:
        progress = opti.variable(periods + 1)
        opti.subject_to(progress[0] == 0)
        opti.subject_to(opti.bounded(0, progress, path.length))
        opti.subject_to(opti.bounded(
            (task.get("local_speed_min_mps", task["target_speed_mps"]-2.)) * block_s,
            ca.diff(progress),
            (task.get("local_speed_max_mps", task["target_speed_mps"]+2.)) * block_s))
        if horizon_s * task.get("local_speed_max_mps", task["target_speed_mps"]+2.) >= .995*path.length:
            opti.subject_to(progress[-1] >= .995 * path.length)
    else:
        progress = None
    initial = np.zeros(z.shape[0])
    initial[:3] = path.points[0]
    initial[3:5] = [path.pitch[0], path.yaw[0]]
    initial[5] = task["target_speed_mps"]
    opti.subject_to(z[:, 0] == initial)
    exogenous = current_schedule is not None or disturbance_schedule is not None
    if exogenous:
        if not target_domain or current_schedule is None or disturbance_schedule is None:
            raise ValueError("current and disturbance schedules require target dynamics")
        required = periods * round(block_s / DT)
        if (current_schedule.shape != (required, 3) or
                disturbance_schedule.shape != (required, 5)):
            raise ValueError("invalid exogenous schedule dimensions")
    stage = stage_function(block_s, plant, target_domain, exogenous)
    objective = 0
    limits = [task["lateral_corridor_m"], task["depth_corridor_m"],
              np.deg2rad(task["pitch_limit_deg"]), task["speed_tolerance_mps"]]

    def path_quantities(hull, s):
        point, tangent, yaw = target(path, s)
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        lateral = ca.dot(ca.DM(normal), hull[:2] - ca.DM(point[:2]))
        depth = hull[2] - point[2]
        pitch = hull[3]
        ct, st, cp, sp = ca.cos(pitch), ca.sin(pitch), ca.cos(hull[4]), ca.sin(hull[4])
        velocity = ca.vertcat(cp * ct * hull[5] - sp * hull[6] + cp * st * hull[7],
                              sp * ct * hull[5] + cp * hull[6] + sp * st * hull[7],
                              -st * hull[5] + ct * hull[7])
        speed_error = ca.dot(ca.DM(tangent), velocity) - task["target_speed_mps"]
        return [lateral, depth, pitch, speed_error], point, yaw

    if free_progress:
        grid = path.distances
        def interpolate(values):
            return ca.interpolant(
                f"path_{len(interpolants)}", "linear", [grid],
                np.asarray(values, dtype=float))
        interpolants = []
        px = interpolate(path.points[:, 0])
        interpolants.append(px)
        py = interpolate(path.points[:, 1])
        interpolants.append(py)
        pz = interpolate(path.points[:, 2])
        interpolants.append(pz)
        tx = interpolate(path.tangent[:, 0])
        interpolants.append(tx)
        ty = interpolate(path.tangent[:, 1])
        interpolants.append(ty)
        tz = interpolate(path.tangent[:, 2])
        interpolants.append(tz)
        yaw_at = interpolate(path.yaw)
        interpolants.append(yaw_at)

        def path_quantities_free(hull, s):
            point = ca.vertcat(px(s), py(s), pz(s))
            tangent = ca.vertcat(tx(s), ty(s), tz(s))
            yaw = yaw_at(s)
            normal = ca.vertcat(-ca.sin(yaw), ca.cos(yaw))
            lateral = ca.dot(normal, hull[:2] - point[:2])
            depth = hull[2] - point[2]
            pitch = hull[3]
            ct, st, cp, sp = ca.cos(pitch), ca.sin(pitch), ca.cos(hull[4]), ca.sin(hull[4])
            velocity = ca.vertcat(cp * ct * hull[5] - sp * hull[6] + cp * st * hull[7],
                                  sp * ct * hull[5] + cp * hull[6] + sp * st * hull[7],
                                  -st * hull[5] + ct * hull[7])
            speed_error = ca.dot(tangent, velocity) - task["target_speed_mps"]
            return [lateral, depth, pitch, speed_error], point, yaw
    else:
        path_quantities_free = None

    for k in range(periods):
        if exogenous:
            start = k * round(block_s / DT)
            stop = (k + 1) * round(block_s / DT)
            end, snapshots = stage(z[:, k], cmd[:, k],
                                   ca.DM(current_schedule[start:stop].T),
                                   ca.DM(disturbance_schedule[start:stop].T))
        else:
            end, snapshots = stage(z[:, k], cmd[:, k])
        opti.subject_to(z[:, k + 1] == end)
        s = (progress[k + 1] if free_progress else
             min(path.length, (k + 1) * block_s * task["target_speed_mps"]))
        hull = z[:10, k + 1]
        quantities, point, yaw = ((path_quantities_free(hull, s) if free_progress
                                   else path_quantities(hull, s)))
        lateral, depth, pitch, speed_error = quantities
        for j in range(4):
            opti.subject_to(opti.bounded(-limits[j] - slack[j, k], quantities[j],
                                          limits[j] + slack[j, k]))
        if target_domain:
            # Replay and the online task check boundaries every 0.1 s. The
            # source problem predates this guard; preserve its old behavior.
            for j in range(snapshots.shape[1] - 1):
                sample_s = (progress[k] + (j + 1) / snapshots.shape[1] *
                            (progress[k + 1] - progress[k])
                            if free_progress else min(path.length,
                            (k * block_s + (j + 1) * .1) * task["target_speed_mps"]))
                sample_quantities, _, _ = ((path_quantities_free(snapshots[:, j], sample_s)
                                            if free_progress else
                                            path_quantities(snapshots[:, j], sample_s)))
                for i in range(4):
                    opti.subject_to(opti.bounded(-limits[i] - slack[i, k],
                                                  sample_quantities[i],
                                                  limits[i] + slack[i, k]))
        objective += (lateral / limits[0]) ** 2 + (depth / limits[1]) ** 2
        objective += 0.25 * (speed_error / limits[3]) ** 2
        objective += 1000 * ca.sumsqr(slack[:, k] / ca.DM(limits))
        objective += .002 * ca.sumsqr(cmd[:, k])
        if k:
            objective += .02 * ca.sumsqr(cmd[:, k] - cmd[:, k - 1])
        if free_progress:
            nominal_s = min(path.length, (k + 1) * block_s * task["target_speed_mps"])
            nominal_point, _, nominal_yaw = target(path, nominal_s)
            opti.set_initial(z[:3, k + 1], nominal_point)
            opti.set_initial(z[3:5, k + 1], [path.pitch_at(nominal_s), nominal_yaw])
        else:
            opti.set_initial(z[:3, k + 1], point)
            opti.set_initial(z[3:5, k + 1], [path.pitch_at(s), yaw])
        opti.set_initial(z[5, k + 1], task["target_speed_mps"])
        if free_progress:
            opti.set_initial(progress[k + 1], min(path.length,
                (k + 1) * block_s * task["target_speed_mps"]))
    opti.minimize(objective)
    ipopt_options = {"print_level": 0, "max_iter": 500, "tol": 1e-5}
    if max_solve_s is not None:
        ipopt_options["max_cpu_time"] = max_solve_s
    opti.solver("ipopt", {"expand": False, "print_time": False}, ipopt_options)
    return opti, z, cmd, slack


class SecondOrderActuatorReplay:
    """NumPy replay of the target environment's delayed second-order actuator."""

    def __init__(self, plant):
        self.plant = plant
        self.actual = np.zeros(3)
        self.rate = np.zeros(3)
        delays = [plant.thrust_delay_s, plant.fin_delay_s, plant.fin_delay_s]
        self.queues = [deque([0.] * int(np.ceil(delay / DT - 1e-12)))
                       for delay in delays]
        self.tau = np.array([plant.thrust_tau_s, plant.fin_tau_s, plant.fin_tau_s])

    def step(self, command):
        target = np.asarray(command, dtype=float).copy()
        for channel, queue in enumerate(self.queues):
            if queue:
                queue.append(target[channel])
                target[channel] = queue.popleft()
        omega = 1. / np.maximum(self.tau, .03)
        acceleration = omega ** 2 * (target - self.actual) - 2. * omega * self.rate
        self.rate += DT * acceleration
        self.actual += DT * self.rate
        return self.actual.copy()


def target_midpoint_step(state, actual, plant, current=None, disturbance=None):
    x = state.copy()
    half = DT / 2.
    current = np.zeros(3) if current is None else np.asarray(current)
    disturbance = np.zeros(5) if disturbance is None else np.asarray(disturbance)
    def body_derivative(value):
        derivative_value = derivative(value, actual, plant)
        derivative_value[:3] += current
        derivative_value[5:] += disturbance
        return derivative_value
    for _ in range(2):
        k1 = body_derivative(x)
        k2 = body_derivative(x + half * k1 / 2.)
        x += half * k2
    return x


def replay(path, task, normalized, block_s, plant=MISMATCHED_PLANT,
           target_domain=False, stop_on_completion=False,
           current_schedule=None, disturbance_schedule=None):
    if ((current_schedule is None) != (disturbance_schedule is None) or
            (current_schedule is not None and not target_domain)):
        raise ValueError("both exogenous schedules require target dynamics")
    actuator = SecondOrderActuatorReplay(plant) if target_domain else ActuatorModel(plant)
    state = np.zeros(10)
    state[:3] = path.points[0]
    state[3:5] = path.pitch[0], path.yaw[0]
    state[5] = task["target_speed_mps"]
    margins = {key: float("inf") for key in ("lateral_m", "depth_m", "pitch_deg", "speed_mps")}
    trace = []
    completed = False
    for k, control in enumerate(normalized):
        for j in range(round(block_s / DT)):
            index = k * round(block_s / DT) + j
            current = current_schedule[index] if current_schedule is not None else None
            disturbance = (disturbance_schedule[index]
                           if disturbance_schedule is not None else None)
            actual = actuator.step(control * SCALE)
            state = (target_midpoint_step(state, actual, plant, current, disturbance)
                     if target_domain
                     else rk4_step(state, actual, plant_params=plant))
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
            if (stop_on_completion and path.progress >= path.length - 2 and
                    np.linalg.norm(state[:3] - path.points[-1]) < 10):
                completed = True
                break
        if completed:
            break
    found = (all(v >= 0 for v in margins.values()) and
             path.progress >= path.length - 2 and np.linalg.norm(state[:3] - path.points[-1]) < 10)
    target_distance = min(path.length, len(normalized)*block_s*task["target_speed_mps"])
    prefix_found = (all(v >= 0 for v in margins.values()) and
                    path.progress >= target_distance - 2 and
                    np.linalg.norm(state[:3] - path.position_at(target_distance)) < 10)
    return {"found_full_path_feasible": bool(found), "progress_fraction": float(path.progress / path.length),
            "found_command_horizon_prefix_feasible": bool(prefix_found),
            "minimum_constraint_margins": margins, "trace": trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("T1", "H15"), default="T1")
    parser.add_argument("--path", type=Path)
    parser.add_argument("--horizon-s", type=float, default=20.)
    parser.add_argument("--block-s", type=float, default=.5)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--max-solve-s", type=float,
                        help="IPOPT CPU-time cap for each start")
    parser.add_argument("--fin-gain-scale", type=float, default=1.)
    parser.add_argument("--sampled-plant-seed", type=int,
                        help="reproduce one randomized hull/actuator draw; excludes current and noise")
    parser.add_argument("--realized-target-seed", type=int,
                        help="also use this seed's future harmonic current and process-noise draw")
    parser.add_argument("--target-deterministic", action="store_true",
                        help="optimize fixed no-current/no-noise target actuator and integrator")
    parser.add_argument("--initial-commands", type=Path,
                        help="normalized [blocks,3] command warm start from an earlier run")
    parser.add_argument("--free-progress", action="store_true",
                        help="optimize monotone arc-length progress instead of fixed-speed progress")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    task = (json.loads((CONFIG.parent / "h15.json").read_text())
            if args.task == "H15" else json.loads(CONFIG.read_text())[args.task])
    if args.path is not None:
        task["path"] = str(args.path)
    if "path" not in task:
        parser.error("--path is required for this task")
    if not 0 < args.fin_gain_scale <= 1:
        parser.error("--fin-gain-scale must lie in (0, 1]")
    if args.max_solve_s is not None and args.max_solve_s <= 0:
        parser.error("--max-solve-s must be positive")
    if args.realized_target_seed is not None and (
            not args.target_deterministic or
            args.sampled_plant_seed != args.realized_target_seed):
        parser.error("realized target requires target dynamics and matching sampled plant seed")
    base_plant = (randomized_plant_for_seed(args.sampled_plant_seed)
                  if args.sampled_plant_seed is not None else MISMATCHED_PLANT)
    plant = replace(base_plant,
                    elevator_gain=base_plant.elevator_gain*args.fin_gain_scale,
                    rudder_gain=base_plant.rudder_gain*args.fin_gain_scale)
    path = LoopPath(ROOT / task["path"])
    horizon = (args.horizon_s if args.target_deterministic else
               min(args.horizon_s, path.length / task["target_speed_mps"]))
    period_count = math.ceil(horizon / args.block_s)
    current_schedule = disturbance_schedule = None
    schedule_file = None
    if args.realized_target_seed is not None:
        current_schedule, disturbance_schedule = realized_target_schedule(
            ROOT / task["path"], task, args.realized_target_seed,
            period_count * round(args.block_s / DT))
        schedule_file = args.out.with_name(args.out.stem + "_exogenous.npz")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(schedule_file, current=current_schedule,
                            disturbance=disturbance_schedule)
    opti, z, controls, slack = build_problem(path, task, horizon, args.block_s,
        plant, args.target_deterministic, current_schedule, disturbance_schedule,
        args.max_solve_s, args.free_progress)
    warm_start = None
    if args.initial_commands is not None:
        warm_start = np.load(args.initial_commands, allow_pickle=False)
        if (warm_start.ndim != 2 or warm_start.shape[1] != 3 or
                not 1 <= warm_start.shape[0] <= controls.shape[1]):
            parser.error("--initial-commands must have shape [at most blocks, 3]")
        if warm_start.shape[0] < controls.shape[1]:
            warm_start = np.vstack((warm_start, np.tile(warm_start[-1],
                                (controls.shape[1] - warm_start.shape[0], 1))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(args.starts):
        rng = np.random.default_rng(seed + 1202)
        start = np.zeros((3, controls.shape[1]))
        if warm_start is not None and seed == 0:
            start[:] = warm_start.T
        else:
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
        checked = replay(replay_path, task, plan.T, args.block_s, plant,
                         target_domain=args.target_deterministic,
                         stop_on_completion=args.target_deterministic,
                         current_schedule=current_schedule,
                         disturbance_schedule=disturbance_schedule)
        trace = checked.pop("trace")
        np.save(args.out.with_name(args.out.stem + f"_start{seed}_commands.npy"), plan.T)
        with args.out.with_name(args.out.stem + f"_start{seed}_trace.jsonl").open("w") as stream:
            for row in trace:
                stream.write(json.dumps(row) + "\n")
        row = {"start": seed, "solver_status": status, "wall_s": time.perf_counter() - tic,
               "max_optimized_slack": optimized_slack, **checked}
        results.append(row)
        args.out.write_text(json.dumps({"task": task, "plant": vars(plant), "horizon_s": horizon,
            "block_s": args.block_s, "target_deterministic": args.target_deterministic,
            "sampled_plant_seed": args.sampled_plant_seed,
            "realized_target_seed": args.realized_target_seed,
            "exogenous_schedule": str(schedule_file) if schedule_file else None,
            "max_solve_s": args.max_solve_s,
            "initial_commands": str(args.initial_commands) if args.initial_commands else None,
            "free_progress": args.free_progress,
            "results": results}, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
