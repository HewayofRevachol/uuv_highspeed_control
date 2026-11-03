"""Actuator-aware empirical recovery screen for the repository's RTI/OSQP MPC.

The screen tests *one optimized control sequence per candidate speed*.  It is
neither an existential reachability solver nor a recursive-feasibility proof.
The real plant parameters and current are never read by the screen.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from research.controllers.ddat import ExperimentDDATPathEnv
from simulator.batch_path_env import PathBank
from simulator.model import DT, MISMATCHED_PLANT


class ProgressEnv(ExperimentDDATPathEnv):
    """Use one projected path-progress speed in the existing DDAT/FSP MPC."""

    def __init__(self, *args, **kwargs):
        rti_constraints = kwargs.pop("rti_constraints", None)
        super().__init__(*args, **kwargs)
        self.progress_speed = float(self.paths.speed[0, 0])
        if rti_constraints is not None:
            self.mpc.path_constraints = rti_constraints

    def _reference(self, time_offset=0.):
        offsets = torch.arange(1, self.mpc.horizon + 1, device=self.device,
                               dtype=self.dtype) * self.mpc.dt
        speed = self.progress_speed
        distances = self.progress[:, None] + speed * (offsets[None] + time_offset)
        position, tangent, pitch, yaw, _ = self.paths.sample(self.path_ids, distances)
        velocity = speed * tangent
        reference = torch.cat((position, pitch[..., None], yaw[..., None], velocity), -1)
        reference[..., :3] -= offsets[None, :, None] * self.current_estimate[:, None]
        reference[..., 5:8] -= self.current_estimate[:, None]
        return reference


def controller_snapshot(env):
    names = ("plan", "states", "previous", "last_primal_residual",
             "last_dual_residual", "last_qp_valid")
    saved = {name: getattr(env.mpc, name).clone() for name in names
             if hasattr(env.mpc, name)}
    if env.fsp is not None:
        saved["fsp"] = {name: getattr(env.fsp, name).clone() for name in
                        ("velocity_error", "expected_next", "expected_valid")}
    saved["base_command"] = env.base_command.clone()
    saved["last_mpc_ok"] = env.last_mpc_ok.clone()
    saved["progress_speed"] = env.progress_speed
    return saved


def restore_controller(env, saved):
    for name, value in saved.items():
        if name == "fsp":
            for key, tensor in value.items():
                setattr(env.fsp, key, tensor)
        elif name == "progress_speed":
            env.progress_speed = value
        elif name in ("base_command", "last_mpc_ok"):
            setattr(env, name, value)
        else:
            setattr(env.mpc, name, value)


def speed_components(candidates, feasible):
    """Return every connected component; never fill an infeasible gap."""
    components = []
    for speed, ok in zip(candidates, feasible):
        if ok:
            if components and components[-1][-1] == previous:
                components[-1].append(float(speed))
            else:
                components.append([float(speed)])
        previous = float(speed)
    return [[part[0], part[-1]] for part in components]


def _rollout(env, plan, current, parameter_scale, steps, corridor_y,
             corridor_z, pitch_limit_deg, terminal_slack,
             max_saturation_fraction):
    """Replay a command plan through the observed delay queue and lag model."""
    state = env.state.clone()
    actuator = env.actuator.clone()
    queue = env.command_history.clone()
    progress = env.progress.clone()
    nominal = env.model.tensor([
        MISMATCHED_PLANT.mass_scale, MISMATCHED_PLANT.damping_scale,
        MISMATCHED_PLANT.thrust_gain, MISMATCHED_PLANT.elevator_gain,
        MISMATCHED_PLANT.rudder_gain])
    parameters = (nominal * env.model.tensor(parameter_scale))[None]
    # Use declared nominal actuator parameters, never the sampled plant's
    # hidden delay_steps/tau. Actual actuator angle and command queue are read.
    delay = env.model.tensor([MISMATCHED_PLANT.thrust_delay_s,
                              MISMATCHED_PLANT.fin_delay_s,
                              MISMATCHED_PLANT.fin_delay_s])
    delay_steps = torch.ceil(delay / DT - 1e-12).long()[None]
    tau = env.model.tensor([MISMATCHED_PLANT.thrust_tau_s,
                            MISMATCHED_PLANT.fin_tau_s,
                            MISMATCHED_PLANT.fin_tau_s])
    alpha = torch.where(tau > 0, -torch.expm1(-DT / tau.clamp_min(1e-12)),
                        torch.ones_like(tau))
    max_y = max_z = max_pitch = max_error = 0.
    saturated = 0
    finite = True
    for k in range(steps):
        command = plan[min(k, len(plan) - 1):min(k, len(plan) - 1) + 1]
        command = command * env.model.control_scale
        normalized = plan[min(k, len(plan)-1)]
        saturated += int(bool(((normalized <= env.model.lower + .015) |
                               (normalized >= env.model.upper - .015)).any()))
        for _ in range(round(env.mpc.dt / DT)):
            delayed = queue.gather(1, (delay_steps - 1).clamp_min(0)[:, None]).squeeze(1)
            target = torch.where(delay_steps > 0, delayed, command)
            queue = torch.cat((command[:, None], queue[:, :-1]), 1)
            actuator = actuator + alpha * (target - actuator)
            state = env.model.rk4(state, actuator, parameters, current=current, dt=DT)
        if not bool(torch.isfinite(state).all()):
            finite = False
            break
        progress = env.paths.advance(env.path_ids, progress, state[:, :3])
        point, _, _, yaw, _ = env.paths.sample(env.path_ids, progress[:, None])
        offset = state[0, :3] - point[0, 0]
        lateral = -torch.sin(yaw[0, 0]) * offset[0] + torch.cos(yaw[0, 0]) * offset[1]
        max_y = max(max_y, abs(float(lateral)))
        max_z = max(max_z, abs(float(offset[2])))
        max_pitch = max(max_pitch, abs(float(state[0, 3])) * 180 / np.pi)
        max_error = max(max_error, float(torch.linalg.vector_norm(offset)))
    feasible = (finite and max_y <= corridor_y and max_z <= corridor_z and
                max_pitch <= pitch_limit_deg and max_error <= terminal_slack and
                saturated / max(1, steps) <= max_saturation_fraction)
    return {"feasible": feasible, "max_lateral_m": max_y,
            "max_depth_m": max_z, "max_pitch_deg": max_pitch,
            "max_position_error_m": max_error,
            "saturation_fraction": saturated / max(1, steps),
            "terminal_progress_m": float(progress[0])}


@torch.no_grad()
def screen(env, candidates, steps=10, current_uncertainty=.3,
           gain_uncertainty=.1, corridor_y=8., corridor_z=5.,
           pitch_limit_deg=50., terminal_slack=10.,
           max_saturation_fraction=1.):
    """Evaluate each speed under nominal and four bounded model/current scenarios."""
    saved = controller_snapshot(env)
    rows = []
    scenario_offsets = [(0., 0., 0.), (current_uncertainty, 0., 0.),
                        (-current_uncertainty, 0., 0.),
                        (0., current_uncertainty, 0.),
                        (0., -current_uncertainty, 0.)]
    for speed in candidates:
        restore_controller(env, saved)
        env.progress_speed = float(speed)
        command, ok = env._solve_mpc()
        if not bool(ok[0]) or not bool(torch.isfinite(command).all()):
            rows.append({"speed_mps": float(speed), "feasible": False,
                         "solver_ok": False, "scenarios": []})
            continue
        plan = env.mpc.plan[0].clone()
        scenarios = []
        for index, offset in enumerate(scenario_offsets):
            # The screen uses only the filtered estimate and a declared bound.
            current = env.current_estimate + env.model.tensor(offset)[None]
            scale = [1., 1., 1., 1., 1.]
            if index:
                scale[3] = scale[4] = 1. - gain_uncertainty
            scenarios.append(_rollout(env, plan, current, scale, steps,
                                      corridor_y, corridor_z, pitch_limit_deg,
                                      terminal_slack, max_saturation_fraction))
        rows.append({"speed_mps": float(speed),
                     "feasible": all(s["feasible"] for s in scenarios),
                     "solver_ok": True, "scenarios": scenarios,
                     "risk_cost": max(s["max_position_error_m"] for s in scenarios),
                     "saturation_cost": max(s["saturation_fraction"] for s in scenarios)})
    restore_controller(env, saved)
    return rows


def choose(rows, rule, target):
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        return None
    if rule == "fastest":
        return max(feasible, key=lambda row: row["speed_mps"])
    if rule == "risk":
        return min(feasible, key=lambda row: (row["risk_cost"],
                                             -row["speed_mps"]))
    if rule == "balanced":
        return min(feasible, key=lambda row: (
            row["risk_cost"] + 2 * row["saturation_cost"] +
            .15 * abs(row["speed_mps"] - target), -row["speed_mps"]))
    raise ValueError(rule)


@torch.no_grad()
def run_episode(file, target, seed, args, rule):
    bank = PathBank([file], args.device)
    bank.speed[:] = target
    bank.first_state[:, 5] = target
    bank.durations[:] = bank.lengths / target
    env = ProgressEnv(bank, batch_size=1, device=args.device, horizon=args.horizon,
                      backend=args.backend, seed=seed,
                      randomize_plant=args.randomize_plant,
                      current_max_mps=args.current_max,
                      current_profile=args.current_profile,
                      admm_iterations=400, admm_rho=10.,
                      attitude_pitch_weight=100., attitude_yaw_weight=30.,
                      fsp_enabled=True, preview_mode="multiscale",
                      yaw_reward="course",
                      rti_constraints=(args.corridor_y, args.corridor_z,
                                       args.pitch_limit_deg, args.max_normalized_slew)
                      if args.constrained_rti else None)
    env.progress_speed = float(args.fixed_reference_speed or target)
    env.reset(torch.zeros(1, device=args.device, dtype=torch.long))
    candidates = [float(x) for x in args.candidates if args.task_min <= x <= args.task_max]
    if not candidates:
        raise ValueError("no candidates inside task speed range")
    errors, speeds, cycle_ms, selections, components = [], [], [], [], []
    audit_pending = []
    audit_total = audit_violations = 0
    empty = 0
    sat = fallback = 0
    propulsion_work_j = 0.
    max_pitch_deg = 0.
    start = time.perf_counter()
    last_speed = target
    while True:
        tick = time.perf_counter()
        if rule in ("fastest", "risk", "balanced") and len(errors) % args.screen_every == 0:
            limited = [v for v in candidates if abs(v - last_speed) <= args.max_speed_change + 1e-9]
            if not limited:
                limited = [min(candidates, key=lambda x: abs(x - last_speed))]
            rows = screen(env, limited, steps=args.screen_steps,
                          current_uncertainty=args.current_uncertainty,
                          gain_uncertainty=args.gain_uncertainty,
                          corridor_y=args.corridor_y, corridor_z=args.corridor_z,
                          pitch_limit_deg=args.pitch_limit_deg,
                          terminal_slack=args.terminal_slack,
                          max_saturation_fraction=args.max_saturation_fraction)
            components.append(speed_components([r["speed_mps"] for r in rows],
                                               [r["feasible"] for r in rows]))
            selected = choose(rows, rule, target)
            if selected is None:
                empty += 1
                # An empty set is an explicit replan request; keep the prior
                # executable command until the environment reaches its limit.
            else:
                last_speed = selected["speed_mps"]
                env.progress_speed = last_speed
                selections.append(last_speed)
                audit_pending.append({"remaining": args.audit_steps,
                                      "violation": False})
                env.base_command, env.last_mpc_ok = env._solve_mpc()
        elif rule == "curv6":
            ahead = env.progress[:, None] + env.model.tensor([0., 10., 20., 30.])
            _, _, _, yaw, _ = env.paths.sample(env.path_ids, ahead)
            delta = torch.atan2(torch.sin(yaw[:, 1:] - yaw[:, :-1]),
                                torch.cos(yaw[:, 1:] - yaw[:, :-1]))
            curvature = float((delta.abs() / 10).amax())
            desired = min(target, args.task_max, np.sqrt(6 / max(curvature, 1e-6)))
            env.progress_speed = max(args.task_min, desired)
        else:
            env.progress_speed = float(args.fixed_reference_speed or target)
        _, _, done, info = env.step(torch.zeros(1, 3, device=args.device,
                                             dtype=env.dtype))
        cycle_ms.append((time.perf_counter() - tick) * 1000)
        errors.append(float(info["position_error_m"][0]))
        speeds.append(float(torch.linalg.vector_norm(env.state[0, 5:8])))
        fallback += int(not bool(info["mpc_ok"][0]))
        upper = env.model.upper * env.model.control_scale
        lower = env.model.lower * env.model.control_scale
        sat += int(bool(((info["command"] >= upper - 1e-6) |
                         (info["command"] <= lower + 1e-6)).any()))
        propulsion_work_j += max(0., float(info["command"][0, 0])) * max(
            0., float(env.state[0, 5])) * env.mpc.dt
        max_pitch_deg = max(max_pitch_deg,
                            abs(float(env.state[0, 3])) * 180 / np.pi)
        _, lateral, depth, _, _, _, _ = env._errors()
        observed_violation = (abs(float(lateral[0])) > args.corridor_y or
                              abs(float(depth[0])) > args.corridor_z or
                              abs(float(env.state[0, 3])) * 180 / np.pi > args.pitch_limit_deg or
                              errors[-1] > args.terminal_slack or bool(info["failed"][0]))
        for audit in audit_pending:
            audit["violation"] |= observed_violation
            audit["remaining"] -= 1
        complete = [audit for audit in audit_pending if audit["remaining"] <= 0]
        audit_total += len(complete)
        audit_violations += sum(audit["violation"] for audit in complete)
        audit_pending = [audit for audit in audit_pending if audit["remaining"] > 0]
        if done[0]:
            break
        if len(errors) >= args.max_cycles:
            break
    success = bool(info["success"][0])
    speed_band_fraction = float(np.mean(
        np.abs(np.asarray(speeds) - target) <= args.task_speed_tolerance))
    return {"path": file.stem, "target_speed_mps": target, "seed": seed,
            "rule": rule, "backend": args.backend, "success": success,
            "task_speed_band_fraction": speed_band_fraction,
            "task_success": success and speed_band_fraction >= .95,
            "failed": bool(info["failed"][0]), "timeout": bool(info["timeout"][0]),
            "truncated": len(errors) >= args.max_cycles and not bool(done[0]),
            "steps": len(errors), "progress_fraction": float(info["progress_fraction"][0]),
            "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
            "position_p95_m": float(np.quantile(errors, .95)),
            "speed_rmse_mps": float(np.sqrt(np.mean((np.array(speeds) - target)**2))),
            "propulsion_work_proxy_j": propulsion_work_j,
            "max_abs_pitch_deg": max_pitch_deg,
            "saturation_fraction": sat / len(errors), "fallback_fraction": fallback / len(errors),
            "empty_set_count": empty,
            "audit_horizon_cycles": args.audit_steps,
            "audited_feasible_count": audit_total,
            "audited_violation_count": audit_violations,
            "selected_speeds_mps": selections, "feasible_components_mps": components,
            "control_cycle_p95_ms": float(np.quantile(cycle_ms, .95)),
            "control_cycle_max_ms": max(cycle_ms), "wall_s": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", type=Path,
                        default=sorted((ROOT / "DDAT_trajectories/val").glob("*.csv")))
    parser.add_argument("--targets", nargs="+", type=float, default=[6., 18.])
    parser.add_argument("--fixed-reference-speed", type=float,
                        help="reference only; the task target and speed RMSE stay unchanged")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101026])
    parser.add_argument("--rules", nargs="+", choices=("fixed", "curv6", "fastest", "risk", "balanced"),
                        default=["fixed", "curv6", "fastest", "risk", "balanced"])
    parser.add_argument("--backend", choices=("osqp", "rti_cpu", "rti_cuda"), default="osqp")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--constrained-rti", action="store_true")
    parser.add_argument("--max-normalized-slew", type=float, default=.3)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--candidates", nargs="+", type=float, default=[6., 9., 12., 15., 18.])
    parser.add_argument("--task-min", type=float, default=6.)
    parser.add_argument("--task-max", type=float, default=18.)
    parser.add_argument("--task-speed-tolerance", type=float, default=1.,
                        help="m/s band; task_success requires 95%% of cycles inside it")
    parser.add_argument("--max-speed-change", type=float, default=3.)
    parser.add_argument("--screen-every", type=int, default=10)
    parser.add_argument("--screen-steps", type=int, default=10)
    parser.add_argument("--audit-steps", type=int, default=25,
                        help="actual closed-loop cycles to check after a feasible selection")
    parser.add_argument("--corridor-y", type=float, default=8.)
    parser.add_argument("--corridor-z", type=float, default=5.)
    parser.add_argument("--pitch-limit-deg", type=float, default=50.)
    parser.add_argument("--terminal-slack", type=float, default=10.)
    parser.add_argument("--max-saturation-fraction", type=float, default=1.,
                        help="candidate rollout saturation cap; 1 disables this screen")
    parser.add_argument("--current-uncertainty", type=float, default=.3)
    parser.add_argument("--gain-uncertainty", type=float, default=.1)
    parser.add_argument("--current-max", type=float, default=1.)
    parser.add_argument("--current-profile", choices=("steady", "harmonic"), default="steady")
    parser.add_argument("--randomize-plant", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.constrained_rti and args.backend not in ("rti_cpu", "rti_cuda"):
        parser.error("--constrained-rti requires an RTI backend")
    if args.backend == "rti_cuda" and (args.device != "cuda" or
                                       not torch.cuda.is_available()):
        parser.error("rti_cuda requires --device cuda and a working CUDA device")
    if args.backend != "rti_cuda" and args.device != "cpu":
        parser.error("CPU backends require --device cpu")
    if (args.horizon < 1 or args.screen_steps < 1 or args.screen_every < 1 or
            args.audit_steps < 1 or
            args.max_cycles < 1 or args.task_min <= 0 or args.task_min > args.task_max or
            not 0 <= args.max_saturation_fraction <= 1 or
            any(v <= 0 for v in args.candidates)):
        parser.error("invalid horizon, speed range, candidates, or cycle count")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {name: ([str(p) for p in value] if name == "paths" else
                     str(value) if isinstance(value, Path) else value)
              for name, value in vars(args).items()}
    args.output.with_suffix(".config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output.open("w", encoding="utf-8") as stream:
        for file in args.paths:
            for target in args.targets:
                for seed in args.seeds:
                    for rule in args.rules:
                        row = run_episode(file, target, seed, args, rule)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        print(file.stem, target, seed, rule, row["success"],
                              row["progress_fraction"], row["wall_s"], flush=True)


if __name__ == "__main__":
    main()
