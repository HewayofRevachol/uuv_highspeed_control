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
                                EndpointExtendedCondensedEnv,
                                LookaheadOffsetEndpointEnv)
from simulator.batch_path_env import PathBank
from simulator.batch_mpc import CONTROL_SCALE
from simulator.model import LoopPath
from aav_mpc.h15_target_domain import install_target_domain
from simulator.rti_symbolic import AUGMENTED_NX, ActuatorAwareCpuDynamics

CONFIG = Path(__file__).parent / "configs/tasks.json"


def next_event(path, progress):
    s = path.distances
    curvature = np.gradient(path.yaw, s)
    change = np.abs(np.gradient(curvature, s))
    events = s[(change > 0.00015) | (np.abs(np.gradient(path.pitch, s)) > .0005)]
    later = events[events >= progress]
    return float(later[0] - progress) if len(later) else math.nan


def predict_accepted_step(mpc, state, actuator, queue, command):
    """Propagate one step under the controller's model and issued command.

    ``BatchedRTIMPC.states[:, 1]`` is the shifted linearization trajectory from
    before the QP solve; it is not the response to the command accepted after
    that solve. This helper rebuilds the observed 22-state input and evaluates
    the controller's discrete prediction model with the command actually sent
    to the plant during the next control period.
    """
    dynamics = getattr(mpc, "dynamics", None)
    if dynamics is None:
        dynamics = ActuatorAwareCpuDynamics(mpc.dt)
    if getattr(dynamics, "nx", None) != AUGMENTED_NX:
        return None
    scale = mpc.model.control_scale
    normalized = torch.cat((
        state,
        actuator / scale,
        queue[:3].reshape(-1) / scale.repeat(3),
    ), -1)
    augmented = dynamics.evaluate(
        normalized[None, None].numpy(),
        (command / scale)[None, None].numpy(),
    )[0][0, 0]
    return torch.as_tensor(augmented[:10], dtype=state.dtype, device=state.device)


def run(task, method, out, max_steps=None, randomize_plant=False,
        current_max=0., process_noise=False, nominal_body=False,
        path_override=None, speed_override=None, seed_override=None,
        speed_conditioned_cost=False, oracle_no_lag=False,
        horizon_override=None, preview_scale_override=None,
        qp_path_constraints=False, lookahead_offset_m=None,
        current_profile="steady", target_domain=False, target_components=None,
        qp_soft_path_penalty=None, prediction_fin_gain_scale=None,
        prediction_fin_tau_scale=None, target_fin_gain_scale=.9,
        target_fin_tau_scale=1., target_fin_delay_scale=1.,
        fallback_mode="hold", reference_intent_mps=None,
        log_substeps=False, profile_stages=False):
    definition = (dict(json.loads((CONFIG.parent / "h15.json").read_text()))
                  if task == "H15" else dict(json.loads(CONFIG.read_text())[task]))
    if path_override is not None:
        definition["path"] = str(path_override)
    if speed_override is not None:
        definition["target_speed_mps"] = float(speed_override)
    if seed_override is not None:
        definition["seed"] = int(seed_override)
    if "path" not in definition or "seed" not in definition:
        raise ValueError(f"{task} requires --path and --seed")
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
    if preview_scale_override is not None:
        if preview_scale_override <= 0:
            raise ValueError("preview scale must be positive")
        opts["preview_scale"] = float(preview_scale_override)
    backend = "rti_cpu" if method.startswith(("rti", "aware", "condensed")) else "osqp"
    horizon = (int(horizon_override) if horizon_override is not None else
               20 if method.endswith("20") else 35 if method.endswith("35") else
               40 if method.endswith("40") else 25)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    env_type = (EndpointExtendedCondensedEnv if method.startswith("condensed_endpoint") else
                CondensedActuatorEnv if method.startswith("condensed") else
                EndpointExtendedAwareEnv if method.startswith("aware_endpoint") else
                EndpointExtendedEnv if method.startswith(("endpoint", "rti_endpoint")) else
                ActuatorAwareEnv if method.startswith("aware") else ScheduledReferenceEnv)
    if lookahead_offset_m is not None:
        if not method.startswith("endpoint"):
            raise ValueError("fixed-speed lookahead offset currently supports endpoint OSQP")
        env_type = LookaheadOffsetEndpointEnv
        opts["lookahead_distance_m"] = float(lookahead_offset_m)
    if reference_intent_mps is not None:
        if method != "condensed_endpoint35":
            raise ValueError("speed-intent diagnostic requires condensed_endpoint35")
        from experiments.h15_mainline.progress_reference import ProgressReferenceEnv
        env_type = ProgressReferenceEnv
    env = env_type(bank, batch_size=1, device="cpu",
        horizon=horizon, backend=backend, seed=definition["seed"],
        randomize_plant=randomize_plant, current_max_mps=current_max,
        current_profile=current_profile,
        admm_iterations=400, admm_rho=10.,
        attitude_pitch_weight=100.*cost_scale, attitude_yaw_weight=30.*cost_scale,
        fsp_enabled=not method.startswith(("aware", "condensed")), preview_mode="multiscale", yaw_reward="course", **opts)
    if prediction_fin_gain_scale is not None or prediction_fin_tau_scale is not None:
        if not method.startswith("condensed_endpoint"):
            raise ValueError("prediction fin response diagnostic requires condensed endpoint MPC")
        from simulator.rti_symbolic import ActuatorAwareCpuDynamics
        env.mpc.dynamics = ActuatorAwareCpuDynamics(env.mpc.dt,
            fin_gain_scale=prediction_fin_gain_scale if prediction_fin_gain_scale is not None else 1.,
            fin_tau_scale=prediction_fin_tau_scale if prediction_fin_tau_scale is not None else 1.)
    if qp_path_constraints:
        if not method.startswith(("rti", "aware")):
            raise ValueError("path constraints are currently supported only by multiple-shooting RTI")
        env.mpc.path_constraints = (definition["lateral_corridor_m"],
                                    definition["depth_corridor_m"],
                                    definition["pitch_limit_deg"], .25)
        if qp_soft_path_penalty is not None:
            if not np.isfinite(qp_soft_path_penalty) or qp_soft_path_penalty <= 0:
                raise ValueError("soft path penalty must be positive and finite")
            env.mpc.soft_path_penalty = float(qp_soft_path_penalty)
    elif qp_soft_path_penalty is not None:
        raise ValueError("soft path penalty requires QP path constraints")
    if fallback_mode != "hold":
        if not method.startswith("condensed"):
            raise ValueError("fallback policy diagnostics require condensed MPC")
        env.mpc.fallback_mode = fallback_mode
    env.reset(torch.zeros(1, dtype=torch.long))
    if reference_intent_mps is not None:
        env.set_intent(reference_intent_mps)
    env.capture_strict_substeps = bool(log_substeps)
    env.profile_stages = bool(profile_stages)
    if not process_noise:
        env.disturbance_std.zero_()
    if nominal_body:
        env.parameters[:] = 1.
    if target_domain:
        if task != "H15":
            raise ValueError("target domain is defined only for H15")
        components = ("fin", "integrator", "actuator") if target_components is None else tuple(target_components)
        if any(part not in ("fin", "integrator", "actuator") for part in components):
            raise ValueError("unknown target-domain component")
        install_target_domain(env, fin_gain_scale=target_fin_gain_scale if "fin" in components else 1.,
                              integrator="integrator" in components,
                              second_order="actuator" in components,
                              fin_tau_scale=target_fin_tau_scale,
                              fin_delay_scale=target_fin_delay_scale)
    elif target_components is not None:
        raise ValueError("target components require target_domain=True")
    if oracle_no_lag:
        env.delay_steps.zero_()
        env.tau.zero_()
        env.fsp = None
        env.mpc.reset(env.state, torch.ones(1, dtype=torch.bool))
        env.base_command, env.last_mpc_ok = env._solve_mpc()
    path = LoopPath(file)
    rows = []
    substep_rows = []
    wall = time.perf_counter()
    while True:
        before = env.state[0].clone()
        actuator_before = env.actuator[0].clone()
        current_estimate_before = env.current_estimate[0].clone()
        command = env.base_command[0].clone()
        mpc_plan_commands = (env.mpc.plan[0].clone() *
                             torch.as_tensor(CONTROL_SCALE, device=env.device,
                                             dtype=env.dtype))
        queue = env.command_history[0].clone()
        progress_before = float(env.progress[0])
        solve_ok = bool(env.last_mpc_ok[0])
        solve_status = getattr(env.mpc, "last_status", ["unknown"])[0]
        nu_ref_before = env.reference_rate() if reference_intent_mps is not None else speed
        slack_before = env.mpc.last_slack[0].clone() if hasattr(env.mpc, "last_slack") else None
        legacy_predicted = env.mpc.states[0, 1].clone() if hasattr(env.mpc, "states") else None
        predicted = predict_accepted_step(env.mpc, before, actuator_before,
                                          queue, command)
        tic = time.perf_counter()
        _, _, done, info = env.step(torch.zeros(1, 3))
        if log_substeps:
            for sample in env.strict_substeps:
                substep_rows.append({
                    "period": len(rows), "substep": sample["substep"],
                    "time_s": len(rows) * env.mpc.dt + sample["substep"] * .02,
                    "state_json": json.dumps(sample["state"][0]),
                    "actuator_json": json.dumps(sample["actuator"][0]),
                    "queue_json": json.dumps(sample["queue"][0]),
                    "progress_m": sample["progress_m"][0],
                    "lateral_error_m": sample["lateral_error_m"][0],
                    "depth_error_m": sample["depth_error_m"][0],
                    "speed_error_mps": sample["speed_error_mps"][0],
                    "pitch_deg": math.degrees(sample["pitch_rad"][0]),
                })
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
            "mpc_plan_commands_json": json.dumps(mpc_plan_commands.tolist()),
            "state_after_json": json.dumps(x.tolist()),
            "current_ned_json": json.dumps(env.current_ned[0].tolist()),
            "progress_m": float(env.progress[0]),
            "distance_to_geometry_change_m": next_event(path, progress_before),
            "target_speed_mps": speed, "body_speed_mps": float(torch.linalg.vector_norm(x[5:8])),
            "nu_ref_mps": nu_ref_before,
            "actual_tangent_water_speed_mps": speed + float(speed_error[0]),
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
            "qp_slack_lateral_max_m": float(slack_before[:, 0].max()) if slack_before is not None else 0.,
            "qp_slack_depth_max_m": float(slack_before[:, 1].max()) if slack_before is not None else 0.,
            "qp_slack_pitch_max_deg": math.degrees(float(slack_before[:, 2].max())) if slack_before is not None else 0.,
            "primal_residual": float(getattr(env.mpc, "last_primal_residual", torch.zeros(1))[0]),
            "dual_residual": float(getattr(env.mpc, "last_dual_residual", torch.zeros(1))[0]),
            "control_period_wall_ms": solve_ms,
            "success": bool(info["success"][0]), "failed": bool(info["failed"][0]),
            "timeout": bool(info["timeout"][0])}
        if profile_stages:
            row.update(info["timing"])
        if legacy_predicted is not None:
            row["prediction_legacy_1_position_error_m"] = float(torch.linalg.vector_norm(
                legacy_predicted[:3] - x[:3]))
            row["prediction_legacy_1_velocity_error_mps"] = float(torch.linalg.vector_norm(
                legacy_predicted[5:8] - x[5:8]))
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
    if log_substeps:
        with out.with_name(out.stem + "_substeps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=substep_rows[0].keys())
            writer.writeheader(); writer.writerows(substep_rows)
    limits = {"speed": definition["speed_tolerance_mps"],
        "lateral": definition["lateral_corridor_m"], "depth": definition["depth_corridor_m"],
        "pitch": definition["pitch_limit_deg"]}
    first = []
    first_by_type = {}
    saturated = []
    limits_command = [1500000., 45., 45.]
    for row in rows:
        hit = (row["thrust_command_n"] >= limits_command[0] - .01 or
               row["thrust_command_n"] <= .01 or
               abs(row["rudder_command_deg"]) >= limits_command[1] - .001 or
               abs(row["elevator_command_deg"]) >= limits_command[2] - .001)
        saturated.append(hit)
        for key, magnitude in (("solver", not row["mpc_ok"]),
                 ("speed", abs(row["speed_error_mps"]) > limits["speed"]),
                 ("lateral", abs(row["lateral_error_m"]) > limits["lateral"]),
                 ("depth", abs(row["depth_error_m"]) > limits["depth"]),
                 ("pitch", abs(row["pitch_deg"]) > limits["pitch"]),
                 ("saturation", hit)):
            if magnitude:
                event = {"time_s": row["time_s"], "progress_m": row["progress_m"],
                         "distance_to_geometry_change_m": row["distance_to_geometry_change_m"]}
                first_by_type.setdefault(key, event)
                if key != "saturation":
                    first.append((row["time_s"], key))
    times = np.array([r["control_period_wall_ms"] for r in rows])
    substep_first = None
    for sample in substep_rows:
        for key, magnitude in (("speed", abs(sample["speed_error_mps"]) > limits["speed"]),
                               ("lateral", abs(sample["lateral_error_m"]) > limits["lateral"]),
                               ("depth", abs(sample["depth_error_m"]) > limits["depth"]),
                               ("pitch", abs(sample["pitch_deg"]) > limits["pitch"])):
            if magnitude and substep_first is None:
                substep_first = (sample["time_s"], key)
    metrics = {"max_abs_lateral_m": max(abs(r["lateral_error_m"]) for r in rows),
               "max_abs_depth_m": max(abs(r["depth_error_m"]) for r in rows),
               "max_abs_speed_error_mps": max(abs(r["speed_error_mps"]) for r in rows),
               "max_abs_pitch_deg": max(abs(r["pitch_deg"]) for r in rows),
               "position_rmse_m": float(np.sqrt(np.mean([
                   r["lateral_error_m"]**2 + r["depth_error_m"]**2 for r in rows]))),
               "one_step_position_prediction_rmse_m": float(np.sqrt(np.mean([
                   r["prediction_1_position_error_m"]**2 for r in rows
                   if "prediction_1_position_error_m" in r]))) if predicted is not None else None,
               "legacy_one_step_position_prediction_rmse_m": float(np.sqrt(np.mean([
                   r["prediction_legacy_1_position_error_m"]**2 for r in rows
                   if "prediction_legacy_1_position_error_m" in r])))
                   if legacy_predicted is not None else None,
               "control_period_p50_ms": float(np.percentile(times, 50)),
               "control_period_p95_ms": float(np.percentile(times, 95)),
               "control_period_p99_ms": float(np.percentile(times, 99))}
    if profile_stages:
        for field in ("plant_integration_wall_ms", "mpc_wall_ms", "step_wall_ms"):
            values = np.array([row[field] for row in rows])
            for percentile in (50, 95, 99):
                metrics[f"{field}_p{percentile}"] = float(np.percentile(values, percentile))
            metrics[f"{field}_max"] = float(values.max())
    for key in ("qp_slack_lateral_max_m", "qp_slack_depth_max_m", "qp_slack_pitch_max_deg"):
        metrics[key] = max(r[key] for r in rows)
    metrics["qp_slack_period_fraction"] = float(np.mean([
        r["qp_slack_lateral_max_m"] > 1e-6 or
        r["qp_slack_depth_max_m"] > 1e-6 or
        r["qp_slack_pitch_max_deg"] > 1e-6 for r in rows]))
    summary = {"task": task, "method": method, "path": definition["path"],
        "target_speed_mps": speed,
        "seed": definition["seed"], "backend": backend, "horizon": horizon,
        "preview_scale": opts.get("preview_scale", 1.),
        "lookahead_offset_m": lookahead_offset_m,
        "randomize_plant": randomize_plant, "current_max_mps": current_max,
        "current_profile": current_profile, "target_domain": target_domain,
        "target_components": list(components) if target_domain else [],
        "process_noise": process_noise, "nominal_body": nominal_body,
        "speed_conditioned_cost": speed_conditioned_cost,
        "oracle_no_lag": oracle_no_lag,
        "qp_path_constraints": qp_path_constraints,
        "qp_soft_path_penalty": qp_soft_path_penalty,
        "prediction_fin_gain_scale": prediction_fin_gain_scale,
        "prediction_fin_tau_scale": prediction_fin_tau_scale,
        "target_fin_gain_scale": target_fin_gain_scale,
        "target_fin_tau_scale": target_fin_tau_scale,
        "target_fin_delay_scale": target_fin_delay_scale,
        "fallback_mode": fallback_mode,
        "reference_intent_mps": reference_intent_mps,
        "log_substeps": log_substeps,
        "profile_stages": profile_stages,
        "steps": len(rows), "wall_s": time.perf_counter() - wall,
        "progress_fraction": float(env.progress[0] / bank.lengths[0]),
        "simulator_success": rows[-1]["success"], "simulator_failed": rows[-1]["failed"],
        "strict_success": rows[-1]["success"] and not first,
        "strict_hist": rows[-1]["success"] and not first,
        "strict_substep_success": (rows[-1]["success"] and substep_first is None)
            if log_substeps else None,
        "strict_mainline_provisional": (rows[-1]["success"] and substep_first is None
            and not first and all(t <= 100. for t in times)) if log_substeps else None,
        "first_substep_violation": substep_first,
        "first_anomaly": min(first) if first else None,
        "first_by_type": first_by_type,
        "fallback_count": sum(not r["mpc_ok"] for r in rows),
        "saturation_fraction": float(np.mean(saturated)),
        **metrics,
        "trace": str(out.resolve().relative_to(ROOT))}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("T1", "T3", "H15"), required=True)
    parser.add_argument("--method", choices=("osqp25", "osqp40", "rti25", "rti40",
        "preview15", "aware25", "aware40", "endpoint40", "aware_endpoint40",
        "condensed25", "condensed40", "condensed_endpoint25", "condensed_endpoint40",
        "rti_endpoint25", "rti_endpoint40",
        "endpoint20", "condensed_endpoint20", "condensed_endpoint35",
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
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--preview-scale", type=float)
    parser.add_argument("--qp-path-constraints", action="store_true")
    parser.add_argument("--qp-soft-path-penalty", type=float)
    parser.add_argument("--lookahead-offset-m", type=float)
    parser.add_argument("--current-profile", choices=("steady", "harmonic"),
                        default="steady")
    parser.add_argument("--target-domain", action="store_true")
    parser.add_argument("--target-components", nargs="+",
                        choices=("fin", "integrator", "actuator"))
    parser.add_argument("--prediction-fin-gain-scale", type=float)
    parser.add_argument("--prediction-fin-tau-scale", type=float)
    parser.add_argument("--target-fin-gain-scale", type=float, default=.9)
    parser.add_argument("--target-fin-tau-scale", type=float, default=1.)
    parser.add_argument("--target-fin-delay-scale", type=float, default=1.)
    parser.add_argument("--fallback-mode", choices=("hold", "center_fins"),
                        default="hold")
    parser.add_argument("--reference-intent", type=float)
    parser.add_argument("--log-substeps", action="store_true")
    args = parser.parse_args()
    run(args.task, args.method, args.out, args.max_steps, args.randomize_plant,
        args.current_max, args.process_noise, args.nominal_body,
        args.path, args.speed, args.seed, args.speed_conditioned_cost,
        args.oracle_no_lag, args.horizon, args.preview_scale,
        args.qp_path_constraints, args.lookahead_offset_m,
        args.current_profile, args.target_domain, args.target_components,
        args.qp_soft_path_penalty, args.prediction_fin_gain_scale,
        args.prediction_fin_tau_scale, args.target_fin_gain_scale,
        args.target_fin_tau_scale, args.target_fin_delay_scale,
        args.fallback_mode, args.reference_intent, args.log_substeps)
