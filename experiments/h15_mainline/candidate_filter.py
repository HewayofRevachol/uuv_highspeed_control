"""Offline candidate-specific H15 MPC solve and nonlinear replay screen.

Finite scenarios and dev residual margins make this an empirical diagnostic.
An online candidate is accepted only when the entire screen fits the frozen
budget; the current N35 implementation is expected to fail that budget.
"""

from dataclasses import replace
import copy
import math
import time

import numpy as np

from simulator.batch_mpc import CONTROL_SCALE
from simulator.model import MISMATCHED_PLANT, rk4_step, rotation_ned


SCALE = np.asarray(CONTROL_SCALE, dtype=float)
LOWER = np.asarray([0., -1., -1.]) * SCALE
UPPER = np.asarray([1., 1., 1.]) * SCALE
PHYSICAL_FIELDS = ("lateral_m", "depth_m", "pitch_deg",
                   "tangent_water_speed_error_mps")
RADIUS_FIELDS = ("lateral_m", "depth_m", "pitch_deg",
                 "tangent_water_speed_mps")


def _radius_for_period(calibration, period):
    horizon = next(h for h in (1, 5, 25, 35) if period <= h)
    return calibration[str(horizon)]


def _scenarios(estimate, bounds):
    gain_lo = max(bounds["gain_bounds"][0], estimate.fin_gain-estimate.gain_radius)
    gain_hi = min(bounds["gain_bounds"][1], estimate.fin_gain+estimate.gain_radius)
    tau_lo = max(bounds["tau_scale_bounds"][0], estimate.fin_tau_scale-estimate.tau_radius)
    tau_hi = min(bounds["tau_scale_bounds"][1], estimate.fin_tau_scale+estimate.tau_radius)
    return [("point", estimate.fin_gain, estimate.fin_tau_scale)] + [
        (f"corner_{g}_{t}", gain, tau)
        for g, gain in (("lo", gain_lo), ("hi", gain_hi))
        for t, tau in (("lo", tau_lo), ("hi", tau_hi))]


def _physical_values(state, path):
    tangent = path.tangent_at(path.progress)
    reference_position = path.position_at(path.progress)
    normal = np.asarray([-tangent[1], tangent[0], 0.])
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    lateral = float(normal @ (state[:3]-reference_position))
    depth = float(state[2]-reference_position[2])
    pitch = math.degrees(float(state[3]))
    tangent_speed = float(tangent @ (rotation_ned(state[3], state[4]) @ state[5:8]))
    return (lateral, depth, pitch, tangent_speed-15.)


def replay_plan(state, actuator, queue, commands, progress, path,
                gain, tau_scale, calibration_radii, config):
    """Replay accepted physical commands from measured state/queue at 0.02 s."""
    state = np.asarray(state, dtype=float).copy()
    actuator = np.asarray(actuator, dtype=float).copy()
    queue = np.asarray(queue, dtype=float)[:3].copy()
    if state.shape != (10,) or actuator.shape != (3,) or queue.shape != (3, 3):
        raise ValueError("invalid measured replay initial state")
    if len(commands) != config["horizon_periods"]:
        raise ValueError("candidate plan does not span N35")
    local_path = copy.copy(path)
    local_path.progress = float(progress)
    plant = replace(MISMATCHED_PLANT,
                    elevator_gain=MISMATCHED_PLANT.elevator_gain*gain,
                    rudder_gain=MISMATCHED_PLANT.rudder_gain*gain)
    delays = (3, 3, 3)  # declared H15 mismatch model, not target-domain truth
    taus = (MISMATCHED_PLANT.thrust_tau_s,
            MISMATCHED_PLANT.fin_tau_s*tau_scale,
            MISMATCHED_PLANT.fin_tau_s*tau_scale)
    alpha = np.asarray([-math.expm1(-.02/tau) if tau else 1. for tau in taus])
    min_margins = {field: math.inf for field in PHYSICAL_FIELDS}
    maxima = {field: 0. for field in PHYSICAL_FIELDS}
    normalized_occupancy = 0.
    first_rejection = None
    first_time_s = None
    period_errors = []
    first_step_state = None
    first_step_actuator = None
    first_step_queue = None
    start_progress = local_path.progress
    if not np.isfinite(np.concatenate((state, actuator, queue.ravel()))).all():
        raise ValueError("nonfinite measured replay initial state")
    if (queue < LOWER-1e-8).any() or (queue > UPPER+1e-8).any():
        first_rejection, first_time_s = "queue_box", 0.
    elif (actuator < LOWER-1e-8).any() or (actuator > UPPER+1e-8).any():
        first_rejection, first_time_s = "actual_actuator_box", 0.
    for field, value in zip(PHYSICAL_FIELDS, _physical_values(state, local_path)):
        if abs(value) > config["strict_limits"][field] and first_rejection is None:
            first_rejection, first_time_s = field, 0.
    for period, command in enumerate(commands, 1):
        command = np.asarray(command, dtype=float)
        for substep in range(1, 6):
            target = np.asarray([queue[d-1, j] if d else command[j]
                                 for j, d in enumerate(delays)])
            queue = np.vstack((command, queue[:-1]))
            actuator += alpha*(target-actuator)
            state = rk4_step(state, actuator, plant_params=plant)
            if not np.isfinite(np.concatenate((state, actuator, queue.ravel()))).all():
                return {"accepted": False, "first_rejection": "nonfinite_replay",
                        "first_time_s": (period-1)*.1+substep*.02,
                        "min_margins": min_margins, "maxima": maxima,
                        "risk": math.inf, "progress_delta_m": local_path.progress-start_progress,
                        "period_errors": period_errors,
                        "first_step_state": first_step_state,
                        "first_step_actuator": first_step_actuator,
                        "first_step_queue": first_step_queue}
            actuator_margin = min(float((actuator-LOWER).min()),
                                  float((UPPER-actuator).min()))
            if actuator_margin < -1e-8 and first_rejection is None:
                first_rejection = "actual_actuator_box"
                first_time_s = (period-1)*.1+substep*.02
            local_path.advance(state[:3])
            values = _physical_values(state, local_path)
            radii = _radius_for_period(calibration_radii, period)
            for field, radius_field, value in zip(PHYSICAL_FIELDS, RADIUS_FIELDS, values):
                limit = config["strict_limits"][field]
                radius = radii[radius_field]
                margin = limit-abs(value)-radius
                min_margins[field] = min(min_margins[field], float(margin))
                maxima[field] = max(maxima[field], abs(value))
                normalized_occupancy = max(normalized_occupancy,
                                           (abs(value)+radius)/limit)
                if margin < 0 and first_rejection is None:
                    first_rejection = field
                    first_time_s = (period-1)*.1+substep*.02
            if substep == 5:
                if period == 1:
                    first_step_state = state.tolist()
                    first_step_actuator = actuator.tolist()
                    first_step_queue = queue.tolist()
                period_errors.append(values)
                if period == config["horizon_periods"]:
                    for field, radius_field, value in zip(PHYSICAL_FIELDS,
                                                          RADIUS_FIELDS, values):
                        margin = (config["strict_limits"][field]-abs(value)-
                                  radii[radius_field]-
                                  config["terminal_extra_margin"][field])
                        if margin < 0 and first_rejection is None:
                            first_rejection = "terminal_" + field
                            first_time_s = period*.1
    return {"accepted": first_rejection is None,
            "first_rejection": first_rejection, "first_time_s": first_time_s,
            "min_margins": min_margins, "maxima": maxima,
            "risk": float(normalized_occupancy),
            "progress_delta_m": float(local_path.progress-start_progress),
            "period_errors": period_errors,
            "first_step_state": first_step_state,
            "first_step_actuator": first_step_actuator,
            "first_step_queue": first_step_queue}


def screen_candidate(env, path, candidate, estimate, config, bounds):
    """Solve one candidate's N35 QP, then replay its accepted command plan."""
    if estimate.constraint_radii is None:
        raise ValueError("candidate screen requires frozen dev residual radii")
    started = time.perf_counter()
    prior_rate = env.reference_rate()
    old_intent = getattr(env, "intent_mps", 15.)
    old_plan = env.mpc.plan.clone()
    old_previous = env.mpc.previous.clone()
    old_states = env.mpc.states.clone()
    old_valid = env.mpc.last_qp_valid.clone()
    old_primal = env.mpc.last_primal_residual.clone()
    old_dual = env.mpc.last_dual_residual.clone()
    old_status = list(env.mpc.last_status)
    old_reference_step = env._reference_step
    old_reference_value = env._reference_value
    try:
        env.set_intent(candidate)
        reference = env._reference()
        first_rate = prior_rate + max(-.05, min(.05, candidate-prior_rate))
        tic = time.perf_counter()
        _, valid = env.mpc.solve(env.observed_augmented_state(), reference)
        qp_ms = (time.perf_counter()-tic)*1000.
        qp_status = env.mpc.last_status[0]
        if not bool(valid[0]):
            total_ms = (time.perf_counter()-started)*1000.
            return {"candidate_mps": candidate, "first_ramped_mps": first_rate,
                    "qp_valid": False, "qp_status": qp_status, "qp_ms": qp_ms,
                    "replay_ms": 0., "total_ms": total_ms,
                    "physical_accepted": False, "online_accepted": False,
                    "first_rejection": "qp_invalid", "scenarios": []}
        plan_normalized = env.mpc.plan[0].numpy().copy()
        commands = plan_normalized*SCALE
        prior = env.command_history[0, 0].numpy()
        deltas = np.abs(np.diff(np.vstack((prior, commands)), axis=0))/SCALE
        if not np.isfinite(commands).all() or (commands < LOWER-1e-8).any() or (commands > UPPER+1e-8).any():
            command_reason = "command_box"
        elif (deltas > np.asarray(config["command_max_delta_normalized_per_period"])+1e-12).any():
            command_reason = "command_rate"
        else:
            command_reason = None
        scenario_results = []
        tic = time.perf_counter()
        if command_reason is None:
            for name, gain, tau in _scenarios(estimate, bounds):
                result = replay_plan(
                    env.state[0].numpy(), env.actuator[0].numpy(),
                    env.command_history[0].numpy(), commands,
                    float(env.progress[0]), path, gain, tau,
                    estimate.constraint_radii, config)
                result["scenario"] = name
                result["fin_gain"] = gain
                result["fin_tau_scale"] = tau
                scenario_results.append(result)
        replay_ms = (time.perf_counter()-tic)*1000.
        physical_accepted = command_reason is None and all(
            item["accepted"] for item in scenario_results)
        first_physical = command_reason or next(
            (item["first_rejection"] for item in scenario_results
             if not item["accepted"]), None)
        total_ms = (time.perf_counter()-started)*1000.
        online_accepted = physical_accepted and total_ms <= config["screen_budget_ms"]
        point = scenario_results[0] if scenario_results else None
        cost = None
        if point is not None:
            errors = np.asarray(point["period_errors"])
            if len(errors) == config["horizon_periods"]:
                normalized = errors/np.asarray([8., 5., 40., 2.])
                risk = max(item["risk"] for item in scenario_results)
                cost = (-point["progress_delta_m"]/(15*config["horizon_periods"]*.1)
                        + 2*float(np.mean(normalized[:, 0]**2+normalized[:, 1]**2))
                        + float(np.mean(normalized[:, 2]**2+normalized[:, 3]**2))
                        + 5*risk)
        return {"candidate_mps": candidate, "first_ramped_mps": first_rate,
                "qp_valid": True, "qp_status": qp_status, "qp_ms": qp_ms,
                "replay_ms": replay_ms, "total_ms": total_ms,
                "plan_normalized": plan_normalized,
                "first_command": commands[0].tolist(),
                "physical_accepted": physical_accepted,
                "online_accepted": online_accepted,
                "first_physical_rejection": first_physical,
                "first_rejection": first_physical if first_physical else (
                    "screen_budget" if not online_accepted else None),
                "cost": cost, "scenarios": scenario_results}
    finally:
        env.intent_mps = old_intent
        env.mpc.plan[:] = old_plan
        env.mpc.previous[:] = old_previous
        env.mpc.states[:] = old_states
        env.mpc.last_qp_valid[:] = old_valid
        env.mpc.last_primal_residual[:] = old_primal
        env.mpc.last_dual_residual[:] = old_dual
        env.mpc.last_status[:] = old_status
        env._reference_step = old_reference_step
        env._reference_value = old_reference_value


def screen_one_step_backup(env, path, candidate, estimate, config, bounds):
    """Check the fixed-15 candidate's first command as a logged short backup."""
    if not candidate["qp_valid"] or "first_command" not in candidate:
        return {"accepted": False, "reason": "backup_qp_invalid", "scenarios": []}
    command = np.asarray(candidate["first_command"], dtype=float)
    prior = env.command_history[0, 0].numpy()
    if (not np.isfinite(command).all() or (command < LOWER-1e-8).any() or
            (command > UPPER+1e-8).any()):
        return {"accepted": False, "reason": "backup_command_box", "scenarios": []}
    if (np.abs(command-prior)/SCALE >
            np.asarray(config["command_max_delta_normalized_per_period"])+1e-12).any():
        return {"accepted": False, "reason": "backup_command_rate", "scenarios": []}
    one_step_config = dict(config, horizon_periods=1)
    outcomes = []
    for name, gain, tau in _scenarios(estimate, bounds):
        result = replay_plan(
            env.state[0].numpy(), env.actuator[0].numpy(),
            env.command_history[0].numpy(), [command], float(env.progress[0]),
            path, gain, tau, estimate.constraint_radii, one_step_config)
        outcomes.append({"scenario": name, "accepted": result["accepted"],
                         "first_rejection": result["first_rejection"],
                         "min_margins": result["min_margins"]})
    reason = next((item["first_rejection"] for item in outcomes
                   if not item["accepted"]), None)
    return {"accepted": reason is None, "reason": reason, "scenarios": outcomes}


def select_b2(candidates, online=False):
    """Frozen short-window cost selector; ties prefer 15 then slower intent."""
    key = "online_accepted" if online else "physical_accepted"
    accepted = [item for item in candidates if item[key] and item["cost"] is not None]
    if not accepted:
        return None
    return min(accepted, key=lambda item: (item["cost"],
                                           abs(item["candidate_mps"]-15.),
                                           item["candidate_mps"]))["candidate_mps"]
