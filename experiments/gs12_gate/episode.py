"""One causal episode implementation shared by all GS-12 methods."""
from __future__ import annotations

import json
from pathlib import Path
from collections import deque
import hashlib
import time

import numpy as np
import torch

from experiments.gs12_gate.data_io import digest, file_hash, read_json, write_npz, immutable_json
from experiments.gs12_gate.features import (CausalResidual, ObservationHistory,
                                             FRAME_SCALE, fast_reward, failure_cost, schema)
from experiments.gs12_gate.model_bridge import (HybridMPC, THETA_SAFE,
                                                 ThrustRateObserver, bounded_gate,
                                                 mix_gate)
from experiments.gs12_gate.plant_harness import PlantHarness
from experiments.h15_mainline.observation_features import FEATURE_NAMES, from_period_row
from experiments.h15_mainline.response_estimator import SlidingWindowResponseEstimator
from simulator.batch_mpc import BatchedUUVModel, CONTROL_SCALE
from simulator.model import rotation_ned


def bootstrap_estimator_normalization():
    return {"split": "train", "feature_names": list(FEATURE_NAMES),
            "mean": [0.]*len(FEATURE_NAMES),
            "scale": [8., 5., 2., 40., 1., 1., 1., 1., 1., 1., 1., 1., 1.]}


def estimator_parameters(bootstrap=True):
    p = dict(SlidingWindowResponseEstimator.DEFAULTS)
    p.update(window=10, gain_bounds=[.7, 1.2], tau_scale_bounds=[.5, 2.5],
             e0_gain=.85, e0_tau_scale=2.)
    if bootstrap:
        p["max_abs_normalized_feature"] = 1e12
    return p


def _error(measurement):
    state, ref = measurement["state"], measurement["reference"]
    inertial_velocity = rotation_ned(state[3], state[4])@state[5:8]
    return np.array([state[1]-ref[1], state[2]-ref[2],
                     inertial_velocity[0]-ref[5], state[0]-ref[0]]), inertial_velocity


def _estimator_row(before, after, issued, plan, errors):
    state0, state1 = before["state"], after["state"]
    reference = after["reference"]
    velocity = rotation_ned(state1[3], state1[4])@state1[5:8]
    tangent = reference[5:8]/max(np.linalg.norm(reference[5:8]), 1e-12)
    tangent_speed_error = float(velocity@tangent-np.linalg.norm(reference[5:8]))
    north_error = state1[0]-reference[0]
    lateral_error = -np.sin(reference[4])*north_error+np.cos(reference[4])*errors[0]
    return {"time_s": before["t"],
            "state_before_json": json.dumps(state0.tolist()),
            "state_after_json": json.dumps(state1.tolist()),
            "actuator_before_json": json.dumps(before["actual"].tolist()),
            "queue_before_json": json.dumps(before["queue"].tolist()),
            "command_json": json.dumps(issued.tolist()),
            "mpc_base_command_json": json.dumps(plan[0].tolist()),
            "mpc_plan_commands_json": json.dumps(plan.tolist()),
            "thrust_actual_n": float(after["actual"][0]),
            "elevator_actual_deg": float(np.rad2deg(after["actual"][1])),
            "rudder_actual_deg": float(np.rad2deg(after["actual"][2])),
            "lateral_error_m": float(lateral_error),
            "depth_error_m": float(errors[1]),
            "speed_error_mps": tangent_speed_error,
            "pitch_deg": float(np.rad2deg(state1[3]))}


def _make_observation(history, measurement, last_issued, innovation,
                      theta, radius, alpha, qp_ok, qp_ms):
    errors, velocity = _error(measurement)
    state, actual = measurement["state"], measurement["actual"]
    frame = np.r_[errors, state[3], state[8:10], actual, last_issued, innovation]
    history.append(frame)
    ref = measurement["reference"]
    margins = np.minimum((actual-np.array([0., -CONTROL_SCALE[1], -CONTROL_SCALE[2]]))/CONTROL_SCALE,
                         (np.asarray(CONTROL_SCALE)-actual)/CONTROL_SCALE)
    current = np.r_[theta, radius, alpha, margins, float(qp_ok), qp_ms,
                    measurement["preview_0p5"][[1, 2, 5]]-ref[[1, 2, 5]],
                    measurement["preview_1p0"][[1, 2, 5]]-ref[[1, 2, 5]], 0.]
    return history.observe(current), errors, velocity


def _rule_target(shadow_history, source, previous_alpha, threshold=.1):
    if source != "E1" or len(shadow_history) < 10:
        return np.zeros(4)
    errors0 = np.sum([row[0] for row in shadow_history], axis=0)
    errors1 = np.sum([row[1] for row in shadow_history], axis=0)
    ratio = errors1/np.maximum(errors0, 1e-12)
    return np.where(ratio <= 1-threshold, .8,
                    np.where(ratio >= 1+threshold, .2, previous_alpha))


def _shadow_errors(dynamics, z, issued_normalized, nu, theta_hat, theta_used, residual):
    """Score each branch against the model used to define its correction."""
    def predict(theta):
        return dynamics.evaluate(z[None, None], issued_normalized[None, None],
                                 theta)[0][0, 0, 5:10]

    safe, estimated, applied = map(predict, (THETA_SAFE, theta_hat, theta_used))
    groups = (np.arange(5), [0], [2, 3], [1, 4])
    bases = (safe, applied, applied, applied)
    corrected = [estimated]
    for indices in groups[1:]:
        candidate = applied.copy()
        candidate[indices] += .1*residual[indices]
        corrected.append(candidate)
    scale = FRAME_SCALE[-5:]
    base_errors = np.array([np.linalg.norm(((nu-pred)/scale)[indices])
                            for pred, indices in zip(bases, groups)])
    corrected_errors = np.array([np.linalg.norm(((nu-pred)/scale)[indices])
                                 for pred, indices in zip(corrected, groups)])
    return base_errors, corrected_errors


def run_episode(case, method, out, *, artifacts=None, policy=None,
                random_seed=None, max_seconds=None, save_substeps=True,
                snapshot_path=None, snapshot_out=None, freeze_gate=False,
                ablation=None, gate_phase_steps=0):
    out = Path(out)
    snapshot_hash = file_hash(snapshot_path) if snapshot_path is not None else None
    policy_hash = None
    if policy is not None:
        h = hashlib.sha256()
        for name, value in policy.state_dict().items():
            h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
        policy_hash = h.hexdigest()
    signature = digest({"case": case, "method": method, "artifacts": artifacts,
                        "policy_hash": policy_hash, "random_seed": random_seed,
                        "max_seconds": max_seconds, "save_substeps": save_substeps,
                        "snapshot_hash": snapshot_hash, "snapshot_out": bool(snapshot_out),
                        "freeze_gate": freeze_gate, "ablation": ablation,
                        "gate_phase_steps": gate_phase_steps})
    if ablation not in (None, "freeze4", "no_theta"):
        raise ValueError("unsupported inference ablation")
    if gate_phase_steps not in range(5):
        raise ValueError("gate phase must be 0..4 control steps")
    if out.exists():
        summary = out/"episode.json"
        if summary.exists():
            prior = read_json(summary)
            if prior.get("signature") == signature:
                for name, expected in prior.get("files", {}).items():
                    file = out/name
                    if not file.exists() or file_hash(file) != expected:
                        raise ValueError(f"cached episode artifact hash mismatch: {file}")
                return prior
        raise FileExistsError(out)
    out.mkdir(parents=True)
    immutable_json(out/"case.json", case)
    torch.set_num_threads(1)
    rng = np.random.default_rng(case["seed"] if random_seed is None else random_seed)
    plant = PlantHarness(case)
    model = BatchedUUVModel("cpu")
    mpc = HybridMPC(model)
    trim_normalized = plant.trim/CONTROL_SCALE
    mpc.plan[:] = model.tensor(np.tile(trim_normalized, (35, 1)))
    mpc.previous[:] = model.tensor(trim_normalized)
    thrust_observer = ThrustRateObserver()
    thrust_observer.update(trim_normalized[0])
    residual = CausalResidual(mpc.dynamics,
        limits=(artifacts or {}).get("residual_limits", [1., 1., 1., .1, .1]))
    shadow_history = deque(maxlen=10)
    history = ObservationHistory((artifacts or {}).get("observation_scale"))
    normalization = (artifacts or {}).get("estimator_normalization", bootstrap_estimator_normalization())
    bootstrap = (artifacts or {}).get("bootstrap", True)
    estimator = SlidingWindowResponseEstimator(normalization, estimator_parameters(bootstrap))
    alpha, theta, radius = np.zeros(4), THETA_SAFE.copy(), np.array([.35, 2.])
    last_issued = plant.trim.copy(); last_qp_ok = True; last_qp_ms = 0.
    completed_row = None
    locked_theta, locked_d = None, None
    if snapshot_path is not None:
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        if snapshot["policy_hash"] != policy_hash:
            raise ValueError("snapshot policy differs from requested policy")
        plant.state = np.asarray(snapshot["plant_state"], float).copy()
        plant.actual = np.asarray(snapshot["plant_actual"], float).copy()
        plant.rate = np.asarray(snapshot["plant_rate"], float).copy()
        plant.queue = np.asarray(snapshot["plant_queue"], float).copy()
        plant.t = float(snapshot["plant_t"])
        mpc.plan[:] = model.tensor(snapshot["mpc_plan"])
        mpc.previous[:] = model.tensor(snapshot["mpc_previous"])
        thrust_observer.previous = snapshot["observer_previous"]
        thrust_observer.rate = snapshot["observer_rate"]
        residual.history.extend(snapshot["residual_history"])
        history.frames.extend(snapshot["observation_history"])
        estimator.history.extend(snapshot["estimator_history"])
        estimator.last = snapshot["estimator_last"]
        alpha, theta, radius = (np.asarray(snapshot[key], float).copy()
                                for key in ("alpha", "theta", "radius"))
        last_issued = np.asarray(snapshot["last_issued"], float).copy()
        last_qp_ok, last_qp_ms = snapshot["last_qp_ok"], snapshot["last_qp_ms"]
        locked_theta = np.asarray(snapshot["theta_used"], float)
        locked_d = np.asarray(snapshot["d_used"], float)
    start_t = plant.t
    control, decisions, substeps = [], [], []
    first_failure, failure_reason = None, None
    planned = float(case["duration_s"])
    executed_limit = min(planned, max_seconds if max_seconds is not None else planned)
    planned_steps = int(round(executed_limit/.1))
    started = time.perf_counter()
    for k in range(planned_steps):
        period_started = time.perf_counter()
        before = plant.measurement()
        estimator_started = time.perf_counter()
        if completed_row is not None and not freeze_gate:
            estimate = estimator.update(completed_row)
            theta = np.array([estimate.fin_gain, estimate.fin_tau_scale])
            radius = np.array([estimate.gain_radius, estimate.tau_radius])
            source = estimate.source
            fallback_reason = estimate.fallback_reason or ""
        else:
            source = "E0"
            fallback_reason = "short_history" if not freeze_gate else "locked"
        estimator_ms = 1000*(time.perf_counter()-estimator_started)
        feature_started = time.perf_counter()
        innovation = residual.latest_ref() if not freeze_gate else np.zeros(5)
        obs, errors_before, velocity = _make_observation(history, before,
            last_issued, innovation, theta, radius, alpha, last_qp_ok, last_qp_ms)
        feature_ms = 1000*(time.perf_counter()-feature_started)
        policy_ms = 0.
        if k == 0 or (k-gate_phase_steps) % 5 == 0:
            target = np.zeros(4); raw_z = np.full(4, np.nan); log_prob = np.nan; value = np.nan
            if k == 0 and gate_phase_steps > 0:
                target = alpha.copy()
            elif freeze_gate:
                target = alpha.copy()
            elif method.startswith("fixed:"):
                target = np.fromstring(method.split(":", 1)[1], sep=",")
            elif method.startswith("rule"):
                threshold = .05 if method == "rule5" else .1
                target = _rule_target(shadow_history, source, alpha, threshold)
            elif method == "random":
                target = rng.uniform(0., 1., 4)
            elif method == "ppo":
                if policy is None: raise ValueError("ppo method requires policy")
                policy_started = time.perf_counter()
                with torch.no_grad():
                    result = policy.act(torch.tensor(obs[None]), deterministic=False)
                policy_ms = 1000*(time.perf_counter()-policy_started)
                target = result["target"][0].numpy()
                if policy.action_dim == 2:
                    target = np.array([target[0], target[1], target[1], target[1]])
                raw_z = result["raw_z"][0].numpy()
                log_prob = float(result["log_prob"][0]); value = float(result["value"][0])
            elif method == "ppo_mean":
                if policy is None: raise ValueError("ppo_mean requires policy")
                policy_started = time.perf_counter()
                with torch.no_grad():
                    result = policy.act(torch.tensor(obs[None]), deterministic=True)
                policy_ms = 1000*(time.perf_counter()-policy_started)
                target = result["target"][0].numpy()
                if policy.action_dim == 2:
                    target = np.array([target[0], target[1], target[1], target[1]])
                raw_z = result["raw_z"][0].numpy()
                log_prob = float(result["log_prob"][0]); value = float(result["value"][0])
            elif method != "hybrid_zero": raise ValueError(method)
            if ablation == "freeze4" and before["t"] >= 4.-1e-9:
                target = alpha.copy()
            previous_alpha = alpha.copy()
            alpha, override = bounded_gate(alpha, target)
            if ablation == "no_theta":
                override = override or alpha[0] != 0.
                alpha[0] = 0.
            decisions.append({"t": before["t"], "observation": obs, "raw_z": raw_z,
                              "log_prob": log_prob, "value": value, "target": target,
                              "alpha": alpha.copy(), "previous_alpha": previous_alpha,
                              "override": override, "raw_valid": bool(np.isfinite(log_prob)),
                              "policy_version": policy_hash or "deterministic",
                              "reward": 0., "reward_tracking": 0.,
                              "reward_alpha": 0., "reward_failure": 0.,
                              "fast_steps": 0})
            decisions[-1]["reward_alpha"] = -.01*float(np.sum((alpha-previous_alpha)**2))
            decisions[-1]["reward"] += decisions[-1]["reward_alpha"]
        model_started = time.perf_counter()
        theta_used = THETA_SAFE+alpha[0]*(theta-THETA_SAFE)
        d_ref, d_raw, d_clipped, was_clipped = residual.estimate(theta_used)
        theta_used, d_used = mix_gate(alpha, theta, d_clipped)
        if freeze_gate:
            if locked_theta is None or locked_d is None:
                raise ValueError("freeze_gate requires a locked snapshot")
            theta_used, d_used = locked_theta, locked_d
        z = np.r_[before["state"], before["actual"]/CONTROL_SCALE,
                  thrust_observer.rate, 0., 0.,
                  before["queue"][:3].reshape(-1)/np.tile(CONTROL_SCALE, 3)]
        reference = plant.reference.horizon(before["t"])
        model_ms = 1000*(time.perf_counter()-model_started)
        solve_started = time.perf_counter()
        command, qp_ok = mpc.solve(model.tensor(z[None]), model.tensor(reference[None]),
                                   alpha=[1., 1.], theta_hat=theta_used,
                                   d_hat=np.tile(d_used, (35, 1)))
        qp_ms = 1000*(time.perf_counter()-solve_started)
        issued = command[0].numpy().copy()
        plan = mpc.plan[0].numpy()*CONTROL_SCALE
        def stop_on_failure(sample):
            nonlocal first_failure, failure_reason
            state, ref = sample["state"], sample["reference"]
            velocity = (rotation_ned(state[3], state[4])@state[5:8]
                        if np.isfinite(state).all() else np.full(3, np.nan))
            tangent = ref[5:8]/max(np.linalg.norm(ref[5:8]), 1e-12)
            tangent_speed_error = velocity@tangent-np.linalg.norm(ref[5:8])
            lateral_error = (-np.sin(ref[4])*(state[0]-ref[0])+
                             np.cos(ref[4])*(state[1]-ref[1]))
            flags = {"lateral": abs(lateral_error) > 8.,
                     "depth": abs(state[2]-ref[2]) > 5.,
                     "tangent_speed": abs(tangent_speed_error) > 2.,
                     "pitch": abs(state[3]) > np.deg2rad(40),
                     "nonfinite": not np.isfinite(state).all()}
            if any(flags.values()):
                first_failure = sample["t"]
                failure_reason = [name for name, hit in flags.items() if hit]
                return True
            return False
        plant_started = time.perf_counter()
        rows = plant.step(issued, stop_when=stop_on_failure)
        plant_ms = 1000*(time.perf_counter()-plant_started)
        after = plant.measurement()
        if len(rows) == 5:
            thrust_observer.update(after["actual"][0]/CONTROL_SCALE[0])
        if len(rows) == 5 and not freeze_gate:
            residual.append(z, issued/CONTROL_SCALE, after["state"][5:10])
            # Prequential shadow scores: predictions were fixed before this
            # transition, so the new sample never fits its own predictor.
            shadow_history.append(_shadow_errors(mpc.dynamics, z, issued/CONTROL_SCALE,
                                  after["state"][5:10], theta, theta_used, d_clipped))
        errors_after, _ = _error(after)
        completed_row = (_estimator_row(before, after, issued, plan, errors_after)
                         if not freeze_gate else None)
        saturation = float(np.any(np.isclose(issued, [0., -CONTROL_SCALE[1], -CONTROL_SCALE[2]], atol=1e-6) |
                                      np.isclose(issued, CONTROL_SCALE, atol=1e-6)))
        if np.isfinite(errors_after).all() and np.isfinite(after["state"][3]):
            reward, stage_cost = fast_reward(errors_after, after["state"][3],
                (issued-last_issued)/CONTROL_SCALE, saturation, dt=len(rows)*.02)
        else:
            reward, stage_cost = -50.*len(rows)*.02, 50.
        decisions[-1]["reward"] += reward
        decisions[-1]["reward_tracking"] += reward
        decisions[-1]["fast_steps"] += 1
        physical_flags = []
        for sample in rows:
            state, ref = sample["state"], sample["reference"]
            velocity = (rotation_ned(state[3], state[4])@state[5:8]
                        if np.isfinite(state).all() else np.full(3, np.nan))
            tangent = ref[5:8]/max(np.linalg.norm(ref[5:8]), 1e-12)
            tangent_speed_error = velocity@tangent-np.linalg.norm(ref[5:8])
            lateral_error = (-np.sin(ref[4])*(state[0]-ref[0])+
                             np.cos(ref[4])*(state[1]-ref[1]))
            flags = {"lateral": abs(lateral_error) > 8.,
                     "depth": abs(state[2]-ref[2]) > 5.,
                     "tangent_speed": abs(tangent_speed_error) > 2.,
                     "pitch": abs(state[3]) > np.deg2rad(40),
                     "nonfinite": not np.isfinite(state).all()}
            if save_substeps:
                substeps.append(np.concatenate(([sample["t"]], state, ref,
                                       sample["actual"], sample["issued"],
                                       np.asarray(list(map(float, flags.values()))))))
        control.append({"t": before["t"], "state": before["state"],
                        "model_state": z,
                        "actual": before["actual"], "queue": before["queue"],
                        "issued": issued, "mpc_command": plan[0],
                        "reference": before["reference"],
                        "preview_0p5": before["preview_0p5"],
                        "preview_1p0": before["preview_1p0"],
                        "errors": errors_after, "theta_hat": theta, "theta_used": theta_used,
                        "theta_radius": radius, "theta_source": source,
                        "estimator_fallback": fallback_reason,
                        "innovation": innovation,
                        "d_ref": d_ref, "d_raw": d_raw, "d_used": d_used,
                        "alpha": alpha.copy(), "qp_ok": bool(qp_ok[0]),
                        "qp_status": mpc.last_status[0],
                        "qp_primal_residual": float(mpc.last_primal_residual[0]),
                        "qp_dual_residual": float(mpc.last_dual_residual[0]),
                        "solve_ms": qp_ms, "model_ms": model_ms,
                        "estimator_ms": estimator_ms, "feature_ms": feature_ms,
                        "policy_ms": policy_ms, "plant_ms": plant_ms,
                        "clip": was_clipped, "saturation_fraction": saturation,
                        "reward": reward,
                        "stage_cost": stage_cost, "actual_dt": len(rows)*.02,
                        "residual_valid": len(rows) == 5,
                        "total_ms": 1000*(time.perf_counter()-period_started),
                        "deadline_miss": (time.perf_counter()-period_started) > .1})
        last_issued, last_qp_ok, last_qp_ms = issued, bool(qp_ok[0]), qp_ms
        if first_failure is not None:
            decisions[-1]["reward_failure"] += failure_cost(first_failure-start_t, planned)
            decisions[-1]["reward"] += decisions[-1]["reward_failure"]
            break
    truncated = first_failure is None and executed_limit < planned
    bootstrap_observation = np.zeros(200, dtype=np.float32)
    bootstrap_value = 0.
    if truncated:
        if completed_row is not None and not freeze_gate:
            estimate = estimator.update(completed_row)
            theta = np.array([estimate.fin_gain, estimate.fin_tau_scale])
            radius = np.array([estimate.gain_radius, estimate.tau_radius])
        innovation = residual.latest_ref() if not freeze_gate else np.zeros(5)
        bootstrap_observation, _, _ = _make_observation(history, plant.measurement(),
            last_issued, innovation, theta, radius, alpha, last_qp_ok, last_qp_ms)
        if policy is not None:
            with torch.no_grad():
                bootstrap_value = float(policy.value(torch.tensor(bootstrap_observation[None]))[0])
    for j, decision in enumerate(decisions):
        decision["terminal"] = j == len(decisions)-1 and not truncated
        decision["truncated"] = j == len(decisions)-1 and truncated
        decision["next_observation"] = (decisions[j+1]["observation"] if j+1 < len(decisions)
                                        else bootstrap_observation)
        decision["next_value"] = (decisions[j+1]["value"] if j+1 < len(decisions) else bootstrap_value)
    write_npz(out/"control.npz", **{name: np.asarray([row[name] for row in control])
                                    for name in control[0]})
    write_npz(out/"decisions.npz", **{name: np.asarray([row[name] for row in decisions])
                                      for name in decisions[0]})
    if save_substeps:
        write_npz(out/"substeps.npz", values=np.asarray(substeps))
        immutable_json(out/"substep_schema.json",
            {"columns": ["t_s"]+[f"state_{i}" for i in range(10)]+
             [f"reference_{i}" for i in range(8)]+
             [f"actual_{i}" for i in range(3)]+
             [f"issued_{i}" for i in range(3)]+
             ["lateral_failure", "depth_failure", "tangent_speed_failure",
              "pitch_failure", "nonfinite_failure"]})
    finite_east = [abs(row["errors"][0]) for row in control
                   if np.isfinite(row["errors"][0])]
    finite_depth = [abs(row["errors"][1]) for row in control
                    if np.isfinite(row["errors"][1])]
    summary = {"case_id": case["case_id"], "split": case["split"], "method": method,
               "signature": signature, "policy_hash": policy_hash,
               "policy_variant": policy.variant if policy is not None else None,
               "duration_expected_s": planned, "duration_actual_s": plant.t-start_t,
               "complete": first_failure is None and not truncated and len(control) == planned_steps,
               "truncated": truncated,
               "first_failure_s": (first_failure-start_t if first_failure is not None else None),
               "failure_reason": failure_reason,
               "control_steps": len(control), "substeps": len(substeps),
               "decisions": len(decisions), "qp_failures": sum(not row["qp_ok"] for row in control),
               "deadline_misses": sum(row["deadline_miss"] for row in control),
               "saturation_fraction": float(np.mean([row["saturation_fraction"] for row in control])),
               "tracking_cost_integral": float(sum(row["actual_dt"]*row["stage_cost"] for row in control)),
               "peak_east_m": float(max(finite_east)) if finite_east else None,
               "peak_depth_m": float(max(finite_depth)) if finite_depth else None,
               "wall_s": time.perf_counter()-started,
               "snapshot_source_hash": snapshot_hash, "freeze_gate": freeze_gate,
               "ablation": ablation, "gate_phase_steps": gate_phase_steps,
               "files": {name: file_hash(out/name) for name in
                         ("case.json", "control.npz", "decisions.npz") +
                         (("substeps.npz", "substep_schema.json") if save_substeps else ())}}
    if snapshot_out is not None:
        if Path(snapshot_out).exists(): raise FileExistsError(snapshot_out)
        snapshot = {"policy_hash": policy_hash,
                    "plant_state": plant.state.copy(), "plant_actual": plant.actual.copy(),
                    "plant_rate": plant.rate.copy(), "plant_queue": plant.queue.copy(),
                    "plant_t": plant.t, "mpc_plan": mpc.plan.numpy().copy(),
                    "mpc_previous": mpc.previous.numpy().copy(),
                    "observer_previous": thrust_observer.previous,
                    "observer_rate": thrust_observer.rate,
                    "residual_history": list(residual.history),
                    "observation_history": list(history.frames),
                    "estimator_history": list(estimator.history),
                    "estimator_last": estimator.last, "alpha": alpha.copy(),
                    "theta": theta.copy(), "radius": radius.copy(),
                    "last_issued": last_issued.copy(), "last_qp_ok": last_qp_ok,
                    "last_qp_ms": last_qp_ms,
                    "theta_used": control[-1]["theta_used"].copy(),
                    "d_used": control[-1]["d_used"].copy()}
        Path(snapshot_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, snapshot_out)
        summary["files"]["snapshot.pt"] = file_hash(snapshot_out)
    immutable_json(out/"episode.json", summary)
    return summary
