"""Single CAT-MPC episode with causal adapter/predictor/gate call order."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from aav_mpc.controller import EndpointExtendedCondensedEnv
from aav_mpc.h15_target_domain import install_target_domain
from experiments.h15_cat_mpc.adapter import FrozenResponseAdapter, THETA0
from experiments.h15_cat_mpc.continuous_gate import ActorCritic
from experiments.h15_cat_mpc.residual_predictor import ResidualAR1
from simulator.batch_mpc import CONTROL_SCALE
from simulator.batch_path_env import PathBank
from simulator.model import MISMATCHED_PLANT
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GateEnv(EndpointExtendedCondensedEnv):
    def __init__(self, *args, gate_callback=None, **kwargs):
        self.gate_callback = gate_callback
        self.completed_rows = []
        self.residual_history = []
        self.alpha = np.zeros(2, dtype=float)
        self.theta_hat = np.asarray(THETA0, dtype=float)
        self.theta_radius = np.array([.25, 2.5])
        self.d_forecast = np.zeros((35, 5))
        self.gate_log = []
        self.pending = None
        self.period_index = 0
        self.last_period_ms = 0.
        self.physical_intervention = None
        self.pending_alpha_penalty = 0.
        super().__init__(*args, **kwargs)

    def _start_pending(self):
        if self.pending is None and not bool(self.finished[0]):
            self.pending = {
                "state_before": self.state[0].detach().cpu().numpy().copy(),
                "actuator_before": self.actuator[0].detach().cpu().numpy().copy(),
                "queue_before": self.command_history[0].detach().cpu().numpy().copy(),
                "command": self.base_command[0].detach().cpu().numpy().copy(),
                "plan": (self.mpc.plan[0].detach().cpu().numpy().copy() *
                         np.asarray(CONTROL_SCALE)),
                "theta_used": self.theta_hat.copy(),
                "d_forecast": self.d_forecast.copy(),
            }

    def _complete_pending(self):
        if self.pending is None:
            return
        p = self.pending
        state_after = self.state[0].detach().cpu().numpy().copy()
        actual_after = self.actuator[0].detach().cpu().numpy().copy()
        errors = [float(value[0]) for value in self._errors()[1:6]]
        lateral, depth, yaw, pitch, speed_error = errors
        row = {
            "time_s": self.period_index*.1,
            "state_before_json": json.dumps(p["state_before"].tolist()),
            "state_after_json": json.dumps(state_after.tolist()),
            "actuator_before_json": json.dumps(p["actuator_before"].tolist()),
            "actuator_after_json": json.dumps(actual_after.tolist()),
            "queue_before_json": json.dumps(p["queue_before"].tolist()),
            "command_json": json.dumps(p["command"].tolist()),
            "mpc_base_command_json": json.dumps(p["plan"][0].tolist()),
            "mpc_plan_commands_json": json.dumps(p["plan"].tolist()),
            "thrust_actual_n": float(actual_after[0]),
            "elevator_actual_deg": float(np.rad2deg(actual_after[1])),
            "rudder_actual_deg": float(np.rad2deg(actual_after[2])),
            "lateral_error_m": lateral, "depth_error_m": depth,
            "yaw_error_rad": yaw, "pitch_deg": float(np.rad2deg(state_after[3])),
            "speed_error_mps": speed_error,
            "progress_m": float(self.progress[0]),
            "residual_valid": False,
            "d_forecast": p["d_forecast"].tolist(),
        }
        # Fixed, observable theta0 point dynamics; the hidden sampled plant
        # parameters and second-order actuator rate are never read here.
        model_parameters = self.model.tensor([
            MISMATCHED_PLANT.mass_scale, MISMATCHED_PLANT.damping_scale,
            MISMATCHED_PLANT.thrust_gain,
            MISMATCHED_PLANT.elevator_gain*p["theta_used"][0],
            MISMATCHED_PLANT.rudder_gain*p["theta_used"][0]])[None]
        actual_control = self.model.tensor(p["actuator_before"])[None]
        before = self.model.tensor(p["state_before"])[None]
        predicted = self.model.rk4(before, actual_control,
                                   parameters=model_parameters, dt=.1)
        residual = ((state_after[5:10]-predicted[0, 5:10].detach().cpu().numpy())/.1)
        if np.isfinite(residual).all():
            row["residual_valid"] = True
            self.residual_history.append(residual.copy())
        row["residual_body_acc"] = residual.tolist() if row["residual_valid"] else [0.]*5
        self.completed_rows.append(row)
        self.pending = None
        self.period_index += 1

    def _solve_mpc(self):
        self._complete_pending()
        violations = []
        for sample in getattr(self, "strict_substeps", []):
            lateral = abs(float(sample["lateral_error_m"][0]))
            depth = abs(float(sample["depth_error_m"][0]))
            speed = abs(float(sample["speed_error_mps"][0]))
            pitch = abs(float(np.rad2deg(sample["pitch_rad"][0])))
            violations.extend(kind for kind, flag in (
                ("lateral", lateral > 8.), ("depth", depth > 5.),
                ("speed", speed > 2.), ("pitch", pitch > 40.)) if flag)
        if violations:
            self.physical_intervention = sorted(set(violations))
            return self.base_command.clone(), self.last_mpc_ok.clone()
        estimate = (self.gate_callback(self) if self.gate_callback is not None
                    else {"alpha": self.alpha, "theta_hat": self.theta_hat,
                          "d_hat": self.d_forecast})
        self.alpha = np.asarray(estimate["alpha"], dtype=float)
        self.theta_hat = np.asarray(estimate["theta_hat"], dtype=float)
        self.d_forecast = np.asarray(estimate["d_hat"], dtype=float)
        self.mpc.set_continuous_model(self.alpha, self.theta_hat,
                                      self.d_forecast[:self.mpc.horizon])
        started = time.perf_counter()
        command, ok = self.mpc.solve(self.observed_augmented_state(),
                                     self._reference())
        self.last_period_ms = (time.perf_counter()-started)*1000.
        self.gate_log.append({"period": self.period_index,
                              "alpha": self.alpha.tolist(),
                              "theta_hat": self.theta_hat.tolist(),
                              "theta_source": estimate.get("theta_source", "fixed"),
                              "fallback_reason": estimate.get("fallback_reason"),
                              "raw_z": estimate.get("raw_z"),
                              "raw_log_prob": estimate.get("raw_log_prob"),
                              "value": estimate.get("value"),
                              "observation": estimate.get("observation"),
                              "alpha_target": estimate.get("alpha_target"),
                              "slew_limited": estimate.get("slew_limited"),
                              "qp_status": self.mpc.last_status[0],
                              "qp_ms": self.last_period_ms,
                              **estimate.get("timing", {})})
        return command, ok

    def step(self, action, **kwargs):
        self.physical_intervention = None
        alpha_penalty = self.pending_alpha_penalty
        self.pending_alpha_penalty = 0.
        self._start_pending()
        result = super().step(action, **kwargs)
        observation, reward, done, info = result
        if self.physical_intervention is None:
            violations = []
            for sample in getattr(self, "strict_substeps", []):
                lateral = abs(float(sample["lateral_error_m"][0]))
                depth = abs(float(sample["depth_error_m"][0]))
                speed = abs(float(sample["speed_error_mps"][0]))
                pitch = abs(float(np.rad2deg(sample["pitch_rad"][0])))
                violations.extend(kind for kind, flag in (
                    ("lateral", lateral > 8.), ("depth", depth > 5.),
                    ("speed", speed > 2.), ("pitch", pitch > 40.)) if flag)
            self.physical_intervention = sorted(set(violations)) or None
        saturation = bool(torch.any(torch.abs(info["command"][0]/self.model.control_scale-
            torch.tensor([.5, 0., 0.], device=self.device, dtype=self.dtype)) >
            torch.tensor([.5, .999, .999], device=self.device, dtype=self.dtype)))
        shaped_reward = reward.clone()-float(alpha_penalty)
        if saturation:
            shaped_reward[0] -= .02
        if not bool(info["mpc_ok"][0]):
            shaped_reward[0] -= 10.
        if self.physical_intervention:
            done = done.clone(); done[0] = True
            self.finished[0] = True
            failed = info["failed"].clone(); failed[0] = True
            success = info["success"].clone(); success[0] = False
            info["failed"], info["success"] = failed, success
            info["physical_intervention"] = self.physical_intervention
            shaped_reward[0] = -100.
        if bool(result[2][0]) or self.physical_intervention:
            self._complete_pending()
        return observation, shaped_reward, done, info


def observation37(env, theta_radius, alpha_previous, forecast):
    _, lateral, depth, yaw, pitch, speed_error, _ = env._errors()
    state = env.state[0].detach().cpu().numpy()
    geometry = np.array([float(lateral[0]), float(depth[0]), float(yaw[0]),
                         float(pitch[0]), float(speed_error[0]), state[3],
                         state[4], float(env.progress[0]/env.paths.lengths[env.path_ids[0]])])
    errors = np.asarray(env.residual_history[-10:], dtype=float).reshape(-1, 5)
    summary = np.concatenate((forecast.mean(0), forecast[0],
                              forecast[:min(10, len(forecast))].mean(0)))
    margins = np.array([8.-abs(float(lateral[0])), 5.-abs(float(depth[0])),
                        40.-abs(float(np.rad2deg(state[3]))),
                        2.-abs(float(speed_error[0])),
                        float(env.mpc.last_primal_residual[0]),
                        float(env.mpc.last_dual_residual[0])])
    status = float(env.mpc.last_qp_valid[0])
    values = np.concatenate((np.asarray(env.theta_hat), np.asarray(theta_radius),
                             summary, margins, [status, env.last_period_ms/100.],
                             np.asarray(alpha_previous), geometry))
    scales = np.asarray([1., 2., .25, 2.5] +
        [1., 1., 1., .1, .1]*3 + [8., 5., 40., 2., 1., 1.,
        1., 1., 1., 1., 8., 5., 1., 1., 2., 1., np.pi, 1.])
    if values.shape != (37,) or scales.shape != (37,) or not np.isfinite(values).all():
        raise ValueError(f"CAT-MPC observation must be finite [37], got {values.shape}")
    return np.clip(values/scales, -10., 10.).astype(np.float32)


def run_episode(path, seed, out, method="A0", target_domain=True,
                adapter_artifacts=None, predictor_checkpoint=None,
                policy_checkpoint=None, max_steps=None):
    path = Path(path)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    bank = PathBank([path], "cpu")
    bank.speed[:] = 15.; bank.first_state[:, 5] = 15.
    bank.durations[:] = bank.lengths/15.
    env = GateEnv(bank, batch_size=1, device="cpu", horizon=35,
        backend="rti_cpu", seed=seed, randomize_plant=target_domain,
        current_max_mps=1. if target_domain else 0.,
        current_profile="harmonic" if target_domain else "steady",
        admm_iterations=400, admm_rho=10., attitude_pitch_weight=100.,
        attitude_yaw_weight=30., fsp_enabled=False,
        preview_mode="multiscale", yaw_reward="course")
    env.mpc.dynamics = ActuatorAwareCpuDynamics(.1, .85, 2.)
    env.profile_stages = True
    if not target_domain:
        env.disturbance_std.zero_()
    install_target_domain(env, fin_gain_scale=.9, second_order=True,
                          integrator=True) if target_domain else None
    adapter = FrozenResponseAdapter(**(adapter_artifacts or {}))
    predictor = ResidualAR1.load(predictor_checkpoint) if predictor_checkpoint else ResidualAR1()
    model, policy_previous, alpha = None, None, np.zeros(2)
    if policy_checkpoint:
        model = ActorCritic(37, 64)
        checkpoint = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["policy"])
        model.eval()
    fixed = {"A0": (0., 0.), "A1": (1., 0.), "A2": (0., 1.),
             "A3": (1., 1.), "A4": (.5, .5)}
    if method in fixed:
        alpha = np.asarray(fixed[method])

    def choose(current):
        nonlocal alpha, policy_previous
        alpha_before_decision = np.asarray(alpha).copy()
        adapter_started = time.perf_counter()
        estimate = adapter.update(current.completed_rows[-1]) if current.completed_rows else {
            "theta_hat": THETA0, "source": "E0", "fallback_reason": "short_history",
            "radius": (.25, 2.5), "forward_ms": 0.}
        adapter_wall_ms = (time.perf_counter()-adapter_started)*1000.
        theta = np.asarray(estimate["theta_hat"], dtype=float)
        radius = np.asarray(estimate["radius"], dtype=float)
        residuals = np.asarray(current.residual_history[-10:], dtype=float).reshape(-1, 5)
        predictor_started = time.perf_counter()
        forecast = predictor.predict(residuals, 35)
        predictor_wall_ms = (time.perf_counter()-predictor_started)*1000.
        current.theta_hat = theta
        current.theta_radius = radius
        current.d_forecast = forecast
        decision = current.period_index % 5 == 0
        raw_z, raw_log_prob = None, None
        value, observation = None, None
        alpha_target, slew_limited = None, False
        policy_wall_ms = 0.
        if model is not None and decision:
            obs = torch.as_tensor(observation37(current, radius, alpha, forecast))[None]
            policy_started = time.perf_counter()
            with torch.no_grad():
                action = model.act(obs, torch.as_tensor(alpha, dtype=torch.float32)[None])
            policy_wall_ms = (time.perf_counter()-policy_started)*1000.
            raw_z = action["raw_z"][0].tolist()
            raw_log_prob = float(action["log_prob"][0])
            value = float(action["value"][0])
            observation = obs[0].tolist()
            alpha_target = action["alpha_target"][0].tolist()
            proposed = np.asarray(alpha_target)
            alpha = action["alpha"][0].numpy()
            slew_limited = bool(np.any(np.abs(proposed-alpha) > 1e-8))
        alpha_delta = np.asarray(alpha)-alpha_before_decision
        current.pending_alpha_penalty = .01*float(alpha_delta@alpha_delta)
        return {"alpha": alpha.copy(), "theta_hat": theta,
                "d_hat": forecast, "theta_source": estimate["source"],
                "fallback_reason": estimate["fallback_reason"],
                "raw_z": raw_z, "raw_log_prob": raw_log_prob,
                "value": value, "observation": observation,
                "alpha_target": alpha_target, "slew_limited": slew_limited,
                "timing": {"adapter_ms": adapter_wall_ms,
                           "predictor_ms": predictor_wall_ms,
                           "policy_ms": policy_wall_ms}}

    env.gate_callback = choose
    env.reset(torch.zeros(1, dtype=torch.long))
    env.capture_strict_substeps = True
    trace, substeps, rewards, period_ms = [], [], [], []
    started = time.perf_counter()
    for _ in range(int(max_steps or env.max_steps[0])):
        tick = time.perf_counter()
        obs, reward, done, info = env.step(torch.zeros(1, 3))
        wall_ms = (time.perf_counter()-tick)*1000.
        period_ms.append(wall_ms); rewards.append(float(reward[0]))
        for sample in env.strict_substeps:
            lateral = abs(float(sample["lateral_error_m"][0]))
            depth = abs(float(sample["depth_error_m"][0]))
            speed = abs(float(sample["speed_error_mps"][0]))
            pitch = abs(float(np.rad2deg(sample["pitch_rad"][0])))
            substeps.append({"period": int(env.steps[0])-1,
                "substep": sample["substep"], "time_s": (int(env.steps[0])-1)*.1+
                sample["substep"]*.02, "state": sample["state"][0],
                "actuator": sample["actuator"][0], "queue": sample["queue"][0],
                "lateral_error_m": sample["lateral_error_m"][0],
                "depth_error_m": sample["depth_error_m"][0],
                "speed_error_mps": sample["speed_error_mps"][0],
                "pitch_rad": sample["pitch_rad"][0],
                "physical_flags": {"lateral": lateral > 8., "depth": depth > 5.,
                    "speed": speed > 2., "pitch": pitch > 40.}})
        _, lat, dep, _, pit, speed_err, _ = env._errors()
        hist_flags = {"lateral": abs(float(lat[0])) > 8.,
                      "depth": abs(float(dep[0])) > 5.,
                      "speed": abs(float(speed_err[0])) > 2.,
                      "pitch": abs(float(np.rad2deg(pit[0]))) > 40.}
        trace.append({"period": len(trace), "command": info["command"][0].tolist(),
                      "state": env.state[0].detach().cpu().tolist(),
                      "alpha": env.alpha.tolist(), "qp_ok": bool(info["mpc_ok"][0]),
                      "physical_flags": hist_flags,
                      "physical_intervention": info.get("physical_intervention"),
                      "deadline_miss": wall_ms > 100., "reward": rewards[-1],
                      "timing": info.get("timing", {})})
        if bool(done[0]):
            break
    any_physical = any(any(sample["physical_flags"].values()) for sample in substeps)
    hist_physical = any(any(row["physical_flags"].values()) for row in trace)
    full_path = bool(info["success"][0])
    forecast_metrics = {}
    valid_rows = [i for i, row in enumerate(env.completed_rows)
                  if row["residual_valid"]]
    for horizon in (1, 5, 25):
        ar_errors, persistence_errors = [], []
        for i in valid_rows:
            target_i = i+horizon-1
            if target_i >= len(env.completed_rows) or not env.completed_rows[target_i]["residual_valid"]:
                continue
            target = np.asarray(env.completed_rows[target_i]["residual_body_acc"])
            forecast = np.asarray(env.completed_rows[i]["d_forecast"])[horizon-1]
            ar_errors.append(np.abs(target-forecast))
            prior = (np.asarray(env.completed_rows[i-1]["residual_body_acc"])
                     if i > 0 and env.completed_rows[i-1]["residual_valid"]
                     else np.zeros(5))
            persistence_errors.append(np.abs(target-prior))
        forecast_metrics[str(horizon)] = {
            "count": len(ar_errors),
            "ar1_mae": np.mean(ar_errors, axis=0).tolist() if ar_errors else None,
            "persistence_mae": (np.mean(persistence_errors, axis=0).tolist()
                                if persistence_errors else None)}
    result = {"schema": 1, "method": method, "seed": int(seed),
        "path": str(path), "periods": len(trace), "wall_s": time.perf_counter()-started,
        "success": bool(info["success"][0]), "failed": bool(info["failed"][0]),
        "timeout": bool(info["timeout"][0]), "progress_fraction": float(info["progress_fraction"][0]),
        "fallback_periods": sum(not item["qp_ok"] for item in trace),
        "physical_intervention": getattr(env, "physical_intervention", None),
        "backup_periods": 0, "period_ms": period_ms,
        "deadline_miss": [value > 100. for value in period_ms],
        "strict_hist": full_path and not hist_physical,
        "physical_strict": full_path and not any_physical,
        "strict_mainline": (full_path and not any_physical and
                            not any(not item["qp_ok"] for item in trace) and
                            not any(value > 100. for value in period_ms)),
        "strict_substeps": substeps, "trace": trace,
        "alpha_log": env.gate_log, "completed_rows": env.completed_rows,
        "residual_forecast_errors": forecast_metrics,
        "residual_forecast_version": predictor.version,
        "reward_sum": sum(rewards)}
    (out/"episode.json").write_text(json.dumps(result, indent=2)+"\n")
    source_files = [ROOT/"experiments/h15_cat_mpc/run_episode.py",
                    ROOT/"experiments/h15_cat_mpc/adapter.py",
                    ROOT/"experiments/h15_cat_mpc/residual_predictor.py",
                    ROOT/"experiments/h15_cat_mpc/continuous_gate.py",
                    ROOT/"aav_mpc/condensed_mpc.py",
                    ROOT/"simulator/rti_symbolic.py"]
    record = {"schema": 1, "method": method, "seed": int(seed),
              "path": str(path), "path_sha256": sha256(path),
              "config_sha256": sha256(ROOT/"experiments/h15_cat_mpc/config.json"),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                                for p in source_files},
              "episode_sha256": sha256(out/"episode.json"),
              "python_executable": "/bin/python3", "target_domain": target_domain,
              "policy_checkpoint": str(policy_checkpoint) if policy_checkpoint else None,
              "adapter_artifacts": ({k: str(v) for k, v in adapter_artifacts.items()}
                                     if adapter_artifacts else None),
              "predictor_checkpoint": str(predictor_checkpoint) if predictor_checkpoint else None}
    (out/"run.json").write_text(json.dumps(record, indent=2)+"\n")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=("A0", "A1", "A2", "A3", "A4", "PPO"), default="A0")
    parser.add_argument("--adapter-normalization", type=Path)
    parser.add_argument("--adapter-calibration", type=Path)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--target-domain", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_episode(args.path, args.seed, args.out, args.method,
        args.target_domain,
        {"normalization_path": args.adapter_normalization,
         "calibration_path": args.adapter_calibration}
        if args.adapter_normalization else None,
        args.predictor, args.policy_checkpoint, args.max_steps)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("trace", "strict_substeps", "completed_rows")}, indent=2))


if __name__ == "__main__":
    main()
