"""Closed-loop H15 B0/B2/B3 mechanism run using the full offline screen.

The screen ignores only the 5 ms deployment budget. It still solves each
candidate's own N35 QP and replays its command plan against frozen limits.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import time

import numpy as np
import torch

from aav_mpc.h15_target_domain import install_target_domain
from experiments.h15_mainline.candidate_filter import (SCALE, screen_candidate,
    screen_one_step_backup, select_b2)
from experiments.h15_mainline.cycle_log import FIELDS, TIMING_FIELDS
from experiments.h15_mainline.cycle_writer import timing_summary, write_once as write_cycles
from experiments.h15_mainline.estimator_io import load_frozen_estimator
from experiments.h15_mainline.masked_speed_ppo import (GRID, MaskedSpeedActorCritic,
                                                        decision_observation)
from experiments.h15_mainline.progress_reference import ProgressReferenceEnv
from experiments.h15_mainline.run_record import RESULT_FIELDS, TIMING_STAGES, TOP_LEVEL
from experiments.h15_mainline.run_record import write_once as write_run
from experiments.h15_mainline.run_smoke import CASES
from experiments.h15_mainline.collect_train_observations import MODES as TRAIN_MODES
from experiments.h15_mainline.collect_dev_calibration import MODES as DEV_MODES
from experiments.h15_mainline.strict_contract import evaluate, first_physical_violation
from simulator.batch_path_env import PathBank
from simulator.batch_path_env import BatchedPathResidualEnv
from simulator.model import LoopPath, rotation_ned
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeferredSolveEnv(ProgressReferenceEnv):
    """Avoid an unused MPC solve at the end of each plant step."""

    def __init__(self, *args, **kwargs):
        self.defer_mpc_solve = False
        super().__init__(*args, **kwargs)

    def _solve_mpc(self):
        if self.defer_mpc_solve:
            return self.base_command.clone(), self.last_mpc_ok.clone()
        return super()._solve_mpc()

    @torch.no_grad()
    def step_verified_command(self):
        """Apply the selected MPC command directly, then update observed current."""
        old_state = self.state.clone()
        issued = self.base_command.clone()
        active = ~self.finished
        _, reward, done, info = BatchedPathResidualEnv.step(
            self, torch.zeros(self.batch_size, 3), return_observation=False)
        if not torch.allclose(info["command"], issued, rtol=0., atol=1e-8):
            raise RuntimeError("plant executed a command different from verified MPC")
        current_innovation = ((self.state[:, :3]-old_state[:, :3]-
                               info["water_displacement_ned"])/self.mpc.dt)
        current_innovation = current_innovation.clamp(-2*self.current_scale,
                                                       2*self.current_scale)
        valid = torch.isfinite(current_innovation).all(-1)
        updated = ((1-self.current_gain)*self.current_estimate +
                   self.current_gain*current_innovation)
        self.current_estimate = torch.where(
            (active & valid & ~info["failed"])[:, None], updated,
            self.current_estimate)
        return reward, done, info


def create_env(path_file, seed, mismatch, current_max=1., target_fin_gain=.9):
    bank = PathBank([path_file], "cpu")
    bank.speed[:] = 15.
    bank.first_state[:, 5] = 15.
    bank.durations[:] = bank.lengths / 15.
    env = DeferredSolveEnv(
        bank, batch_size=1, device="cpu", horizon=35,
        backend="rti_cpu", seed=seed, randomize_plant=mismatch,
        current_max_mps=current_max if mismatch else 0.,
        current_profile="harmonic" if mismatch else "steady",
        admm_iterations=400, admm_rho=10.,
        attitude_pitch_weight=100., attitude_yaw_weight=30.,
        fsp_enabled=False, preview_mode="multiscale", yaw_reward="course")
    env.mpc.dynamics = ActuatorAwareCpuDynamics(.1, fin_gain_scale=.85,
                                                fin_tau_scale=2.)
    env.reset(torch.zeros(1, dtype=torch.long))
    if not mismatch:
        env.disturbance_std.zero_()
    else:
        install_target_domain(env, fin_gain_scale=target_fin_gain, integrator=True,
                              second_order=True, fin_tau_scale=1.,
                              fin_delay_scale=1.)
    env.capture_strict_substeps = True
    env.profile_stages = True
    env.defer_mpc_solve = True
    return env


def physical_sample(env, path, time_s):
    state = env.state[0].numpy()
    progress = float(env.progress[0])
    tangent = path.tangent_at(progress)
    position = path.position_at(progress)
    normal = np.array([-tangent[1], tangent[0], 0.])
    normal /= max(np.linalg.norm(normal), 1e-12)
    water_speed = float(tangent @ (rotation_ned(state[3], state[4]) @ state[5:8]))
    return {"time_s": time_s,
            "lateral_error_m": float(normal @ (state[:3] - position)),
            "depth_error_m": float(state[2] - position[2]),
            "pitch_deg": math.degrees(float(state[3])),
            "speed_error_mps": water_speed - 15.}, water_speed


def estimator_observation_row(period, before, after, actuator_before,
                              actuator_after, queue_before, command, plan_commands,
                              sample_after):
    return {
        "time_s": period*.1,
        "lateral_error_m": sample_after["lateral_error_m"],
        "depth_error_m": sample_after["depth_error_m"],
        "speed_error_mps": sample_after["speed_error_mps"],
        "pitch_deg": sample_after["pitch_deg"],
        "thrust_actual_n": actuator_after[0],
        "elevator_actual_deg": math.degrees(actuator_after[1]),
        "rudder_actual_deg": math.degrees(actuator_after[2]),
        "state_before_json": json.dumps(before),
        "state_after_json": json.dumps(after),
        "actuator_before_json": json.dumps(actuator_before),
        "queue_before_json": json.dumps(queue_before),
        "command_json": json.dumps(command),
        "mpc_base_command_json": json.dumps(command),
        "mpc_plan_commands_json": json.dumps(plan_commands),
    }


def policy_reward(weights, progress_delta, sample, command, intent_change,
                  fallback, failed):
    command = np.asarray(command, dtype=float)
    saturation = bool(((command >= SCALE-1e-6) |
                       (command <= np.asarray([0., -SCALE[1], -SCALE[2]])+1e-6)).any())
    return (weights["progress_weight"]*progress_delta/1.5
            - weights["lateral_squared_weight"]*(sample["lateral_error_m"]/8.)**2
            - weights["depth_squared_weight"]*(sample["depth_error_m"]/5.)**2
            - weights["pitch_squared_weight"]*(sample["pitch_deg"]/40.)**2
            - weights["speed_squared_weight"]*(sample["speed_error_mps"]/2.)**2
            - weights["slowdown_weight"]*max(0., -sample["speed_error_mps"])/2.
            - weights["saturation_weight"]*saturation
            - weights["intent_change_weight"]*abs(intent_change)/4.
            - weights["fallback_weight"]*bool(fallback)
            - weights["failure_weight"]*bool(failed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", choices=[case[0] for case in CASES])
    parser.add_argument("--split", choices=("train", "dev"))
    parser.add_argument("--case-id")
    parser.add_argument("--plant-seed", type=int)
    parser.add_argument("--method", choices=("B0_shared", "B2_gov", "B3_RL"),
                        required=True)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--reward-config", type=Path)
    parser.add_argument("--policy-sample", action="store_true")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--max-periods", type=int, default=0,
                        help="0 means the environment's full fixed maximum horizon")
    parser.add_argument("--backup-mode", choices=("abort", "one_step_verified"),
                        default="abort")
    parser.add_argument("--estimator-mode", choices=("E0", "E1"), default="E0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    freeze = json.loads(args.freeze.read_text())
    config = json.loads(args.config.read_text())
    checkpoint = json.loads(args.checkpoint.read_text())
    if (config["candidate_grid_mps"] != freeze["candidate_grid_mps"] or
            config["candidate_limit_k"] != freeze["candidate_limit_k"] or
            config["screen_budget_ms"] != freeze["screen_budget_ms"] or
            config["estimator_checkpoint_sha256"] != sha(args.checkpoint)):
        raise RuntimeError("candidate configuration differs from frozen inputs")
    policy = None
    reward_weights = None
    if args.method == "B3_RL":
        if (args.policy_checkpoint is None or args.reward_config is None or
                args.estimator_mode != "E1" or
                args.backup_mode != "one_step_verified" or
                tuple(config["candidate_grid_mps"]) != GRID):
            raise ValueError("B3 requires frozen E1, backup, policy, reward and five-action grid")
        reward_weights = json.loads(args.reward_config.read_text())
        saved = torch.load(args.policy_checkpoint, map_location="cpu",
                           weights_only=True)
        if (saved["reward_config_sha256"] != sha(args.reward_config) or
                saved["estimator_checkpoint_sha256"] != sha(args.checkpoint) or
                saved["candidate_config_sha256"] != sha(args.config)):
            raise RuntimeError("policy checkpoint does not match frozen H15 inputs")
        policy = MaskedSpeedActorCritic()
        policy.load_state_dict(saved["model_state_dict"])
        policy.eval()
        torch.manual_seed(args.policy_seed)
    elif args.policy_checkpoint is not None or args.reward_config is not None or args.policy_sample:
        raise ValueError("policy inputs are only valid for B3_RL")
    if args.case is not None and any(value is not None for value in
                                     (args.split, args.case_id, args.plant_seed)):
        raise ValueError("smoke alias cannot be combined with manifest case arguments")
    if args.case is not None:
        label, split, case_id, seed, mismatch = next(
            case for case in CASES if case[0] == args.case)
        current_max, target_fin_gain = (1. if mismatch else 0.), .9
    elif args.split is not None and args.case_id is not None and args.plant_seed is not None:
        split, case_id, seed = args.split, args.case_id, args.plant_seed
        mode = (TRAIN_MODES if split == "train" else DEV_MODES).get(seed)
        if mode is None:
            raise ValueError("plant seed is outside the frozen split")
        label = case_id
        mismatch = bool(mode["target_domain"])
        current_max = float(mode["current_max"])
        target_fin_gain = float(mode.get("target_fin_gain_scale", .9))
    else:
        raise ValueError("provide a smoke alias or split, case ID and plant seed")
    manifest_entry = freeze["manifests"][split]
    manifest_path = ROOT / manifest_entry["path"]
    if sha(manifest_path) != manifest_entry["sha256"]:
        raise RuntimeError("manifest changed")
    case = next(item for item in json.loads(manifest_path.read_text())["cases"]
                if item["id"] == case_id)
    path_file = ROOT / case["path"]
    if sha(path_file) != case["path_sha256"]:
        raise RuntimeError("path changed")
    path = LoopPath(path_file)
    env = create_env(path_file, seed, mismatch, current_max, target_fin_gain)
    if args.backup_mode == "one_step_verified":
        env.mpc.set_command_slew_limits(
            config["command_max_delta_normalized_per_period"])
    estimator = load_frozen_estimator(args.checkpoint)
    cycles, period_samples, substep_samples, backup_log = [], [], [], []
    estimator_log = []
    policy_decisions = []
    held_intent = 15.
    total_policy_reward = 0.
    previous_observation = None
    prediction_parameters = (.85, 2.)
    candidate_empty = False
    backup_used = False
    completed = False
    abort_reason = None
    max_periods = int(env.max_steps[0]) if args.max_periods == 0 else args.max_periods
    if max_periods < 1 or max_periods > int(env.max_steps[0]):
        raise ValueError("invalid maximum period count")
    for period in range(max_periods):
        period_started = time.perf_counter()
        before = env.state[0].tolist()
        progress_before = float(env.progress[0])
        actuator_before = env.actuator[0].tolist()
        queue_before = env.command_history[0].tolist()
        sample_before, water_speed = physical_sample(env, path, period * .1)
        estimator_started = time.perf_counter()
        if args.estimator_mode == "E1" and previous_observation is not None:
            estimate = estimator.update(previous_observation)
        else:
            estimate = estimator._fallback(
                "initial_no_history" if args.estimator_mode == "E1" else
                "gate1_e0_short_history", estimator_started)
        parameters = (estimate.fin_gain, estimate.fin_tau_scale)
        if args.estimator_mode == "E1" and parameters != prediction_parameters:
            env.mpc.dynamics = ActuatorAwareCpuDynamics(.1, *parameters)
            prediction_parameters = parameters
        estimator_ms = (time.perf_counter()-estimator_started)*1000.
        estimator_log.append({"period": period, "source": estimate.source,
                              "fallback_reason": estimate.fallback_reason,
                              "fin_gain": estimate.fin_gain,
                              "fin_tau_scale": estimate.fin_tau_scale,
                              "gain_radius": estimate.gain_radius,
                              "tau_radius": estimate.tau_radius,
                              "samples": estimate.samples,
                              "forward_ms": estimate.forward_ms,
                              "fallback_ms": estimate.fallback_ms,
                              "wall_ms": estimator_ms})
        screen_started = time.perf_counter()
        candidates = [screen_candidate(env, path, candidate, estimate,
                                       config, checkpoint["parameters"])
                      for candidate in config["candidate_grid_mps"]]
        screen_ms = (time.perf_counter()-screen_started)*1000.
        verified = [item["candidate_mps"] for item in candidates
                    if item["physical_accepted"]]
        empty = not verified
        candidate_empty |= empty
        policy_ms = 0.
        raw_rl_intent = None
        intent_change = 0.
        if args.method == "B3_RL":
            policy_started = time.perf_counter()
            if period % 5 == 0 and verified:
                observation, mask = decision_observation(
                    before, sample_before, actuator_before, queue_before, path,
                    progress_before, estimate, candidates, held_intent,
                    previous_observation, env.mpc.dynamics)
                action, logp, value = policy.choose(
                    observation, mask, deterministic=not args.policy_sample)
                prior_intent = held_intent
                held_intent = GRID[action]
                intent_change = held_intent-prior_intent
                policy_decisions.append({"period": period,
                                         "observation": observation.tolist(),
                                         "mask": mask.tolist(), "action_index": action,
                                         "intent_mps": held_intent, "logp": logp,
                                         "value": value, "reward": 0.,
                                         "terminal": False})
            policy_ms = (time.perf_counter()-policy_started)*1000.
            raw_rl_intent = held_intent
            selected = held_intent if held_intent in verified else None
        elif args.method == "B0_shared":
            selected = 15. if 15. in verified else None
        else:
            selected = select_b2(candidates, online=False)
        fallback = None
        command_candidate = next((item for item in candidates
                                  if item["candidate_mps"] == selected), None)
        if selected is None and args.backup_mode == "one_step_verified":
            fixed = next(item for item in candidates if item["candidate_mps"] == 15.)
            backup = screen_one_step_backup(env, path, fixed, estimate, config,
                                            checkpoint["parameters"])
            backup_log.append({"period": period, "candidate_reason":
                               "candidate_empty" if empty else
                               "held_intent_rejected" if args.method == "B3_RL" else
                               "fixed_intent_rejected",
                               "accepted": backup["accepted"],
                               "reason": backup["reason"],
                               "scenarios": backup["scenarios"]})
            backup_used = True
            if backup["accepted"]:
                command_candidate = fixed
                fallback = "one_step_verified_fixed15"
            else:
                fallback = "abort_backup_rejected"
                abort_reason = "backup_rejected"
        elif selected is None:
            fallback = "abort_no_verified_intent"
            backup_used = True
            abort_reason = ("candidate_empty" if empty else
                            "held_intent_rejected" if args.method == "B3_RL" else
                            "fixed_intent_rejected")
        if command_candidate is None:
            command = [0., 0., 0.]
            after = before
            actuator_after = actuator_before
            qp_status = "not_solved_after_abort"
            mpc_ms = 0.
            plant_ms = 0.
            done = True
        else:
            env.set_intent(command_candidate["candidate_mps"])
            if args.backup_mode == "one_step_verified":
                plan = torch.as_tensor(command_candidate["plan_normalized"],
                                       device=env.device, dtype=env.dtype)
                env.mpc.plan[0] = plan
                env.mpc.previous[0] = plan[0]
                env.mpc.last_qp_valid[0] = True
                env.mpc.last_status[0] = command_candidate["qp_status"]
                env.base_command[0] = torch.as_tensor(
                    command_candidate["first_command"], device=env.device,
                    dtype=env.dtype)
                env.last_mpc_ok[0] = True
                mpc_ms = 0.
            else:
                mpc_started = time.perf_counter()
                env.defer_mpc_solve = False
                env.base_command, env.last_mpc_ok = env._solve_mpc()
                env.defer_mpc_solve = True
                mpc_ms = (time.perf_counter()-mpc_started)*1000.
            qp_status = env.mpc.last_status[0]
            command = env.base_command[0].tolist()
            if not bool(env.last_mpc_ok[0]):
                fallback = "abort_qp_invalid"
                backup_used = True
                abort_reason = "qp_invalid"
                after = before
                actuator_after = actuator_before
                plant_ms = 0.
                done = True
            else:
                _, done_tensor, info = env.step_verified_command()
                done = bool(done_tensor[0])
                completed = bool(info["success"][0])
                plant_ms = float(info["timing"]["plant_integration_wall_ms"])
                after = env.state[0].tolist()
                actuator_after = env.actuator[0].tolist()
                for substep in env.strict_substeps:
                    substep_samples.append({"time_s": period*.1+substep["substep"]*.02,
                                            "lateral_error_m": substep["lateral_error_m"][0],
                                            "depth_error_m": substep["depth_error_m"][0],
                                            "pitch_deg": math.degrees(substep["pitch_rad"][0]),
                                            "speed_error_mps": substep["speed_error_mps"][0]})
                if done and not completed:
                    abort_reason = "environment_failed_or_timeout"
        total_ms = (time.perf_counter()-period_started)*1000.
        sample_after, _ = physical_sample(env, path, (period+1)*.1)
        period_samples.append(sample_after)
        if args.method == "B3_RL":
            reward = policy_reward(reward_weights,
                                   float(env.progress[0])-progress_before,
                                   sample_after, command, intent_change,
                                   fallback is not None, done and not completed)
            total_policy_reward += reward
            if policy_decisions:
                policy_decisions[-1]["reward"] += reward
        if done is False or completed:
            previous_observation = estimator_observation_row(
                period, before, after, actuator_before, actuator_after,
                queue_before, command,
                (np.asarray(env.mpc.plan[0].numpy()) * SCALE).tolist(),
                sample_after)
        first = first_physical_violation(substep_samples or period_samples)
        first_event = None if first is None else {"time_s": first[0], "kind": first[1]}
        rejections = [{"candidate_mps": item["candidate_mps"],
                       "reason": item["first_rejection"] if not item["physical_accepted"] else None}
                      for item in candidates]
        timings = dict(zip(TIMING_FIELDS,
                           (estimator_ms, screen_ms, policy_ms, mpc_ms, plant_ms,
                            total_ms)))
        values = (1, args.method, f"{label}_s{seed}", period, period*.1,
                  before, after, actuator_before, actuator_after, queue_before,
                  command, raw_rl_intent, float(env.reference_rate()), water_speed,
                  empty, rejections, selected, fallback, first_event, qp_status,
                  timings)
        cycles.append(dict(zip(FIELDS, values)))
        if done:
            break
    if not cycles:
        raise RuntimeError("no cycle decisions recorded")
    if policy_decisions:
        policy_decisions[-1]["terminal"] = True
    limited = len(cycles) == max_periods and not completed and abort_reason is None
    strict = evaluate(completed=completed, period_samples=period_samples,
                      substep_samples=substep_samples if substep_samples else None,
                      mpc_fallback=backup_used, candidate_empty=candidate_empty,
                      backup_used=backup_used, deadline_missed=True,
                      fallback_defined=True)
    if limited:
        strict["strict_mainline"] = None
    args.out.mkdir(parents=True, exist_ok=True)
    write_cycles(args.out / "cycle_log.jsonl", cycles)
    (args.out / "backup_log.json").write_text(
        json.dumps(backup_log, indent=2, allow_nan=False) + "\n")
    (args.out / "backup_mode.json").write_text(
        json.dumps({"mode": args.backup_mode,
                    "policy": "fixed 15 first MPC command; five-scenario one-step replay",
                    "strict_mainline_on_backup": False}, indent=2) + "\n")
    (args.out / "estimator_mode.json").write_text(
        json.dumps({"mode": args.estimator_mode,
                    "checkpoint_sha256": sha(args.checkpoint),
                    "prediction_uses_estimate": args.estimator_mode == "E1"},
                   indent=2) + "\n")
    (args.out / "estimator_log.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in estimator_log))
    if args.method == "B3_RL":
        (args.out / "policy_decisions.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n"
                    for row in policy_decisions))
        (args.out / "policy_mode.json").write_text(json.dumps(
            {"checkpoint": str(args.policy_checkpoint),
             "checkpoint_sha256": sha(args.policy_checkpoint),
             "reward_config": str(args.reward_config),
             "reward_config_sha256": sha(args.reward_config),
             "selection": "sample" if args.policy_sample else "deterministic",
             "policy_seed": args.policy_seed, "decision_periods": 5,
             "action_grid_mps": GRID}, indent=2) + "\n")
    (args.out / "substep_samples.json").write_text(
        json.dumps(substep_samples, indent=2, allow_nan=False) + "\n")
    summary = {"scope": "offline closed-loop mechanism smoke; no 5 ms claim",
               "method": args.method, "case": label, "seed": seed,
               "periods": len(cycles), "completed": completed,
               "limited_diagnostic": limited, "abort_reason": abort_reason,
               "candidate_empty": candidate_empty,
               "backup_used": backup_used, "strict_hist": strict["strict_hist"],
               "backup_periods": sum(row["fallback"] == "one_step_verified_fixed15"
                                     for row in cycles),
               "estimator_mode": args.estimator_mode,
               "estimator_source_counts": dict(Counter(
                   row["source"] for row in estimator_log)),
               "estimator_fallback_reasons": dict(Counter(
                   row["fallback_reason"] for row in estimator_log
                   if row["fallback_reason"] is not None)),
               "strict_mainline": strict["strict_mainline"],
               "first_physical_violation": strict["first_substep_violation"],
               "final_progress_m": float(env.progress[0]),
               "path_length_m": float(env.paths.lengths[0]),
               "max_abs_lateral_m": max(abs(x["lateral_error_m"]) for x in substep_samples)
                                    if substep_samples else None,
               "max_abs_depth_m": max(abs(x["depth_error_m"]) for x in substep_samples)
                                  if substep_samples else None,
               "max_abs_speed_error_mps": max(abs(x["speed_error_mps"])
                                              for x in substep_samples) if substep_samples else None,
               "selected_intents_mps": [x["selected_candidate_mps"] for x in cycles],
               "policy_decision_count": len(policy_decisions) if policy is not None else None,
               "policy_reward": total_policy_reward if policy is not None else None,
               "timing_ms": timing_summary(cycles)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    results = dict.fromkeys(RESULT_FIELDS)
    results.update(episodes=1, strict_hist_completed=int(strict["strict_hist"]),
                   strict_mainline_completed=(None if limited or strict["strict_mainline"] is None
                                              else int(strict["strict_mainline"])),
                   completed_with_backup=int(bool(strict["completed_with_backup"])),
                   fallback_episodes=int(backup_used),
                   empty_candidate_episodes=int(candidate_empty),
                   first_violations={} if first is None else {first[1]: 1})
    sources = ("run_offline_governor_smoke.py", "candidate_filter.py",
               "progress_reference.py", "cycle_log.py", "cycle_writer.py")
    record = dict(zip(TOP_LEVEL, (1, args.method,
                                  {name: sha(ROOT / f"experiments/h15_mainline/{name}")
                                   for name in sources} |
                                  {"candidate_filter_config.json": sha(args.config),
                                   "checkpoint.json": sha(args.checkpoint),
                                   "backup_mode.json": sha(args.out / "backup_mode.json"),
                                   "estimator_mode.json": sha(args.out / "estimator_mode.json"),
                                   "quick_run_manifest.json": sha(args.freeze)} |
                                  ({"policy_checkpoint.pt": sha(args.policy_checkpoint),
                                    "reward_config.json": sha(args.reward_config),
                                    "policy_mode.json": sha(args.out / "policy_mode.json"),
                                    "masked_speed_ppo.py": sha(ROOT / "experiments/h15_mainline/masked_speed_ppo.py")}
                                   if policy is not None else {}),
                                  {f"{split}_manifest.json": sha(manifest_path),
                                   "path.csv": sha(path_file)}, [seed],
                                  {key: summary["timing_ms"][key] for key in TIMING_STAGES},
                                  results)))
    write_run(args.out / "run.json", record)
    (args.out / "console.log").write_text(
        json.dumps({key: summary[key] for key in
                    ("method", "case", "periods", "completed", "abort_reason",
                     "candidate_empty", "final_progress_m")}, indent=2) + "\n")
    print((args.out / "console.log").read_text())


if __name__ == "__main__":
    main()
