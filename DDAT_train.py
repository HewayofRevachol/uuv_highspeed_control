"""Train DDAT on screened UUV paths with multiscale preview and course reward.

The UUV policy receives a learned feedforward trajectory embedding, body-frame
tracking error, current state and a filtered nominal-model error estimate. It
sets desired thrust and pitch/yaw rates. Nominal inverse dynamics and the
existing actuator model form the low-level controller. The current estimate is
derived from position drift; the true current is hidden from the policy.

Default experiment: randomize plant and actuator parameters, keep the 8/20
training paths passing the steady-trim screen at all 16 drag/thrust/fin-gain
range corners, use 12 preview points over 0–160 m, and penalize ground-course
error instead of body-heading error. MPC uses the filtered actuator-delay
predictor by default, matching animate_mpc_tracking.py. Currents and process
disturbances remain.
Use --list-training-paths to inspect selection without starting training.
The screen does not certify transient feasibility.
"""

import argparse
import csv
import json
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from simulator.batch_path_env import PathBank
from simulator.ddat_recovery import MPCFaultInjector, RecoveryBuffer, recovery_reward, restore
from simulator.model import MISMATCHED_PLANT
from research.controllers.ddat import ExperimentDDATPathEnv
from research.networks.ddat import ActorCritic
from research.variants import controller_class, network_class


ROOT = Path(__file__).resolve().parent
DEFAULT_PATH_SET = ROOT / "DDAT_trajectories"


def select_paths(files, feasible_only, randomize_plant=False):
    """Filter sustained steady command-limit violations; keep audit metadata."""
    if not feasible_only:
        return files, []
    from high_speed_experiments.screen_dynamic_feasibility import screen_path

    plant = MISMATCHED_PLANT
    if randomize_plant:
        # BatchPathResidualEnv.reset multiplies the five baseline parameters
        # by [.97,1.03], [.93,1.07], [.95,1.05], [.90,1.10], [.90,1.10].
        # Mass does not alter a zero-acceleration trim. Delays/time constants
        # affect transients, which this steady screen cannot certify.
        # Test the expected worst corner first, then every other gain corner.
        plant = replace(plant, mass_scale=plant.mass_scale*1.03,
                        damping_scale=plant.damping_scale*1.07,
                        thrust_gain=plant.thrust_gain*.95,
                        elevator_gain=plant.elevator_gain*.90,
                        rudder_gain=plant.rudder_gain*.90)
    screening = []
    for file in files:
        result = screen_path(file, plant)
        result["parameter_corners_checked"] = 1
        if result["status"] == "not_screened_out" and result["unresolved_steady_samples"]:
            result["status"] = "screened_out"
            result["reason"] = "unresolved_steady_trim"
        if randomize_plant and result["status"] == "not_screened_out":
            for damping, thrust, elevator, rudder in product(
                    (.93, 1.07), (.95, 1.05), (.90, 1.10), (.90, 1.10)):
                if (damping, thrust, elevator, rudder) == (1.07, .95, .90, .90):
                    continue
                corner = replace(MISMATCHED_PLANT,
                                 damping_scale=MISMATCHED_PLANT.damping_scale*damping,
                                 thrust_gain=MISMATCHED_PLANT.thrust_gain*thrust,
                                 elevator_gain=MISMATCHED_PLANT.elevator_gain*elevator,
                                 rudder_gain=MISMATCHED_PLANT.rudder_gain*rudder)
                checked = screen_path(file, corner)
                result["parameter_corners_checked"] += 1
                if (checked["status"] == "screened_out" or
                        checked["unresolved_steady_samples"]):
                    result["status"] = "screened_out"
                    result["reason"] = "randomized_parameter_corner_limit_or_unresolved"
                    result["failed_corner"] = {
                        "damping_multiplier": damping,
                        "thrust_multiplier": thrust,
                        "elevator_multiplier": elevator,
                        "rudder_multiplier": rudder,
                        "screen_result": checked,
                    }
                    break
        screening.append(result)
    selected = [file for file, result in zip(files, screening)
                if result["status"] == "not_screened_out"]
    if not selected:
        raise ValueError("steady-feasibility screen retained no paths")
    return selected, screening


def advantages(rewards, values, next_values, done, terminated,
               gamma=.99, lam=.95):
    """Bootstrap time limits; stop GAE recursion across every reset."""
    result = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma*next_values[t]*(~terminated[t]) - values[t]
        gae = delta + gamma*lam*(~done[t])*gae
        result[t] = gae
    return result, result+values


def ppo_update(agent, optimizer, batch, epochs=4, minibatch=2048,
               clip=.2, value_weight=.5, entropy_weight=.001):
    observations, actions, old_logp, returns, adv = batch
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    count = len(observations)
    for _ in range(epochs):
        for indices in torch.randperm(count, device=observations.device).split(minibatch):
            obs, act = observations[indices], actions[indices]
            logp = agent.log_prob(obs, act)
            ratio = (logp-old_logp[indices]).exp()
            policy_loss = -torch.minimum(ratio*adv[indices],
                                         ratio.clamp(1-clip, 1+clip)*adv[indices]).mean()
            value = agent.value(obs)
            value_loss = (value-returns[indices]).square().mean()
            entropy = agent.distribution(obs).entropy().sum(-1).mean()
            loss = policy_loss + value_weight*value_loss-entropy_weight*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), .5)
            optimizer.step()


@torch.no_grad()
def evaluate(agent, env, steps=100, seed=None):
    if seed is not None:
        env.generator.manual_seed(seed)
    env.reset()
    reward_sum = torch.zeros(env.batch_size, device=env.device)
    error_sum = torch.zeros_like(reward_sum)
    failures = torch.zeros_like(reward_sum)
    mpc_failures = torch.zeros_like(reward_sum)
    for _ in range(steps):
        observation = env.observe()
        action = (agent.deterministic(observation) if agent else
                  torch.zeros(env.batch_size, 3, device=env.device))
        _, reward, done, info = env.step(action)
        reward_sum += reward
        error_sum += info["position_error_m"].square()
        failures += info["failed"].float()
        mpc_failures += (~info["mpc_ok"]).float()
        if done.any():
            env.reset(done)
    return {"mean_reward": reward_sum.mean().item()/steps,
            "position_rmse_m": torch.sqrt(error_sum.mean()/steps).item(),
            "failure_fraction": failures.sum().item()/(env.batch_size*steps),
            "mpc_fallback_fraction": mpc_failures.sum().item()/(env.batch_size*steps)}


@torch.no_grad()
def evaluate_paths(agent, bank, args, seed):
    """Run each supplied path once, returning SAC-compatible per-path metrics."""
    rows = []
    environment_class = controller_class(vars(args))
    for start in range(0, len(bank.files), args.eval_envs):
        count = min(args.eval_envs, len(bank.files)-start)
        env = environment_class(
            bank, count, device=args.device, horizon=args.horizon,
            backend=args.backend, seed=seed+start, randomize_plant=False,
            admm_iterations=args.admm_iterations,
            admm_rho=getattr(args, "admm_rho", 2.),
            adaptation_time=args.adaptation_time,
            current_max_mps=args.current_max_mps,
            current_filter_time=args.current_filter_time,
            attitude_pitch_weight=getattr(args, "attitude_pitch_weight", 0.),
            attitude_yaw_weight=getattr(args, "attitude_yaw_weight", 0.),
            fsp_enabled=getattr(args, "fsp_enabled", True),
            fsp_filter_time=getattr(args, "fsp_filter_time", .4),
            fsp_preview_time=getattr(args, "fsp_preview_time", None),
            preview_mode=getattr(args, "preview_mode", "dense"),
            yaw_reward=getattr(args, "yaw_reward", "heading"))
        observation = env.reset(torch.arange(start, start+count, device=env.device))
        episode_return = torch.zeros(count, device=env.device)
        squared_error = torch.zeros_like(episode_return)
        steps = torch.zeros(count, device=env.device)
        fallbacks = torch.zeros_like(episode_return)
        active = torch.ones(count, device=env.device, dtype=torch.bool)
        succeeded = torch.zeros_like(active)
        failed = torch.zeros_like(active)
        timed_out = torch.zeros_like(active)
        while active.any():
            action = (agent.deterministic(observation) if agent is not None else
                      torch.zeros(count, 3, device=env.device))
            observation, reward, done, info = env.step(action)
            episode_return += reward
            squared_error += torch.where(active, info["position_error_m"].square(), 0)
            fallbacks += active.float()*(~info["mpc_ok"]).float()
            steps += active.float()
            succeeded |= info["success"]
            failed |= info["failed"]
            timed_out |= info["timeout"]
            active &= ~done
        for index in range(count):
            rows.append({"path": str(bank.files[start+index]),
                         "return": episode_return[index].item(),
                         "success": int(succeeded[index].item()),
                         "failed": int(failed[index].item()),
                         "timeout": int(timed_out[index].item()),
                         "progress_fraction": (env.progress[index]/bank.lengths[start+index]).item(),
                         "position_rmse_m": torch.sqrt(squared_error[index]/steps[index]).item(),
                         "mpc_fallback_fraction": (fallbacks[index]/steps[index]).item()})
    return rows


def summarize_paths(rows):
    return {"mean_return": float(np.mean([row["return"] for row in rows])),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_position_rmse_m": float(np.mean([row["position_rmse_m"] for row in rows])),
            "mpc_fallback_fraction": float(np.mean([row["mpc_fallback_fraction"] for row in rows]))}


class PathSampler:
    """Cycle through every training path once before reshuffling the pool."""

    def __init__(self, rng, path_count):
        if path_count < 1:
            raise ValueError("path_count must be positive")
        self.rng = rng
        self.path_count = path_count
        self.order = rng.permutation(path_count)
        self.cursor = 0

    def sample(self, count):
        choices = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            if self.cursor == self.path_count:
                self.order = self.rng.permutation(self.path_count)
                self.cursor = 0
            take = min(count-filled, self.path_count-self.cursor)
            choices[filled:filled+take] = self.order[self.cursor:self.cursor+take]
            self.cursor += take
            filled += take
        return choices


def initialize_actor(agent, checkpoint_path, preview_mode="dense"):
    """Warm-start the 112-input actor from either a 107- or 112-input model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "DDAT_UUV_current":
        raise ValueError("--init-actor must be a current-aware DDAT checkpoint")
    previous_preview = checkpoint.get("config", {}).get("preview_mode", "dense")
    if previous_preview != preview_mode:
        raise ValueError("--init-actor preview offsets differ; use the same "
                         "--preview-mode or train this experiment from scratch")
    source = checkpoint["agent"]
    target = agent.state_dict()
    if source.keys() != target.keys():
        raise ValueError("--init-actor has a different network architecture")
    for name, value in source.items():
        if value.shape == target[name].shape:
            target[name] = value
        elif (name in ("actor.0.weight", "critic.0.weight") and
              value.shape[0] == target[name].shape[0] and
              value.shape[1]+5 == target[name].shape[1]):
            target[name].zero_()
            target[name][:, :value.shape[1]] = value
        else:
            raise ValueError(f"--init-actor tensor shape mismatch: {name}")
    agent.load_state_dict(target)


def save_final_model(path, agent, optimizer, env, args, rng, sampler, update, env_steps,
                     training_seconds, ids,
                     episode_return, current_observation):
    """Save the final policy and the full state needed for a future continuation."""
    environment_fields = ("state", "parameters", "actuator", "command_history",
                          "delay_steps", "tau", "disturbance", "previous_action",
                          "base_command", "last_mpc_ok", "mpc_failure_streak",
                          "path_ids", "progress",
                          "steps", "max_steps", "finished", "disturbance_estimate",
                          "current_ned", "current_estimate")
    environment = {name: getattr(env, name).detach().cpu().clone()
                   for name in environment_fields}
    environment["mpc"] = {name: getattr(env.mpc, name).detach().cpu().clone()
                          for name in ("previous", "plan", "states",
                                       "last_primal_residual", "last_dual_residual",
                                       "last_qp_valid")
                          if hasattr(env.mpc, name)}
    if env.fsp is not None:
        environment["fsp"] = {
            name: getattr(env.fsp, name).detach().cpu().clone()
            for name in ("velocity_error", "expected_next", "expected_valid")}
    environment["generator_state"] = env.generator.get_state()
    payload = {
        "format_version": 6,
        "architecture": "DDAT_UUV_current",
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "observation_dim": env.observation_size,
        "policy_output": "thrust_delta_N_and_pitch_yaw_rate_delta_rad_s",
        "training_step": env_steps,
        "update": update,
        "wall_seconds": training_seconds,
        "config": vars(args),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (torch.cuda.get_rng_state_all()
                            if args.device == "cuda" else None),
        "environment": environment,
        "path_selection": {"ids": torch.from_numpy(ids.copy()),
                           "sampler_order": torch.from_numpy(sampler.order.copy()),
                           "sampler_cursor": sampler.cursor,
                           "episode_return": episode_return.detach().cpu().clone(),
                           "observation": current_observation.detach().cpu().clone()},
    }
    temporary = path.with_name(path.stem+".tmp"+path.suffix)
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter; use --device cpu")
    if min(args.envs, args.eval_envs, args.rollout, args.total_steps,
           args.eval_episodes, args.torch_threads,
           args.minibatch, args.epochs) < 1:
        raise ValueError("counts and intervals must be positive")
    if not (0 <= args.recovery_ratio <= 1 and 0 <= args.fault_ratio <= 1):
        raise ValueError("recovery/fault ratios must be between 0 and 1")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    path_dir = Path(args.path_dir)
    available_paths = sorted((path_dir/"train").glob("*.csv")) + sorted(path_dir.glob("*.csv"))
    if not available_paths:
        raise FileNotFoundError(f"no training CSV paths found in {path_dir}")
    train_paths, screening = select_paths(
        available_paths, args.feasible_only, args.randomize_plant)
    train_bank = PathBank(train_paths, device)
    env = controller_class(vars(args))(
        train_bank, args.envs, device=device, horizon=args.horizon,
        backend=args.backend, seed=args.seed,
        randomize_plant=args.randomize_plant,
        admm_iterations=args.admm_iterations,
        admm_rho=args.admm_rho,
        adaptation_time=args.adaptation_time,
        current_max_mps=args.current_max_mps,
        current_filter_time=args.current_filter_time,
        attitude_pitch_weight=args.attitude_pitch_weight,
        attitude_yaw_weight=args.attitude_yaw_weight,
        fsp_enabled=args.fsp_enabled,
        fsp_filter_time=args.fsp_filter_time,
        fsp_preview_time=args.fsp_preview_time,
        preview_mode=args.preview_mode, yaw_reward=args.yaw_reward)
    agent = network_class(vars(args))(env.observation_size).to(device)
    if args.init_actor:
        initialize_actor(agent, args.init_actor, args.preview_mode)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    recovery = (RecoveryBuffer(args.envs, args.recovery_lookback_min,
                               args.recovery_lookback_max,
                               args.recovery_buffer_size, args.seed+1)
                if args.recovery_training else None)
    faults = MPCFaultInjector(args.envs, device) if recovery is not None else None
    run_dir = (Path(args.output) if args.output else
               ROOT/"outputs/DDAT"/datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/"config.json").write_text(json.dumps(vars(args), indent=2)+"\n")
    (run_dir/"training_paths.json").write_text(json.dumps({
        "paths": [str(path.resolve()) for path in train_paths],
        "feasibility_screen": screening,
        "screen_limitation": "steady trim only; passing does not prove transient feasibility",
        "plant_randomization": args.randomize_plant,
        "screen_plant": ("all_randomized_gain_corners"
                         if args.randomize_plant else "fixed_mismatched"),
    }, indent=2)+"\n")
    history_path = run_dir/"history.csv"
    reward_path = run_dir/"step_rewards.csv"
    episode_path = run_dir/"episode_rewards.csv"
    log_mpc_diagnostics = getattr(args, "log_mpc_diagnostics", False)
    diagnostic_path = run_dir/"mpc_diagnostics.csv"
    history_fields = (
        "update", "env_steps", "wall_seconds", "steps_per_second", "train_reward",
        "recovery_cases")
    if log_mpc_diagnostics:
        history_fields += ("mpc_fallback_fraction", "mpc_qp_invalid_fraction",
                           "mpc_primal_residual_mean", "mpc_primal_residual_max",
                           "mpc_dual_residual_mean", "mpc_dual_residual_max")
    training_seconds = 0.
    update = 0
    env_steps = 0
    sampler = PathSampler(rng, len(train_paths))
    ids = sampler.sample(args.envs)
    current_observation = env.reset(torch.as_tensor(ids, device=device))
    episode_return = torch.zeros(args.envs, device=device)
    episode_source = np.full(args.envs, "normal", dtype=object)
    episode_start_fraction = np.zeros(args.envs)
    with (tqdm(total=args.total_steps,
               desc="Training", unit="env step", dynamic_ncols=True,
               mininterval=1.0) as progress,
          history_path.open("w", newline="") as history_stream,
          reward_path.open("w", newline="") as reward_stream,
          episode_path.open("w", newline="") as episode_stream,
          (diagnostic_path.open("w", newline="") if log_mpc_diagnostics
           else nullcontext()) as diagnostic_stream):
        history = csv.DictWriter(history_stream, fieldnames=history_fields)
        rewards_writer = csv.writer(reward_stream)
        episodes = csv.writer(episode_stream)
        diagnostics_writer = (csv.writer(diagnostic_stream)
                              if diagnostic_stream is not None else None)
        history.writeheader()
        rewards_writer.writerow(("step", "reward"))
        episodes.writerow(("end_step", "path", "return", "success",
                          "progress_fraction", "source", "start_progress_fraction"))
        if diagnostics_writer is not None:
            diagnostics_writer.writerow((
                "env_step", "env_index", "mpc_fallback", "mpc_qp_valid",
                "mpc_failure_streak", "mpc_primal_residual", "mpc_dual_residual"))
        while env_steps < args.total_steps:
            update += 1
            start = time.perf_counter()
            rollout = [[] for _ in range(7)]
            terminal_masks = []
            rollout_rewards = []
            diagnostic_steps = []
            count = min(args.rollout, int(np.ceil((args.total_steps-env_steps)/args.envs)))
            for _ in range(count):
                if recovery is not None:
                    recovery.record(env)
                obs = current_observation
                with torch.no_grad():
                    action, logp, value = agent.sample(obs)
                    next_obs, reward, done, info = env.step(action)
                    if log_mpc_diagnostics:
                        diagnostic_steps.append(torch.stack((
                            ~info["mpc_ok"], info["mpc_qp_valid"],
                            info["mpc_failure_streak"],
                            info["mpc_primal_residual"],
                            info["mpc_dual_residual"]), -1))
                    if recovery is not None:
                        recovery.collect(env, info["failed"])
                        reward = reward+recovery_reward(env)
                episode_return += reward
                rollout_rewards.append(reward)
                env_steps += args.envs
                progress.update(min(args.envs, args.total_steps-progress.n))
                if done.any():
                    finished = done.nonzero().flatten().cpu().numpy()
                    for index in finished:
                        episodes.writerow((env_steps-args.envs+index+1,
                                          str(train_bank.files[ids[index]]),
                                          episode_return[index].item(),
                                          int(info["success"][index].item()),
                                          info["progress_fraction"][index].item(),
                                          episode_source[index],
                                          episode_start_fraction[index]))
                    episode_return[done] = 0
                    if recovery is not None:
                        recovery.reset_rows(done)
                        faults.clear(done)
                    ids[finished] = sampler.sample(len(finished))
                    current_observation = env.reset(torch.as_tensor(ids, device=device), done)
                    episode_source[finished] = "normal"
                    if recovery is not None and len(recovery):
                        selected = finished[rng.random(len(finished)) < args.recovery_ratio]
                        if len(selected):
                            restore(env, selected, recovery.sample(len(selected)))
                            ids[selected] = env.path_ids[selected].cpu().numpy()
                            episode_source[selected] = "recovery"
                            faulted = selected[rng.random(len(selected)) < args.fault_ratio]
                            if len(faulted):
                                faults.schedule(env, faulted, rng)
                                episode_source[faulted] = "recovery_fault"
                    episode_start_fraction[finished] = (
                        env.progress[finished]/env.paths.lengths[env.path_ids[finished]]
                    ).cpu().numpy()
                else:
                    current_observation = next_obs
                if recovery is not None:
                    faults.apply(env)
                    current_observation = env.observe()
                with torch.no_grad():
                    next_value = agent.value(current_observation)
                for target, value_to_store in zip(rollout,
                                                  (obs, action, logp, value, reward,
                                                   next_value, done)):
                    target.append(value_to_store.detach())
                # The SAC task treats success, failure and timeout as terminal.
                terminal_masks.append(done.detach())
            obs, action, logp, value, reward, next_value, done = [torch.stack(x) for x in rollout]
            terminated = torch.stack(terminal_masks)
            adv, ret = advantages(reward, value, next_value, done, terminated)
            flat = lambda x: x.reshape(-1, *x.shape[2:])
            ppo_update(agent, optimizer, (flat(obs), flat(action), flat(logp),
                                          flat(ret), flat(adv)),
                       epochs=args.epochs, minibatch=args.minibatch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_seconds += time.perf_counter()-start
            first_step = env_steps-count*args.envs+1
            for index, item in enumerate(torch.stack(rollout_rewards).flatten().cpu().tolist()):
                rewards_writer.writerow((first_step+index, item))
            if diagnostics_writer is not None:
                diagnostic_values = torch.stack(diagnostic_steps).cpu().numpy()
                for step_index, values in enumerate(diagnostic_values):
                    for env_index, (fallback, valid, streak, primal, dual) in enumerate(values):
                        diagnostics_writer.writerow((
                            first_step+step_index*args.envs+env_index, env_index,
                            int(fallback), int(valid), int(streak), primal, dual))
            row = {"update": update, "env_steps": env_steps,
                   "wall_seconds": training_seconds,
                   "steps_per_second": env_steps/training_seconds,
                   "train_reward": reward.mean().item(),
                   "recovery_cases": len(recovery) if recovery is not None else 0}
            if diagnostics_writer is not None:
                row.update({
                    "mpc_fallback_fraction": diagnostic_values[:, :, 0].mean(),
                    "mpc_qp_invalid_fraction": 1-diagnostic_values[:, :, 1].mean(),
                    "mpc_primal_residual_mean": diagnostic_values[:, :, 3].mean(),
                    "mpc_primal_residual_max": diagnostic_values[:, :, 3].max(),
                    "mpc_dual_residual_mean": diagnostic_values[:, :, 4].mean(),
                    "mpc_dual_residual_max": diagnostic_values[:, :, 4].max(),
                })
            history.writerow(row)
            history_stream.flush()
            reward_stream.flush()
            episode_stream.flush()
            if diagnostic_stream is not None:
                diagnostic_stream.flush()
            progress.set_postfix(train_reward=f"{row['train_reward']:.2f}")
            progress.write(json.dumps(row))
    save_final_model(run_dir/"final_actor.pt", agent, optimizer, env, args, rng,
                     sampler, update, env_steps, training_seconds, ids,
                     episode_return, current_observation)
    print(f"final model saved: {run_dir/'final_actor.pt'}", flush=True)
    return history_path


def evaluate_actor(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter; use --device cpu")
    model = torch.load(args.eval_actor, map_location=args.device, weights_only=True)
    config = model["config"]
    for key in ("horizon", "backend", "admm_iterations", "adaptation_time",
                "current_max_mps", "current_filter_time", "seed"):
        setattr(args, key, config[key])
    # Old checkpoints did not record these experiment switches.
    args.feasible_only = config.get("feasible_only", False)
    args.preview_mode = config.get("preview_mode", "dense")
    args.yaw_reward = config.get("yaw_reward", "heading")
    args.attitude_pitch_weight = config.get("attitude_pitch_weight", 0.)
    args.attitude_yaw_weight = config.get("attitude_yaw_weight", 0.)
    args.admm_rho = config.get("admm_rho", 2.)
    args.fsp_enabled = config.get("fsp_enabled", False)
    args.fsp_filter_time = config.get("fsp_filter_time", .4)
    args.fsp_preview_time = config.get("fsp_preview_time")
    args.path_dir = config.get("path_dir", config.get("corpus_dir", args.path_dir))
    args.controller_module = config.get("controller_module", "research.controllers.ddat")
    args.network_module = config.get("network_module", "research.networks.ddat")
    agent = network_class(config)(model["observation_dim"]).to(args.device)
    agent.load_state_dict(model["agent"])
    agent.eval()
    files = sorted((Path(args.path_dir)/"val").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no validation CSV paths found in {args.path_dir}")
    files, _ = select_paths(files, args.feasible_only)
    indices = np.linspace(0, len(files)-1, min(args.eval_episodes, len(files)), dtype=int)
    bank = PathBank([files[index] for index in indices], args.device)
    result = {"training_step": model["training_step"],
              "fsp_enabled": args.fsp_enabled,
              "fsp_filter_time": args.fsp_filter_time,
              "fsp_preview_time": args.fsp_preview_time,
              "policy": summarize_paths(evaluate_paths(agent, bank, args,
                                                       args.seed+100_000)),
              "mpc_only": summarize_paths(evaluate_paths(None, bank, args,
                                                         args.seed+100_000))}
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("admm", "osqp", "rti_cpu", "rti_cuda"),
                        default="admm")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--attitude-pitch-weight", type=float, default=100.,
                        help="MPC cost weight for explicit path pitch tracking")
    parser.add_argument("--attitude-yaw-weight", type=float, default=30.,
                        help="MPC cost weight for explicit path yaw tracking")
    parser.add_argument("--admm-iterations", type=int, default=400)
    parser.add_argument("--admm-rho", type=float, default=10.,
                        help="ADMM penalty parameter; try 10 for fewer MPC fallbacks")
    parser.add_argument("--fsp-enabled", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="use the filtered actuator-delay predictor in MPC")
    parser.add_argument("--fsp-filter-time", type=float, default=.4,
                        help="seconds for filtering historical velocity prediction error")
    parser.add_argument("--fsp-preview-time", type=float, default=None,
                        help="prediction lead in seconds; default uses nominal fin delay plus lag")
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=240_000*5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1027)
    parser.add_argument("--torch-threads", type=int, default=5)
    parser.add_argument("--path-dir", default=str(DEFAULT_PATH_SET))
    parser.add_argument("--controller-module", default="research.controllers.ddat",
                        help="research.* module defining ExperimentDDATPathEnv")
    parser.add_argument("--network-module", default="research.networks.ddat",
                        help="research.* module defining ActorCritic")
    parser.add_argument("--feasible-only", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="train/evaluate only paths passing the steady command-limit screen")
    parser.add_argument("--randomize-plant", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="enable parameter randomization and use a conservative screen for its parameter range")
    parser.add_argument("--preview-mode", choices=("dense", "multiscale"),
                        default="multiscale",
                        help="12-point preview: dense 0–44 m, multiscale 0–160 m")
    parser.add_argument("--yaw-reward", choices=("heading", "course"),
                        default="course",
                        help="penalize body heading or ground-course error to path tangent")
    parser.add_argument("--list-training-paths", action="store_true",
                        help="print the selected train paths and exit without training")
    parser.add_argument("--adaptation-time", type=float, default=.5,
                        help="seconds for the filtered model-error estimate")
    parser.add_argument("--current-max-mps", type=float, default=1.,
                        help="maximum per-episode steady horizontal current speed; 0 disables it")
    parser.add_argument("--current-filter-time", type=float, default=1.,
                        help="seconds for the position-drift current estimate")
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--recovery-training", action="store_true",
                        help="mix normal starts with pre-failure states and MPC-command faults")
    parser.add_argument("--recovery-ratio", type=float, default=.5,
                        help="fraction of resets drawn from captured pre-failure states")
    parser.add_argument("--fault-ratio", type=float, default=.4,
                        help="fraction of recovery starts with a short command fault")
    parser.add_argument("--recovery-lookback-min", type=int, default=5)
    parser.add_argument("--recovery-lookback-max", type=int, default=20)
    parser.add_argument("--recovery-buffer-size", type=int, default=2048)
    parser.add_argument("--init-actor",
                        help="warm-start from an existing 107- or 112-input DDAT checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--eval-actor", help="Evaluate a completed final_actor.pt")
    args = parser.parse_args()
    if not np.isfinite(args.admm_rho) or args.admm_rho <= 0:
        parser.error("--admm-rho must be finite and positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.list_training_paths:
        path_dir = Path(arguments.path_dir)
        files = sorted((path_dir/"train").glob("*.csv")) + sorted(path_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no training CSV paths found in {path_dir}")
        selected, screening = select_paths(
            files, arguments.feasible_only, arguments.randomize_plant)
        print(json.dumps({"selected_count": len(selected), "available_count": len(files),
                          "selected_paths": [str(file) for file in selected],
                          "screen_plant": ("all_randomized_gain_corners"
                                           if arguments.randomize_plant else "fixed_mismatched"),
                          "feasibility_screen": screening}, ensure_ascii=False, indent=2))
    elif arguments.eval_actor:
        evaluate_actor(arguments)
    else:
        train(arguments)
