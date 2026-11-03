"""PPO reference-speed policy wrapped around the existing FSP + MPC controller.

The actor changes one high-level speed reference; the MPC still computes all
actuator commands. This is a research prototype, not a safety certificate.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DDAT_train import ExperimentDDATPathEnv, advantages, ppo_update
from simulator.batch_path_env import PathBank


class ReferencePolicyEnv(ExperimentDDATPathEnv):
    """One actor action controls a bounded reference speed or rule residual."""

    observation_size = ExperimentDDATPathEnv.observation_size + 1

    def __init__(self, paths, batch_size=8, device="cpu",
                 saturation_penalty=0., base_reference="task",
                 residual_scale=2., residual_penalty=0., **kwargs):
        if base_reference not in ("task", "curvature"):
            raise ValueError("base_reference must be task or curvature")
        self.reference_speed = torch.zeros(batch_size, device=device,
                                           dtype=paths.speed.dtype)
        self.policy_residual = torch.zeros_like(self.reference_speed)
        self.saturation_penalty = saturation_penalty
        self.base_reference = base_reference
        self.policy_residual_scale = residual_scale
        self.residual_penalty = residual_penalty
        super().__init__(paths, batch_size=batch_size, device=device, **kwargs)
        self.observation_size = type(self).observation_size

    def _scheduled_speed(self, speed):
        return self.reference_speed[:, None].expand_as(speed)

    def _reference(self, time_offset=0.):
        if self.base_reference == "curvature":
            target = self.paths.speed[self.path_ids, 0]
            ahead = self.progress[:, None] + self.model.tensor([0., 10., 20., 30.])
            _, _, _, yaw_ahead, _ = self.paths.sample(self.path_ids, ahead)
            delta = torch.atan2(torch.sin(yaw_ahead[:, 1:] - yaw_ahead[:, :-1]),
                                torch.cos(yaw_ahead[:, 1:] - yaw_ahead[:, :-1]))
            curvature = (delta.abs() / 10.).amax(-1).clamp_min(1e-6)
            cap = torch.sqrt(6. / curvature).clamp_max(18.)
            self.reference_speed = torch.minimum(target, cap).add(
                self.policy_residual).clamp(9., 18.)
        offsets = torch.arange(1, self.mpc.horizon + 1, device=self.device,
                               dtype=self.dtype) * self.mpc.dt
        distances = self.progress[:, None] + self.reference_speed[:, None] * (
            offsets[None] + time_offset)
        position, tangent, pitch, yaw, _ = self.paths.sample(self.path_ids, distances)
        velocity = self.reference_speed[:, None, None] * tangent
        reference = torch.cat((position, pitch[..., None], yaw[..., None], velocity), -1)
        reference[..., :3] -= offsets[None, :, None] * self.current_estimate[:, None]
        reference[..., 5:8] -= self.current_estimate[:, None]
        return reference

    @torch.no_grad()
    def reset(self, path_ids, mask=None):
        ids = torch.as_tensor(path_ids, device=self.device, dtype=torch.long)
        if mask is None:
            mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        self.reference_speed[mask] = self.paths.speed[ids[mask], 0]
        self.policy_residual[mask] = 0
        return super().reset(ids, mask)

    @torch.no_grad()
    def observe(self):
        return torch.cat((super().observe(),
                          (self.reference_speed / 18.)[:, None].float()), -1)

    @torch.no_grad()
    def step(self, action):
        if action.shape != (self.batch_size, 1):
            raise ValueError("policy action must have shape [batch_size, 1]")
        active = ~self.finished
        if self.base_reference == "task":
            target = self.paths.speed[self.path_ids, 0]
            desired = (target + 6. * action[:, 0].to(self.dtype)).clamp(9., 18.)
            # Filter the complete speed when the task target is the base.
            self.reference_speed = torch.where(
                active, .7 * self.reference_speed + .3 * desired,
                self.reference_speed)
        else:
            desired_residual = self.policy_residual_scale * action[:, 0].to(self.dtype)
            self.policy_residual = torch.where(
                active, .7 * self.policy_residual + .3 * desired_residual,
                self.policy_residual)
        observation, reward, done, info = super().step(
            torch.zeros(self.batch_size, 3, device=self.device, dtype=self.dtype))
        command = info["command"]
        upper = self.model.upper * self.model.control_scale
        lower = self.model.lower * self.model.control_scale
        saturated = ((command >= upper - 1e-6) |
                     (command <= lower + 1e-6)).any(-1)
        reward = reward - self.saturation_penalty * (active & saturated).to(reward.dtype)
        reward = reward - self.residual_penalty * self.policy_residual.square().to(
            reward.dtype) * active.to(reward.dtype)
        info["reference_speed_mps"] = self.reference_speed.clone()
        info["saturated"] = saturated
        info["policy_residual_mps"] = self.policy_residual.clone()
        return observation, reward, done, info


class SpeedActorCritic(nn.Module):
    def __init__(self, observation_dim):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(observation_dim, 128), nn.ReLU(),
                                   nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
        self.critic = nn.Sequential(nn.Linear(observation_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():
            self.actor[-1].bias[1] = -1.

    def distribution(self, observation):
        mean, log_std = self.actor(observation).chunk(2, dim=-1)
        return Normal(mean, log_std.clamp(-5., 1.).exp())

    def value(self, observation):
        return self.critic(observation).squeeze(-1)

    def sample(self, observation):
        distribution = self.distribution(observation)
        raw = distribution.sample()
        action = raw.tanh()
        logp = (distribution.log_prob(raw) -
                torch.log(1 - action.square() + 1e-6)).sum(-1)
        return action, logp, self.value(observation)

    def log_prob(self, observation, action):
        clipped = action.clamp(-1 + 1e-6, 1 - 1e-6)
        raw = torch.atanh(clipped)
        return (self.distribution(observation).log_prob(raw) -
                torch.log(1 - clipped.square() + 1e-6)).sum(-1)

    def deterministic(self, observation):
        return self.actor(observation)[:, :1].tanh()


def speed_bank(files, device, speeds=(12., 18.)):
    bank = PathBank([file for speed in speeds for file in files], device)
    count = len(files)
    for index, speed in enumerate(speeds):
        rows = slice(index * count, (index + 1) * count)
        bank.speed[rows] = speed
        bank.first_state[rows, 5] = speed
        bank.durations[rows] = bank.lengths[rows] / speed
    return bank


def make_env(bank, batch_size, backend, seed, randomize_plant=True,
             current_max=1., admm_iterations=200, saturation_penalty=0.,
             base_reference="task", residual_scale=2., residual_penalty=0.,
             current_profile="steady"):
    return ReferencePolicyEnv(
        bank, batch_size=batch_size, device="cpu", horizon=25,
        saturation_penalty=saturation_penalty,
        base_reference=base_reference, residual_scale=residual_scale,
        residual_penalty=residual_penalty,
        backend=backend, seed=seed, randomize_plant=randomize_plant,
        admm_iterations=admm_iterations, admm_rho=10.,
        current_max_mps=current_max, current_profile=current_profile,
        attitude_pitch_weight=100.,
        attitude_yaw_weight=30., fsp_enabled=True,
        preview_mode="multiscale", yaw_reward="course")


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    files = sorted(Path(args.train_path_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError("DDAT training path bank is empty")
    bank = speed_bank(files, "cpu", speeds=tuple(args.speeds))
    env = make_env(bank, args.envs, "admm", args.seed,
                   admm_iterations=args.admm_iterations,
                   saturation_penalty=args.saturation_penalty,
                   base_reference=args.base_reference,
                   residual_scale=args.residual_scale,
                   residual_penalty=args.residual_penalty)
    agent = SpeedActorCritic(env.observation_size)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    path_ids = torch.as_tensor(rng.integers(len(bank.files), size=args.envs))
    observation = env.reset(path_ids)
    episode_return = torch.zeros(args.envs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({
        **vars(args), "output": str(args.output),
        "train_path_dir": str(args.train_path_dir),
        "train_paths": [str(p) for p in files],
        "task_speeds_mps": args.speeds, "reference_range_mps": [9., 18.],
        "backend": "admm", "fsp_enabled": True,
    }, indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()
    completed = 0
    recent_success = []
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "update", "env_steps", "wall_s", "mean_step_reward", "episodes",
            "recent_success_rate", "mean_reference_speed_mps", "mpc_fallback_fraction"))
        writer.writeheader()
        env_steps = 0
        update = 0
        while env_steps < args.total_steps:
            update += 1
            count = min(args.rollout, int(np.ceil((args.total_steps-env_steps)/args.envs)))
            trajectories = [[] for _ in range(7)]
            reward_values = []
            fallback_values = []
            reference_values = []
            for _ in range(count):
                current = observation
                with torch.no_grad():
                    action, logp, value = agent.sample(current)
                    next_obs, reward, done, info = env.step(action)
                    next_value = agent.value(next_obs)
                episode_return += reward
                reward_values.append(reward.mean().item())
                fallback_values.append((~info["mpc_ok"]).float().mean().item())
                reference_values.append(info["reference_speed_mps"].mean().item())
                env_steps += args.envs
                if done.any():
                    finished = done.nonzero().flatten()
                    completed += len(finished)
                    recent_success.extend(info["success"][finished].cpu().tolist())
                    recent_success = recent_success[-100:]
                    episode_return[finished] = 0
                    path_ids[finished] = torch.as_tensor(
                        rng.integers(len(bank.files), size=len(finished)))
                    observation = env.reset(path_ids, done)
                else:
                    observation = next_obs
                for target, item in zip(trajectories, (
                        current, action, logp, value, reward, next_value, done)):
                    target.append(item.detach())
            obs, action, logp, value, reward, next_value, done = [
                torch.stack(x) for x in trajectories]
            adv, ret = advantages(reward, value, next_value, done, done)
            flatten = lambda item: item.reshape(-1, *item.shape[2:])
            ppo_update(agent, optimizer, (
                flatten(obs), flatten(action), flatten(logp),
                flatten(ret), flatten(adv)), epochs=args.epochs,
                minibatch=args.minibatch)
            row = {
                "update": update, "env_steps": env_steps,
                "wall_s": round(time.perf_counter() - start, 1),
                "mean_step_reward": float(np.mean(reward_values)),
                "episodes": completed,
                "recent_success_rate": (float(np.mean(recent_success))
                                        if recent_success else None),
                "mean_reference_speed_mps": float(np.mean(reference_values)),
                "mpc_fallback_fraction": float(np.mean(fallback_values)),
            }
            writer.writerow(row)
            stream.flush()
            if update % args.log_every == 0 or env_steps >= args.total_steps:
                print(json.dumps(row), flush=True)
    torch.save({"agent": agent.state_dict(), "observation_dim": env.observation_size,
                "config": json.loads((args.output / "config.json").read_text())},
               args.output / "final_actor.pt")
    print(f"saved {args.output / 'final_actor.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=60_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--admm-iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1102)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--saturation-penalty", type=float, default=0.)
    parser.add_argument("--base-reference", choices=("task", "curvature"),
                        default="task")
    parser.add_argument("--residual-scale", type=float, default=2.)
    parser.add_argument("--residual-penalty", type=float, default=0.)
    parser.add_argument("--train-path-dir", type=Path,
                        default=ROOT / "DDAT_trajectories/train")
    parser.add_argument("--speeds", nargs="+", type=float, default=[12., 18.])
    args = parser.parse_args()
    if min(args.total_steps, args.envs, args.rollout, args.epochs,
           args.minibatch, args.admm_iterations, args.threads, args.log_every) <= 0:
        parser.error("count arguments must be positive")
    if not args.speeds or any(speed <= 0 or speed > 18 for speed in args.speeds):
        parser.error("task speeds must lie in (0, 18]")
    if args.saturation_penalty < 0:
        parser.error("saturation penalty must be nonnegative")
    if args.residual_scale <= 0 or args.residual_penalty < 0:
        parser.error("residual scale must be positive and penalty nonnegative")
    train(args)


if __name__ == "__main__":
    main()
