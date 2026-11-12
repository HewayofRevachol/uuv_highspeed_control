"""Train a continuous beta policy on the existing GS-12 CPU closed loop.

Independent Plant/MPC episodes run in CPU processes. A CUDA device, when
available, handles the dense PPO update; the current GS-12 MPC cannot be
batched on GPU without changing its solver and experimental dynamics.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch
from tqdm import tqdm

from training.ppo import advantages as gae_advantages
from .continuous_ppo import ContinuousBetaActorCritic, update
from .episode_beta import BetaEpisode
from .reward import external_j_track


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rollout(job):
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    policy = ContinuousBetaActorCritic()
    policy.load_state_dict(job["policy"])
    policy.eval()
    env = BetaEpisode(job["task"], job["j_f75"])
    transitions = []
    while not env.done:
        observation = env.observe()
        with torch.no_grad():
            obs = torch.as_tensor(observation).unsqueeze(0)
            raw, beta, logp, value = policy.sample(obs)
        _, reward, done, info = env.step_target(float(beta.item()))
        transitions.append((observation.tolist(), float(raw.item()),
                            float(logp.item()), float(value.item()),
                            float(reward), bool(done), info["beta_target"]))
    result = env.result()
    complete = (result["meta"].get("status") in
                ("complete_valid", "complete_constrained") and env.session.k == 640)
    score = external_j_track(result["arrays"]) if complete else None
    return {"task_id": job["task"]["task_id"], "transitions": transitions,
            "J_track": score, "status": result["meta"].get("status")}


def make_batch(results, device):
    observations, raw, old_logp, all_returns, all_adv = [], [], [], [], []
    for result in results:
        transitions = result["transitions"]
        rewards = torch.tensor([row[4] for row in transitions])
        values = torch.tensor([row[3] for row in transitions])
        next_values = torch.cat((values[1:], torch.zeros(1)))
        done = torch.tensor([row[5] for row in transitions], dtype=torch.bool)
        adv, returns = gae_advantages(rewards, values, next_values, done, done,
                                      gamma=.98, lam=.95)
        observations.extend(row[0] for row in transitions)
        raw.extend(row[1] for row in transitions)
        old_logp.extend(row[2] for row in transitions)
        all_adv.extend(adv.tolist())
        all_returns.extend(returns.tolist())
    return (torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device),
            torch.tensor(raw, dtype=torch.float32, device=device),
            torch.tensor(old_logp, dtype=torch.float32, device=device),
            torch.tensor(all_returns, dtype=torch.float32, device=device),
            torch.tensor(all_adv, dtype=torch.float32, device=device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).with_name("output"),
                        help="directory with frozen training tasks and F75 reward cache")
    parser.add_argument("--out", type=Path, required=True,
                        help="separate output directory for this continuous experiment")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--update-episodes", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.episodes < 1 or args.update_episodes < 1 or args.workers < 1:
        parser.error("episode and worker counts must be positive")
    if args.episodes % args.update_episodes:
        parser.error("--episodes must be a multiple of --update-episodes")
    if args.source.resolve() == args.out.resolve():
        parser.error("--out must differ from --source")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
              else "cpu" if args.device == "auto" else args.device)
    tasks = read_jsonl(args.source / "tasks_train.jsonl")
    if args.episodes > len(tasks):
        parser.error("requested episodes exceed the frozen task count")
    cache = {record["task_id"]: record for record in
             read_jsonl(args.source / "reward_baseline_cache.jsonl")}
    for task in tasks[:args.episodes]:
        record = cache.get(task["task_id"])
        if (record is None or record.get("status") not in
                ("complete_valid", "complete_constrained") or
                record.get("J_track") is None or
                not np.isfinite(record["J_track"]) or record["J_track"] < 0):
            raise ValueError(f"missing valid F75 cache for {task['task_id']}")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must be empty to avoid mixing experiment versions")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    agent = ContinuousBetaActorCritic().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    history = []
    start = time.perf_counter()
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context) as pool:
        with tqdm(total=args.episodes, unit="episode", desc="continuous PPO") as progress:
            for batch_start in range(0, args.episodes, args.update_episodes):
                # Every task in a batch sees the same policy version.
                frozen = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
                jobs = [
                    {"task": task, "j_f75": cache[task["task_id"]]["J_track"],
                     "policy": frozen, "seed": args.seed + batch_start + i}
                    for i, task in enumerate(tasks[batch_start:batch_start + args.update_episodes])
                ]
                futures = {pool.submit(rollout, job): i for i, job in enumerate(jobs)}
                results = [None] * len(jobs)
                for future in concurrent.futures.as_completed(futures):
                    results[futures[future]] = future.result()
                    progress.update(1)
                batch = make_batch(results, device)
                metrics = update(agent, optimizer, batch)
                completed = batch_start + len(results)
                metrics.update({"episodes": completed, "transitions": len(batch[0]),
                                "complete": sum(row["J_track"] is not None for row in results),
                                "mean_train_J": float(np.mean(
                                    [row["J_track"] for row in results if row["J_track"] is not None]))
                                if any(row["J_track"] is not None for row in results) else None,
                                "wall_s": time.perf_counter() - start, "device": device})
                history.append(metrics)
                torch.save({"agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                            "episodes": completed, "seed": args.seed, "device": device},
                           args.out / f"checkpoint_{completed:04d}.pt")
                (args.out / "history.json").write_text(
                    json.dumps(history, indent=2) + "\n", encoding="utf-8")
                with (args.out / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    for result in results:
                        stream.write(json.dumps(result) + "\n")
                progress.set_postfix(loss=f"{metrics['loss']:.3g}",
                                     J=f"{metrics['mean_train_J']:.3g}"
                                     if metrics["mean_train_J"] is not None else "invalid")
    torch.save({"agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                "episodes": args.episodes, "seed": args.seed, "device": device,
                "action": "beta_target=0.75*sigmoid(raw Gaussian)"},
               args.out / "final_actor.pt")
    (args.out / "run.json").write_text(json.dumps(
        {"status": "completed", "episodes": args.episodes,
         "updates": len(history), "workers": args.workers, "device": device,
         "source": str(args.source), "seed": args.seed,
         "tasks_sha256": sha256(args.source / "tasks_train.jsonl"),
         "reward_cache_sha256": sha256(args.source / "reward_baseline_cache.jsonl"),
         "update_episodes": args.update_episodes,
         "ppo": {"epochs": 4, "minibatch": 64, "lr": 3e-4,
                 "clip": .2, "gamma": .98, "gae_lambda": .95},
         "action": "continuous beta target = 0.75*sigmoid(raw Gaussian)",
         "beta_slew_per_cycle": .02,
         "model": "final_actor.pt"}, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.out / 'final_actor.pt'}", flush=True)


if __name__ == "__main__":
    main()
