"""Train GS-12 continuous beta PPO with batched CUDA predictor/plant and ADMM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch
from tqdm import tqdm

from .continuous_ppo import ContinuousBetaActorCritic, gae_batch, update
from .environment import BatchedBetaEnv


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).parents[1]/"gs12_rl_beta_pilot"/"output")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--update-episodes", type=int, default=16)
    parser.add_argument("--predictor", choices=("auto", "torch", "cusadi"), default="cusadi")
    parser.add_argument("--admm-iterations", type=int, default=100)
    parser.add_argument("--admm-rho", type=float, default=10.)
    parser.add_argument("--admm-tolerance", type=float, default=1e-3)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if (args.episodes < 1 or args.update_episodes < 1 or
            args.episodes % args.update_episodes):
        parser.error("episodes must be a positive multiple of update-episodes")
    if args.admm_iterations < 1 or args.admm_rho <= 0 or args.admm_tolerance <= 0:
        parser.error("invalid ADMM settings")
    if args.source.resolve() == args.out.resolve():
        parser.error("source and output directories must differ")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    tasks_file = args.source/"tasks_train.jsonl"
    cache_file = args.source/"reward_baseline_cache.jsonl"
    tasks = read_jsonl(tasks_file)
    if args.episodes > len(tasks):
        parser.error("episodes exceed frozen task count")
    cache = {row["task_id"]: row for row in read_jsonl(cache_file)}
    baselines = []
    for task in tasks[:args.episodes]:
        row = cache.get(task["task_id"])
        if (row is None or row.get("status") not in
                ("complete_valid", "complete_constrained") or
                row.get("J_track") is None or not np.isfinite(row["J_track"]) or
                row["J_track"] < 0):
            raise ValueError(f"missing valid F75 cache for {task['task_id']}")
        baselines.append(row["J_track"])
    predictor = args.predictor
    if predictor == "auto":
        root = Path(os.getenv("CUSADI_ROOT", ".third_party/cusadi"))
        predictor = "cusadi" if (root/"src"/"generateCUDACode.py").is_file() else "torch"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    policy = ContinuousBetaActorCritic().cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    history = []
    started = time.perf_counter()
    print(f"predictor={predictor}, ADMM iterations={args.admm_iterations}", flush=True)
    with tqdm(total=args.episodes, unit="episode", desc="GS-12 CUDA beta PPO") as progress:
        for offset in range(0, args.episodes, args.update_episodes):
            chosen = tasks[offset:offset+args.update_episodes]
            env = BatchedBetaEnv(chosen, baselines[offset:offset+args.update_episodes],
                                 predictor=predictor, admm_iterations=args.admm_iterations,
                                 admm_rho=args.admm_rho,
                                 admm_tolerance=args.admm_tolerance,
                                 cuda_graph=not args.no_cuda_graph)
            observations, raw, logp, values, rewards, done, valid = [], [], [], [], [], [], []
            with torch.no_grad():
                for _ in range(env.decisions):
                    active = env.alive.clone()
                    if not bool(active.any()):
                        break
                    obs = env.observe()
                    z, target, old_logp, value = policy.sample(obs)
                    reward, terminal, _ = env.step_target(target.to(torch.float64))
                    observations.append(obs)
                    raw.append(z)
                    logp.append(old_logp)
                    values.append(value)
                    rewards.append(reward)
                    done.append(terminal)
                    valid.append(active)
            batch = gae_batch(observations, raw, logp, values, rewards, done, valid)
            metrics = update(policy, optimizer, batch)
            torch.cuda.synchronize()
            rows = env.summary()
            for task, row in zip(chosen, rows):
                row["task_id"] = task["task_id"]
            with (args.out/"episodes.jsonl").open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row)+"\n")
            completed = offset+len(chosen)
            metrics.update({"episodes": completed, "transitions": len(batch[0]),
                            "complete": sum(row["J_track"] is not None for row in rows),
                            "mean_train_J": float(np.mean([row["J_track"] for row in rows
                                                if row["J_track"] is not None]))
                            if any(row["J_track"] is not None for row in rows) else None,
                            "wall_s": time.perf_counter()-started})
            history.append(metrics)
            torch.save({"agent": policy.state_dict(), "optimizer": optimizer.state_dict(),
                        "episodes": completed, "seed": args.seed, "device": "cuda",
                        "predictor": predictor}, args.out/f"checkpoint_{completed:04d}.pt")
            (args.out/"history.json").write_text(json.dumps(history, indent=2)+"\n")
            progress.update(len(chosen))
            progress.set_postfix(loss=f"{metrics['loss']:.3g}")
    torch.save({"agent": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "episodes": args.episodes, "seed": args.seed, "device": "cuda",
                "action": "beta_target=0.75*sigmoid(raw Gaussian)"},
               args.out/"final_actor.pt")
    (args.out/"run.json").write_text(json.dumps({
        "status": "completed", "episodes": args.episodes, "updates": len(history),
        "device": "cuda", "predictor": predictor,
        "tasks_sha256": sha256(tasks_file), "reward_cache_sha256": sha256(cache_file),
        "seed": args.seed, "update_episodes": args.update_episodes,
        "admm": {"iterations": args.admm_iterations, "rho": args.admm_rho,
                 "tolerance": args.admm_tolerance, "cuda_graph": not args.no_cuda_graph},
        "ppo": {"epochs": 4, "minibatch": 64, "lr": 3e-4, "clip": .2,
                "gamma": .98, "gae_lambda": .95},
        "action": "continuous beta target = 0.75*sigmoid(raw Gaussian)",
        "model": "final_actor.pt"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
