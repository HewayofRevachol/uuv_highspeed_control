"""Paired fixed-gate or checkpoint evaluation with one episode per job."""
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import concurrent.futures
import json
import multiprocessing as mp
from pathlib import Path

import torch

from experiments.h15_cat_mpc.run_episode import run_episode


ROOT = Path(__file__).resolve().parents[2]


def execute(job):
    torch.set_num_threads(1)
    return run_episode(**job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--split", choices=("train", "dev", "holdout"), required=True)
    parser.add_argument("--methods", nargs="+", default=["A0", "A1", "A2", "A3", "A4"])
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--policy-checkpoints", nargs="+", type=Path)
    parser.add_argument("--adapter-normalization", type=Path)
    parser.add_argument("--adapter-calibration", type=Path)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    manifest_path = ROOT / cfg[{"train": "train_manifest", "dev": "dev_manifest",
                                "holdout": "holdout_manifest"}[args.split]]
    manifest = json.loads(manifest_path.read_text())
    cases = (manifest.get("splits", {}).get("holdout", []) if args.split == "holdout"
             else manifest.get("cases", []))
    seeds = (manifest.get("experiment_freeze", {}).get("seeds", [])
             if args.split == "holdout" else manifest.get("seeds", []))
    checkpoints = args.policy_checkpoints or ([args.policy_checkpoint]
                  if args.policy_checkpoint else [None])
    jobs = []
    for case in cases:
        path = case.get("path")
        geometry = case.get("id", case.get("family", Path(path).stem))
        for seed in seeds:
            for method in args.methods:
                for checkpoint in checkpoints:
                    label = (f"{method}_{checkpoint.stem.removeprefix('ckpt_')}"
                             if checkpoint is not None and args.policy_checkpoints else method)
                    name = f"{args.split}_{geometry}_s{seed}_{label}"
                    jobs.append({"path": ROOT/path, "seed": seed,
                        "out": args.out/name, "method": label,
                        "target_domain": True,
                        "adapter_artifacts": ({"normalization_path": args.adapter_normalization,
                                               "calibration_path": args.adapter_calibration}
                                              if args.adapter_normalization else None),
                        "predictor_checkpoint": args.predictor,
                        "policy_checkpoint": checkpoint})
    args.out.mkdir(parents=True, exist_ok=True)
    run = {"schema": 1, "split": args.split, "methods": args.methods,
           "workers": args.workers, "manifest": str(manifest_path),
           "jobs": len(jobs), "config": str(args.config),
           "policy_checkpoints": [str(x) for x in checkpoints if x is not None]}
    (args.out/"run.json").write_text(json.dumps(run, indent=2)+"\n")
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx) as pool:
        for i, result in enumerate(pool.map(execute, jobs), 1):
            print(f"{i}/{len(jobs)} {result['method']} s{result['seed']} "
                  f"success={result['success']} fallback={result['fallback_periods']}",
                  flush=True)


if __name__ == "__main__":
    main()
