"""Paired fixed-gate or checkpoint evaluation with one episode per job."""
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import torch

from experiments.h15_cat_mpc.run_episode import run_episode
from experiments.h15_cat_mpc.method_contract import FIXED_GATES, checkpoint_method
from experiments.h15_cat_mpc.contracts import (load_matching_predictor,
    validate_manifest, project_path)
from experiments.h15_cat_mpc.residual_predictor import RESIDUAL_LABEL_VERSION


ROOT = Path(__file__).resolve().parents[2]


def execute(job):
    torch.set_num_threads(1)
    return run_episode(**job)


def _resolve_manifest(config, split, override):
    if override is not None:
        path = Path(override)
        return path if path.is_absolute() else ROOT/path
    key = {"train": "t_fit_manifest", "dev": "dev_manifest",
           "holdout": "holdout_manifest"}[split]
    path = Path(config[key])
    return path if path.is_absolute() else ROOT/path


def _validate_methods(methods, checkpoints):
    allowed = set(FIXED_GATES) | {"PPO"}
    unknown = sorted(set(methods)-allowed)
    if unknown:
        raise ValueError(f"unknown method(s) {unknown}; expected A0-A4 or PPO")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not contain duplicates")
    if checkpoints and any(method in FIXED_GATES for method in methods):
        raise ValueError("fixed methods A0-A4 cannot be evaluated with policy checkpoints")
    if "PPO" in methods and not checkpoints:
        raise ValueError("PPO evaluation requires --policy-checkpoint or --policy-checkpoints")
    resolved = [Path(path).resolve() for path in checkpoints]
    if len(set(resolved)) != len(resolved):
        raise ValueError("policy checkpoints must be unique")
    labels = [checkpoint_method(path) for path in resolved]
    if len(set(labels)) != len(labels):
        raise ValueError("policy checkpoint stems must be unique")
    if any(not Path(path).is_file() for path in checkpoints):
        missing = [str(path) for path in checkpoints if not Path(path).is_file()]
        raise ValueError(f"policy checkpoint does not exist: {missing}")


def _load_manifest(manifest_path, split):
    """Validate regular manifests; adapt the legacy holdout shape safely."""
    if split != "holdout":
        expected = "T_fit" if split == "train" else "dev"
        return validate_manifest(manifest_path, ROOT, expected_split=expected)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("splits", {}).get("holdout")
    seeds = manifest.get("experiment_freeze", {}).get("seeds")
    if not isinstance(cases, list) or not cases:
        raise ValueError("holdout manifest has no cases")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("holdout manifest seeds must be unique and non-empty")
    seen = set()
    for case in cases:
        relative = case.get("path")
        if not relative:
            raise ValueError("holdout case lacks path")
        resolved = project_path(ROOT, relative)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if "path_sha256" in case:
            actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if case["path_sha256"] != actual:
                raise ValueError(f"manifest hash mismatch for {relative}")
        identifier = case.get("id", case.get("family", Path(relative).stem))
        if identifier in seen:
            raise ValueError(f"duplicate holdout case {identifier}")
        seen.add(identifier)
        case.setdefault("id", identifier)
    if any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("holdout seeds must be integers")
    return manifest, manifest_path.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--split", choices=("train", "dev", "holdout"), required=True)
    parser.add_argument("--methods", nargs="+", default=["A0", "A1", "A2", "A3", "A4"])
    parser.add_argument("--manifest", type=Path,
                        help="optional manifest override; otherwise train uses t_fit_manifest")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--policy-checkpoints", nargs="+", type=Path)
    parser.add_argument("--adapter-normalization", type=Path)
    parser.add_argument("--adapter-calibration", type=Path)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--predictor-mode", choices=("zero", "persistence", "ar_or_persistence"),
                        default="ar_or_persistence")
    parser.add_argument("--compensation-mode", choices=("on", "off"), default="on")
    parser.add_argument("--action-seed", type=int)
    policy_mode = parser.add_mutually_exclusive_group()
    policy_mode.add_argument("--deterministic-policy", dest="deterministic_policy",
                             action="store_true", help="use learned policy mean (default)")
    policy_mode.add_argument("--stochastic-policy", dest="deterministic_policy",
                             action="store_false", help="sample the learned policy")
    parser.set_defaults(deterministic_policy=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    checkpoints = args.policy_checkpoints or ([args.policy_checkpoint]
                  if args.policy_checkpoint else [])
    if args.policy_checkpoints and args.policy_checkpoint:
        parser.error("use only one of --policy-checkpoint and --policy-checkpoints")
    _validate_methods(args.methods, checkpoints)
    if args.predictor is not None:
        if args.adapter_normalization is None:
            raise ValueError("predictor checkpoint requires matching adapter normalization")
        load_matching_predictor(args.adapter_normalization, args.predictor,
                                args.compensation_mode, RESIDUAL_LABEL_VERSION)
    manifest_path = _resolve_manifest(cfg, args.split, args.manifest)
    if not manifest_path.is_file():
        raise ValueError(f"manifest does not exist: {manifest_path}")
    manifest, manifest_path = _load_manifest(manifest_path, args.split)
    cases = (manifest.get("splits", {}).get("holdout", []) if args.split == "holdout"
             else manifest.get("cases", []))
    seeds = (manifest.get("experiment_freeze", {}).get("seeds", [])
             if args.split == "holdout" else manifest.get("seeds", []))
    case_path_sha256 = {case["path"]: hashlib.sha256(
        project_path(ROOT, case["path"]).read_bytes()).hexdigest() for case in cases}
    checkpoint_jobs = checkpoints or [None]
    jobs = []
    for case in cases:
        path = case.get("path")
        geometry = case.get("id", case.get("family", Path(path).stem))
        for seed in seeds:
            for method in args.methods:
                for checkpoint in checkpoint_jobs:
                    label = (checkpoint_method(checkpoint)
                             if checkpoint is not None and args.policy_checkpoints
                             else method)
                    name = f"{args.split}_{geometry}_s{seed}_{label}"
                    jobs.append({"path": ROOT/path, "seed": seed,
                        "out": args.out/name, "method": label,
                        "target_domain": True,
                        "adapter_artifacts": ({"normalization_path": args.adapter_normalization,
                                               "calibration_path": args.adapter_calibration}
                                              if args.adapter_normalization else None),
                        "predictor_checkpoint": args.predictor,
                        "policy_checkpoint": checkpoint,
                        "compensation_mode": args.compensation_mode,
                        "predictor_mode": args.predictor_mode,
                        "action_seed": args.action_seed,
                        "policy_deterministic": bool(args.deterministic_policy) if checkpoint else False})
    args.out.mkdir(parents=True, exist_ok=True)
    run = {"schema": 3, "split": args.split, "methods": args.methods,
           "workers": args.workers, "manifest": str(manifest_path),
           "jobs": len(jobs), "config": str(args.config),
           "compensation_mode": args.compensation_mode,
           "predictor_mode": args.predictor_mode,
           "action_seed": args.action_seed,
           "deterministic_policy": (bool(args.deterministic_policy)
                                    if "PPO" in args.methods else False),
           "policy_checkpoints": [str(x) for x in checkpoints],
           "policy_checkpoint_sha256": {str(x): hashlib.sha256(Path(x).read_bytes()).hexdigest()
                                         for x in checkpoints},
           "method_contract": {"fixed_methods": sorted(FIXED_GATES),
                               "policy_methods": ["PPO"],
                               "policy_mode": ("fixed_gate" if "PPO" not in args.methods
                                                else ("mean" if args.deterministic_policy
                                                      else "gaussian_sample"))},
           "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
           "case_path_sha256": case_path_sha256}
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
