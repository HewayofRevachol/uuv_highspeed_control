"""Collect the paired 24-episode T_fit set for E1 and export estimator traces."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing as mp
import subprocess
from pathlib import Path

import torch

from experiments.h15_cat_mpc.run_episode import ROOT, run_episode, sha256


def execute(job):
    torch.set_num_threads(1)
    return run_episode(**job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        out.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("E1 outputs must be inside the project root") from exc
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    manifest_path = ROOT/cfg.get("t_fit_manifest", cfg["train_manifest"])
    manifest = json.loads(manifest_path.read_text())
    units = [(case, int(seed)) for case in manifest["cases"]
             for seed in manifest["seeds"]]
    if len(units) != 12 or len({(c["id"], s) for c, s in units}) != 12:
        raise RuntimeError("T_fit must contain the 12 frozen geometry x seed units")
    if manifest.get("split") != "T_fit" or manifest.get("seeds") != [51001, 51002, 51003]:
        raise RuntimeError("E1 collection requires the predeclared T_fit split and seeds")
    jobs = []
    for case, seed in units:
        for mode in ("on", "off"):
            stem = f"{case['id']}_s{seed}_{mode}"
            jobs.append({"path": ROOT/case["path"], "seed": seed,
                "out": out/"episodes"/stem, "method": "A0",
                "target_domain": True, "compensation_mode": mode,
                "predictor_mode": "zero"})
    run_record = {"schema": 1, "status": "running", "scope":
        "train only; no dev or test observation files", "episodes_expected": 24,
        "compensation_modes": ["on", "off"], "workers": args.workers,
        "train_manifest": str(manifest_path.relative_to(ROOT)),
        "train_manifest_sha256": sha256(manifest_path),
        "config_sha256": sha256(args.config), "run_id": out.name}
    (out/"run.json").write_text(json.dumps(run_record, indent=2)+"\n")
    protocol = {"schema": 1, "stage": "E1 T_fit", "control_dt_s": .1,
        "plant_substep_s": .02, "target_domain": True,
        "methods": ["A0"], "compensation_modes": ["on", "off"],
        "predictor_mode": "zero", "paired_units": 12,
        "normalization_split": "train", "holdout_used": False,
        "train_manifest_sha256": run_record["train_manifest_sha256"]}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    source_files = [Path(__file__), ROOT/"experiments/h15_cat_mpc/run_episode.py",
        ROOT/"experiments/h15_cat_mpc/residual_predictor.py",
        ROOT/"experiments/h15_mainline/response_estimator.py",
        ROOT/"simulator/batch_path_env.py", ROOT/"simulator/ddat_env.py"]
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
        diff_hash = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        git_commit, diff_hash = None, None
    source_manifest = {"git_commit": git_commit, "working_diff_sha256": diff_hash,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        "train_manifest_sha256": run_record["train_manifest_sha256"],
        "config_sha256": run_record["config_sha256"]}
    (out/"source_manifest.json").write_text(json.dumps(source_manifest, indent=2)+"\n")
    ctx = mp.get_context("spawn")
    episodes = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=ctx) as pool:
        for i, result in enumerate(pool.map(execute, jobs), 1):
            job = jobs[i-1]
            episode_path = job["out"]/"episode.json"
            trace_path = out/"traces"/job["compensation_mode"]/f"{job['out'].name}.csv"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            rows = result["completed_rows"]
            if len(rows) < 100:
                raise RuntimeError(f"insufficient usable periods in {episode_path}: {len(rows)}")
            fields = ["case_id", "seed", "step"] + sorted(
                {key for row in rows for key in row})
            with trace_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for step, row in enumerate(rows):
                    writer.writerow({"case_id": job["out"].name.split("_s")[0],
                        "seed": job["seed"], "step": step, **row})
            episodes.append({"case_id": job["out"].name.split("_s")[0],
                "seed": job["seed"], "compensation_mode": job["compensation_mode"],
                "episode": str(episode_path.relative_to(ROOT)),
                "episode_sha256": sha256(episode_path), "trace": str(trace_path.relative_to(ROOT)),
                "trace_sha256": sha256(trace_path), "periods": len(rows),
                "success": result["success"]})
            print(f"{i}/24 {job['out'].name}: periods={len(rows)} success={result['success']}",
                  flush=True)
    observed = {(x["case_id"], x["seed"], x["compensation_mode"]) for x in episodes}
    expected = {(case["id"], seed, mode) for case, seed in units
                for mode in ("on", "off")}
    if observed != expected or len(episodes) != 24:
        raise RuntimeError("E1 collection is incomplete or contains duplicate episodes")
    (out/"summary.json").write_text(json.dumps(episodes, indent=2)+"\n")
    run_record["status"] = "completed"
    run_record["episodes_completed"] = len(episodes)
    run_record["summary_sha256"] = sha256(out/"summary.json")
    (out/"run.json").write_text(json.dumps(run_record, indent=2)+"\n")
    print(json.dumps(run_record, indent=2))


if __name__ == "__main__":
    main()
