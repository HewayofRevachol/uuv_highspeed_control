"""Reproducible paired H15 baseline screen; all cases share path and seed."""
import argparse
import contextlib
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from aav_mpc.diagnose import run
from aav_mpc.h15_paths import write_corpus
from aav_mpc.h15_challenge_paths import write_corpus as write_challenge_corpus
from aav_mpc.h15_replication_paths import write_corpus as write_replication_corpus


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("outputs/experiments/mpc_h15_baseline_s101026")
METHODS = ("osqp25", "osqp40", "endpoint40", "condensed25",
           "condensed40", "condensed_endpoint40", "rti25", "rti40")
EXTRA_METHODS = ("endpoint20", "condensed_endpoint20", "condensed_endpoint35")
SEEDS = tuple(range(101026, 101031))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cases, methods):
    grouped = {}
    for case in cases:
        key = (case["split"], case["family"], case["seed"])
        grouped.setdefault(key, {})[case["method"]] = case
    results = {"count": len(cases), "methods": {}, "paired_vs_osqp25": {}}
    for method in methods:
        selected = [r for r in cases if r["method"] == method]
        if not selected:
            continue
        results["methods"][method] = {
            "episodes": len(selected),
            "strict_success": sum(r["strict_success"] for r in selected),
            "simulator_success": sum(r["simulator_success"] for r in selected),
            "first_anomaly_types": {k: sum((r["first_anomaly"] or [None, "none"])[1] == k
                                           for r in selected)
                                    for k in sorted(set((r["first_anomaly"] or [None, "none"])[1]
                                                        for r in selected))},
            "period_p95_ms_median": float(np.median([r["control_period_p95_ms"] for r in selected])),
            "period_p99_ms_median": float(np.median([r["control_period_p99_ms"] for r in selected])),
            "fallback_fraction_mean": float(np.mean([r["fallback_count"] / r["steps"] for r in selected])),
        }
        pairs = [(group["osqp25"], group[method]) for group in grouped.values()
                 if "osqp25" in group and method in group]
        if method != "osqp25" and pairs:
            common = [(a, b) for a, b in pairs if a["strict_success"] and b["strict_success"]]
            results["paired_vs_osqp25"][method] = {
                "pairs": len(pairs),
                "strict_success_delta": sum(int(b["strict_success"]) - int(a["strict_success"])
                                            for a, b in pairs),
                "discordant_improved": sum(not a["strict_success"] and b["strict_success"]
                                            for a, b in pairs),
                "discordant_worsened": sum(a["strict_success"] and not b["strict_success"]
                                            for a, b in pairs),
                "common_success_pairs": len(common),
                "common_success_position_rmse_delta_m": (
                    float(np.mean([b["position_rmse_m"] - a["position_rmse_m"]
                                   for a, b in common])) if common else None),
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", choices=("development", "validation", "test", "replication"),
                        default="development")
    parser.add_argument("--corpus", choices=("calibration", "challenge", "replication"),
                        default="calibration")
    parser.add_argument("--methods", nargs="+", choices=METHODS + EXTRA_METHODS,
                        default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--current-max", type=float, default=.4)
    parser.add_argument("--no-randomize-plant", action="store_true")
    parser.add_argument("--no-process-noise", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if (args.corpus == "replication") != (args.split == "replication"):
        parser.error("replication corpus requires the replication split")
    corpus_dir = Path("data/datasets/h15_mpc_replication" if args.corpus == "replication"
                      else "data/datasets/h15_mpc_challenge" if args.corpus == "challenge"
                      else "data/datasets/h15_mpc")
    generator = (write_replication_corpus if args.corpus == "replication" else
                 write_challenge_corpus if args.corpus == "challenge" else write_corpus)
    manifest = generator(corpus_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"split": args.split, "corpus": args.corpus,
              "methods": args.methods, "seeds": args.seeds,
              "current_max_mps": args.current_max,
              "randomize_plant": not args.no_randomize_plant,
              "process_noise": not args.no_process_noise,
              "task_config_sha256": sha256(Path("aav_mpc/configs/h15.json")),
              "path_manifest_sha256": sha256(corpus_dir / "manifest.json"),
              "path_sha256": {p["path"]: sha256(Path(p["path"]))
                              for p in manifest["splits"][args.split]},
              "source_sha256": {str(p): sha256(p) for p in
                                (Path("aav_mpc/diagnose.py"),
                                 Path("aav_mpc/h15_experiment.py"),
                                 Path("aav_mpc/h15_paths.py"),
                                 Path("aav_mpc/h15_challenge_paths.py"),
                                 Path("aav_mpc/h15_replication_paths.py"),
                                 Path("aav_mpc/controller.py"),
                                 Path("aav_mpc/condensed_mpc.py"),
                                 Path("simulator/batch_mpc.py"),
                                 Path("simulator/rti_mpc.py"),
                                 Path("simulator/model.py"))},
              "python": platform.python_version(), "torch": torch.__version__,
              "git_head": subprocess.run(["git", "rev-parse", "HEAD"],
                                         capture_output=True, text=True, check=True).stdout.strip()}
    run_file = args.out / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()) != config:
        parser.error(f"{run_file} contains a different experiment configuration")
    run_file.write_text(json.dumps(config, indent=2) + "\n")
    cases = []
    started = time.monotonic()
    with (args.out / "console.log").open("a") as log:
        for path in manifest["splits"][args.split]:
            family = path["family"]
            for seed in args.seeds:
                for method in args.methods:
                    stem = f"{args.split}_{family}_s{seed}_{method}"
                    trace = args.out / f"{stem}.csv"
                    summary_file = trace.with_suffix(".json")
                    if summary_file.exists() and trace.exists():
                        case = json.loads(summary_file.read_text())
                    else:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            case = run("H15", method, trace,
                                       randomize_plant=not args.no_randomize_plant,
                                       current_max=args.current_max,
                                       process_noise=not args.no_process_noise,
                                       path_override=path["path"], seed_override=seed)
                    case.update(split=args.split, family=family)
                    cases.append(case)
                    print(f"{len(cases):3d} {stem} strict={case['strict_success']} "
                          f"first={case['first_anomaly']} p95={case['control_period_p95_ms']:.1f} "
                          f"elapsed={time.monotonic()-started:.0f}s", flush=True)
                    (args.out / "summary.json").write_text(
                        json.dumps(summarize(cases, args.methods), indent=2) + "\n")
    with (args.out / "episodes.csv").open("w", newline="") as stream:
        fields = ["split", "family", "seed", "method", "strict_success", "simulator_success",
                  "steps", "progress_fraction", "fallback_count", "saturation_fraction",
                  "position_rmse_m", "max_abs_lateral_m", "max_abs_depth_m",
                  "max_abs_speed_error_mps", "max_abs_pitch_deg",
                  "control_period_p50_ms", "control_period_p95_ms", "control_period_p99_ms"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: case.get(k) for k in fields} for case in cases)


if __name__ == "__main__":
    main()
