"""Paired target-domain evaluation of a fixed, deployable fin-response prior."""
import argparse
import contextlib
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from aav_mpc.diagnose import run


METHODS = {
    "endpoint20": {},
    "condensed_endpoint35": {},
    "robust_condensed_endpoint35": {
        "prediction_fin_gain_scale": .85,
        "prediction_fin_tau_scale": 2.,
    },
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cases):
    result = {"cases": len(cases), "methods": {}}
    for method in {case["method"] for case in cases}:
        selected = [case for case in cases if case["method"] == method]
        result["methods"][method] = {
            "episodes": len(selected),
            "strict_success": sum(case["strict_success"] for case in selected),
            "first_anomaly_types": {
                key: sum((case["first_anomaly"] or [0, "none"])[1] == key
                         for case in selected)
                for key in sorted({(case["first_anomaly"] or [0, "none"])[1]
                                   for case in selected})
            },
            "fallback_count": sum(case["fallback_count"] for case in selected),
            "p95_median_ms": float(np.median([
                case["control_period_p95_ms"] for case in selected])),
            "p99_max_ms": max(case["control_period_p99_ms"] for case in selected),
            "common_success_position_rmse_mean_m": float(np.mean([
                case["position_rmse_m"] for case in selected
                if case["strict_success"]])) if any(
                    case["strict_success"] for case in selected) else None,
        }
    baseline = {(case["family"], case["seed"]): case for case in cases
                if case["method"] == "condensed_endpoint35"}
    for method in ("endpoint20", "robust_condensed_endpoint35"):
        pairs = [(baseline[(case["family"], case["seed"])], case) for case in cases
                 if case["method"] == method and
                 (case["family"], case["seed"]) in baseline]
        if pairs:
            result.setdefault("paired_vs_condensed35", {})[method] = {
                "pairs": len(pairs),
                "baseline_success": sum(a["strict_success"] for a, _ in pairs),
                "method_success": sum(b["strict_success"] for _, b in pairs),
                "improved": sum(not a["strict_success"] and b["strict_success"]
                                for a, b in pairs),
                "worsened": sum(a["strict_success"] and not b["strict_success"]
                                for a, b in pairs),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any(len(set(values)) != len(values) for values in
           (args.families or [], args.seeds, args.methods)):
        parser.error("families, seeds and methods must be unique")
    manifest = json.loads(args.manifest.read_text())
    paths = manifest["splits"][args.split]
    if args.families:
        paths = [path for path in paths if path["family"] in args.families]
        if len(paths) != len(args.families):
            parser.error("some requested families are missing")
    sources = [Path("aav_mpc/diagnose.py"), Path("aav_mpc/h15_robust_prediction.py"),
               Path("aav_mpc/h15_target_domain.py"), Path("aav_mpc/controller.py"),
               Path("aav_mpc/condensed_mpc.py"), Path("simulator/rti_symbolic.py"),
               Path("simulator/batch_path_env.py"), Path("aav_mpc/configs/h15.json"),
               Path("aav_mpc/configs/h15_target.json"), args.manifest,
               *(Path(path["path"]) for path in paths)]
    config = {
        "manifest": str(args.manifest), "split": args.split,
        "families": [path["family"] for path in paths], "seeds": args.seeds,
        "methods": args.methods, "target_domain": {
            "randomize_plant": True, "process_noise": True,
            "current_max_mps": 1., "current_profile": "harmonic",
            "target_components": ["fin", "integrator", "actuator"],
            "fin_gain_scale": .9,
        },
        "source_sha256": {str(path): sha256(path) for path in sources},
        "python": platform.python_version(), "torch": torch.__version__,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    run_file = args.out / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()) != config:
        parser.error("output directory belongs to a different configuration")
    run_file.write_text(json.dumps(config, indent=2) + "\n")
    cases = []
    started = time.monotonic()
    with (args.out / "console.log").open("a") as log:
        for path in paths:
            for seed in args.seeds:
                for method in args.methods:
                    stem = f"{args.split}_{path['family']}_s{seed}_{method}"
                    trace = args.out / f"{stem}.csv"
                    summary_file = trace.with_suffix(".json")
                    if summary_file.exists() and trace.exists():
                        case = json.loads(summary_file.read_text())
                    else:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            case = run(
                                "H15", "condensed_endpoint35" if method.startswith("robust")
                                else method, trace,
                                path_override=path["path"], seed_override=seed,
                                randomize_plant=True, process_noise=True,
                                current_max=1., current_profile="harmonic",
                                target_domain=True,
                                target_components=("fin", "integrator", "actuator"),
                                **METHODS[method])
                    case.update(split=args.split, family=path["family"], method=method)
                    summary_file.write_text(json.dumps(case, indent=2) + "\n")
                    cases.append(case)
                    print(f"{len(cases):3d} {stem} strict={case['strict_success']} "
                          f"first={case['first_anomaly']} p95={case['control_period_p95_ms']:.1f} "
                          f"elapsed={time.monotonic()-started:.0f}s", flush=True)
                    (args.out / "summary.json").write_text(
                        json.dumps(summarize(cases), indent=2) + "\n")


if __name__ == "__main__":
    main()
