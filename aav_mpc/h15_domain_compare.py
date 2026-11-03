"""Paired H15 source vs alternative integrator/actuator target experiment."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np

from aav_mpc.diagnose import run


DEFAULT_METHODS = ("endpoint20", "condensed_endpoint40")
AVAILABLE_METHODS = ("endpoint20", "condensed_endpoint35", "condensed_endpoint40")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cases, methods):
    groups = {}
    for c in cases:
        key = (c["path"], c["seed"], c["method"])
        groups.setdefault(key, {})[c["domain"]] = c
    result = {"cases": len(cases), "methods": {}}
    for method in methods:
        paired = [(g["source"], g["target"]) for (path, seed, name), g in groups.items()
                  if name == method and "source" in g and "target" in g]
        if not paired:
            continue
        common = [(a, b) for a, b in paired if a["strict_success"] and b["strict_success"]]
        result["methods"][method] = {
            "pairs": len(paired),
            "source_success": sum(a["strict_success"] for a, b in paired),
            "target_success": sum(b["strict_success"] for a, b in paired),
            "improved": sum(not a["strict_success"] and b["strict_success"] for a, b in paired),
            "worsened": sum(a["strict_success"] and not b["strict_success"] for a, b in paired),
            "common_success_pairs": len(common),
            "common_success_rmse_delta_m": (float(np.mean([
                b["position_rmse_m"]-a["position_rmse_m"] for a, b in common]))
                if common else None),
            "target_period_p95_median_ms": float(np.median([
                b["control_period_p95_ms"] for a, b in paired])),
            "target_period_p99_max_ms": max(b["control_period_p99_ms"] for a, b in paired),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("challenge", "replication"), default="challenge")
    parser.add_argument("--split", choices=("development", "validation", "test", "replication"), required=True)
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS,
                        default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--source-dir", type=Path,
                        help="reuse completed source-domain episode JSON from a paired baseline run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.corpus == "replication") != (args.split == "replication"):
        parser.error("replication corpus requires the replication split")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("seeds and methods must be unique")
    root = Path("data/datasets/h15_mpc_replication" if args.corpus == "replication"
                else "data/datasets/h15_mpc_challenge")
    manifest = json.loads((root / "manifest.json").read_text())
    paths = manifest["splits"][args.split]
    sources = [Path("aav_mpc/diagnose.py"), Path("aav_mpc/h15_domain_compare.py"),
               Path("aav_mpc/h15_target_domain.py"), Path("aav_mpc/configs/h15.json"),
               Path("aav_mpc/configs/h15_target.json"), Path("simulator/batch_path_env.py"),
               root / "manifest.json", *(Path(p["path"]) for p in paths)]
    source_config = None
    if args.source_dir is not None:
        source_run = args.source_dir / "run.json"
        source_config = json.loads(source_run.read_text())
        if (source_config.get("split") != args.split or
                source_config.get("corpus") != args.corpus or
                set(args.methods) - set(source_config.get("methods", [])) or
                set(args.seeds) - set(source_config.get("seeds", [])) or
                source_config.get("current_max_mps") != .4 or
                not source_config.get("randomize_plant") or
                not source_config.get("process_noise") or
                source_config.get("path_manifest_sha256") != digest(root / "manifest.json")):
            parser.error("source directory is not a matching paired source-domain run")
        for file in (Path("aav_mpc/diagnose.py"), Path("aav_mpc/controller.py"),
                     Path("simulator/batch_mpc.py"), Path("simulator/model.py")):
            if source_config.get("source_sha256", {}).get(str(file)) != digest(file):
                parser.error(f"source result used a different {file}")
    source_episodes = (
        {str(file): digest(file) for path in paths for seed in args.seeds
         for method in args.methods
         for file in [args.source_dir /
                      f"{args.split}_{path['family']}_s{seed}_{method}.json"]}
        if args.source_dir else None)
    config = {"corpus": args.corpus, "split": args.split, "seeds": args.seeds,
              "methods": args.methods,
              "source_dir": str(args.source_dir) if args.source_dir else None,
              "source_run_sha256": digest(args.source_dir / "run.json") if args.source_dir else None,
              "source_episode_sha256": source_episodes,
              "source_sha256": {str(p): digest(p) for p in sources}}
    args.out.mkdir(parents=True, exist_ok=True)
    run_file = args.out / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()) != config:
        parser.error("output directory contains a different configuration or source version")
    run_file.write_text(json.dumps(config, indent=2) + "\n")
    cases = []
    with (args.out / "console.log").open("a") as log:
        for path in paths:
            for seed in args.seeds:
                for method in args.methods:
                    for domain in ("target",) if args.source_dir else ("source", "target"):
                        file = args.out / f"{args.split}_{path['family']}_s{seed}_{method}_{domain}.csv"
                        if file.exists() and file.with_suffix(".json").exists():
                            case = json.loads(file.with_suffix(".json").read_text())
                        else:
                            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                                case = run("H15", method, file, path_override=path["path"],
                                           seed_override=seed, randomize_plant=True,
                                           current_max=1. if domain == "target" else .4,
                                           process_noise=True,
                                           current_profile="harmonic" if domain == "target" else "steady",
                                           target_domain=(domain == "target"))
                        case["domain"] = domain
                        cases.append(case)
                        print(path["family"],seed,method,domain,case["strict_success"],
                              case["first_anomaly"],flush=True)
                        if args.source_dir:
                            source_file = (args.source_dir /
                                           f"{args.split}_{path['family']}_s{seed}_{method}.json")
                            source_case = json.loads(source_file.read_text())
                            if (source_case["path"] != path["path"] or
                                    source_case["seed"] != seed or
                                    source_case["method"] != method or
                                    source_case.get("target_domain") or
                                    source_case.get("current_max_mps") != .4):
                                raise ValueError(f"source episode does not match: {source_file}")
                            source_case["domain"] = "source"
                            cases.append(source_case)
                        (args.out / "summary.json").write_text(
                            json.dumps(summarize(cases, args.methods), indent=2) + "\n")


if __name__ == "__main__":
    main()
