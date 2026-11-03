"""Isolate H15 target-domain changes on paired path/seed cells."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from aav_mpc.diagnose import run


DEFAULT_METHODS = ("endpoint20", "condensed_endpoint40")
AVAILABLE_METHODS = ("endpoint20", "condensed_endpoint35", "condensed_endpoint40")
VARIANTS = {
    "source": (),
    "current": (),
    "fin": ("fin",),
    "integrator": ("integrator",),
    "actuator": ("actuator",),
    "fin_actuator": ("fin", "actuator"),
    "full": ("fin", "integrator", "actuator"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cases, methods, variants):
    result = {"cases": len(cases), "methods": {}}
    for method in methods:
        result["methods"][method] = {}
        for variant in variants:
            selected = [c for c in cases if c["method"] == method and c["variant"] == variant]
            if selected:
                result["methods"][method][variant] = {
                    "episodes": len(selected),
                    "strict_success": sum(c["strict_success"] for c in selected),
                    "first_anomaly_types": {k: sum((c["first_anomaly"] or [0, "none"])[1] == k
                                                  for c in selected)
                                            for k in sorted(set((c["first_anomaly"] or [0, "none"])[1]
                                                                for c in selected))},
                    "fallback_count": sum(c["fallback_count"] for c in selected),
                }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("challenge", "replication"), default="challenge")
    parser.add_argument("--split", choices=("development", "validation", "test", "replication"),
                        default="development")
    parser.add_argument("--families", nargs="+",
                        default=["coupled"])
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS,
                        default=list(DEFAULT_METHODS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["source", "current", "fin", "integrator", "actuator", "full"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(101026, 101031)))
    parser.add_argument("--reuse-source-dir", type=Path)
    parser.add_argument("--reuse-target-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.corpus == "replication") != (args.split == "replication"):
        parser.error("replication corpus requires replication split")
    if any(len(set(values)) != len(values) for values in (args.seeds, args.methods,
                                                          args.variants, args.families)):
        parser.error("seeds, methods, variants and families must be unique")
    root = Path("data/datasets/h15_mpc_replication" if args.corpus == "replication"
                else "data/datasets/h15_mpc_challenge")
    manifest = json.loads((root / "manifest.json").read_text())
    paths = [p for p in manifest["splits"][args.split] if p["family"] in args.families]
    if len(paths) != len(args.families):
        parser.error("some requested families are missing from the manifest")
    files = [Path("aav_mpc/diagnose.py"), Path("aav_mpc/h15_domain_ablation.py"),
             Path("aav_mpc/h15_target_domain.py"), Path("aav_mpc/configs/h15.json"),
             Path("simulator/batch_path_env.py"), root / "manifest.json",
             *(Path(p["path"]) for p in paths)]
    reused = {}
    for variant, directory, suffix in (("source", args.reuse_source_dir, ""),
                                       ("full", args.reuse_target_dir, "_target")):
        if directory is None:
            continue
        if variant not in args.variants:
            parser.error(f"{variant} reuse requires that variant in --variants")
        run_config = json.loads((directory / "run.json").read_text())
        if (run_config.get("corpus") != args.corpus or
                run_config.get("split") != args.split or
                set(args.methods) - set(run_config.get("methods", [])) or
                set(args.seeds) - set(run_config.get("seeds", []))):
            parser.error(f"{variant} reuse directory has a different experiment grid")
        if variant == "source" and (run_config.get("current_max_mps") != .4 or
                                    not run_config.get("randomize_plant") or
                                    not run_config.get("process_noise")):
            parser.error("source reuse directory has different plant/disturbance settings")
        if variant == "full" and run_config.get("source_dir") is None:
            parser.error("full target reuse must come from paired target run")
        if variant == "full" and args.reuse_source_dir is not None:
            if (run_config.get("source_dir") != str(args.reuse_source_dir) or
                    run_config.get("source_run_sha256") != sha256(args.reuse_source_dir / "run.json")):
                parser.error("full target reuse is not paired with the selected source run")
        reused[variant] = {"directory": str(directory),
                           "run_sha256": sha256(directory / "run.json"),
                           "episodes": {str(directory / f"{args.split}_{p['family']}_s{seed}_{method}{suffix}.json"):
                                        sha256(directory / f"{args.split}_{p['family']}_s{seed}_{method}{suffix}.json")
                                        for p in paths for seed in args.seeds for method in args.methods}}
    config = {"corpus": args.corpus, "split": args.split, "families": args.families,
              "seeds": args.seeds, "methods": args.methods,
              "variants": {name: list(VARIANTS[name]) for name in args.variants},
              "reused": reused,
              "sha256": {str(p): sha256(p) for p in files}}
    args.out.mkdir(parents=True, exist_ok=True)
    run_file = args.out / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()) != config:
        parser.error("output directory has a different configuration or source version")
    run_file.write_text(json.dumps(config, indent=2) + "\n")
    cases = []
    with (args.out / "console.log").open("a") as log:
        for path in paths:
            for seed in args.seeds:
                for method in args.methods:
                    for variant in args.variants:
                        components = VARIANTS[variant]
                        file = args.out / f"{args.split}_{path['family']}_s{seed}_{method}_{variant}.csv"
                        if variant in reused:
                            directory = Path(reused[variant]["directory"])
                            suffix = "_target" if variant == "full" else ""
                            source_file = directory / f"{args.split}_{path['family']}_s{seed}_{method}{suffix}.json"
                            case = json.loads(source_file.read_text())
                            if (case["path"] != path["path"] or case["seed"] != seed or
                                    case["method"] != method or
                                    case.get("target_domain") != (variant == "full")):
                                raise ValueError(f"reused episode mismatch: {source_file}")
                            case["reused_from"] = str(source_file)
                        elif file.exists() and file.with_suffix(".json").exists():
                            case = json.loads(file.with_suffix(".json").read_text())
                        else:
                            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                                case = run("H15", method, file, path_override=path["path"],
                                           seed_override=seed, randomize_plant=True,
                                           current_max=1. if variant in ("current", "full") else .4,
                                           current_profile="harmonic" if variant in ("current", "full") else "steady",
                                           process_noise=True,
                                           target_domain=variant in ("fin", "integrator", "actuator", "fin_actuator", "full"),
                                           target_components=components if variant in ("fin", "integrator", "actuator", "fin_actuator", "full") else None)
                        case["variant"] = variant
                        cases.append(case)
                        print(path["family"], seed, method, variant,
                              case["strict_success"], case["first_anomaly"], flush=True)
                        (args.out / "summary.json").write_text(
                            json.dumps(summarize(cases, args.methods, args.variants), indent=2) + "\n")


if __name__ == "__main__":
    main()
