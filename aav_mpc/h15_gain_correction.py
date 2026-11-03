"""Paired diagnostic: give condensed MPC the target domain's 0.9 fin scale.

The target scale is withheld from the ordinary controller, so this is an
oracle mechanism check, not a deployable estimator or final baseline.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from aav_mpc.diagnose import run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cases):
    result = {"cases": len(cases), "variants": {}}
    for variant in ("fin", "full"):
        pairs = [(row["baseline"], row["corrected"]) for row in cases
                 if row["variant"] == variant]
        if not pairs:
            continue
        result["variants"][variant] = {
            "pairs": len(pairs),
            "baseline_success": sum(a["strict_success"] for a, b in pairs),
            "corrected_success": sum(b["strict_success"] for a, b in pairs),
            "improved": sum(not a["strict_success"] and b["strict_success"] for a, b in pairs),
            "worsened": sum(a["strict_success"] and not b["strict_success"] for a, b in pairs),
            "baseline_fallbacks": sum(a["fallback_count"] for a, b in pairs),
            "corrected_fallbacks": sum(b["fallback_count"] for a, b in pairs),
            "corrected_p99_max_ms": max(b["control_period_p99_ms"] for a, b in pairs),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["coupled_a", "coupled_c"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(101041, 101046)))
    parser.add_argument("--variants", nargs="+", choices=("fin", "full"),
                        default=["fin", "full"])
    parser.add_argument("--baseline-ablation-dir", type=Path, required=True)
    parser.add_argument("--baseline-target-dir", type=Path, required=True)
    parser.add_argument("--baseline-gain-dir", type=Path,
                        help="compare against an already completed 0.9 gain correction")
    parser.add_argument("--prediction-fin-tau-scale", type=float, default=1.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.prediction_fin_tau_scale < 10:
        parser.error("prediction fin tau scale must lie in (0, 10)")
    if any(len(set(values)) != len(values) for values in
           (args.families, args.seeds, args.variants)):
        parser.error("families, seeds and variants must be unique")
    manifest_file = Path("data/datasets/h15_mpc_replication/manifest.json")
    manifest = json.loads(manifest_file.read_text())
    paths = [p for p in manifest["splits"]["replication"]
             if p["family"] in args.families]
    if len(paths) != len(args.families):
        parser.error("all families must be replication paths")
    ablation_config = json.loads((args.baseline_ablation_dir / "run.json").read_text())
    target_config = json.loads((args.baseline_target_dir / "run.json").read_text())
    if (ablation_config.get("corpus") != "replication" or
            ablation_config.get("split") != "replication" or
            ("fin" in args.variants and
             set(args.families) - set(ablation_config.get("families", []))) or
            set(args.seeds) - set(ablation_config.get("seeds", [])) or
            "condensed_endpoint35" not in ablation_config.get("methods", []) or
            target_config.get("corpus") != "replication" or
            target_config.get("split") != "replication" or
            "condensed_endpoint35" not in target_config.get("methods", [])):
        parser.error("baseline directories do not cover the requested replication cells")
    baseline = {}
    for path in paths:
        family = path["family"]
        for seed in args.seeds:
            for variant in args.variants:
                file = (args.baseline_gain_dir /
                        f"replication_{family}_s{seed}_condensed_endpoint35_{variant}_gain09.json"
                        if args.baseline_gain_dir else
                        args.baseline_ablation_dir /
                        f"replication_{family}_s{seed}_condensed_endpoint35_fin.json"
                        if variant == "fin" else
                        args.baseline_target_dir /
                        f"replication_{family}_s{seed}_condensed_endpoint35_target.json")
                case = json.loads(file.read_text())
                if (case.get("path") != path["path"] or case.get("seed") != seed or
                        case.get("method") != "condensed_endpoint35" or
                        case.get("target_components") != (["fin"] if variant == "fin"
                                                          else ["fin", "integrator", "actuator"]) or
                        (args.baseline_gain_dir is not None and
                         case.get("prediction_fin_gain_scale") != .9)):
                    raise ValueError(f"baseline case mismatch: {file}")
                baseline[(family, seed, variant)] = (case, file)
    sources = [Path("aav_mpc/h15_gain_correction.py"), Path("aav_mpc/diagnose.py"),
               Path("aav_mpc/controller.py"), Path("aav_mpc/condensed_mpc.py"),
               Path("aav_mpc/h15_target_domain.py"), Path("aav_mpc/configs/h15.json"),
               Path("simulator/rti_symbolic.py"), Path("simulator/batch_path_env.py"),
               manifest_file, *(Path(p["path"]) for p in paths)]
    config = {"families": args.families, "seeds": args.seeds, "variants": args.variants,
              "prediction_fin_gain_scale": .9,
              "prediction_fin_tau_scale": args.prediction_fin_tau_scale,
              "source_sha256": {str(p): digest(p) for p in sources},
              "baseline_run_sha256": {
                  "ablation": digest(args.baseline_ablation_dir / "run.json"),
                  "target": digest(args.baseline_target_dir / "run.json"),
                  "gain": digest(args.baseline_gain_dir / "run.json")
                      if args.baseline_gain_dir else None},
              "baseline_episode_sha256": {str(file): digest(file)
                  for _, file in baseline.values()}}
    args.out.mkdir(parents=True, exist_ok=True)
    run_file = args.out / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()) != config:
        parser.error("output directory belongs to a different configuration")
    run_file.write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    with (args.out / "console.log").open("a") as log:
        for path in paths:
            for seed in args.seeds:
                for variant in args.variants:
                    family = path["family"]
                    tag = ("gain09" if args.prediction_fin_tau_scale == 1. else
                           f"gain09_tau{round(args.prediction_fin_tau_scale*100)}")
                    file = args.out / f"replication_{family}_s{seed}_condensed_endpoint35_{variant}_{tag}.csv"
                    if file.exists() and file.with_suffix(".json").exists():
                        corrected = json.loads(file.with_suffix(".json").read_text())
                    else:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            corrected = run("H15", "condensed_endpoint35", file,
                                path_override=path["path"], seed_override=seed,
                                randomize_plant=True, process_noise=True,
                                current_max=1. if variant == "full" else .4,
                                current_profile="harmonic" if variant == "full" else "steady",
                                target_domain=True,
                                target_components=("fin", "integrator", "actuator")
                                    if variant == "full" else ("fin",),
                                prediction_fin_gain_scale=.9,
                                prediction_fin_tau_scale=args.prediction_fin_tau_scale)
                    original = baseline[(family, seed, variant)][0]
                    rows.append({"family": family, "seed": seed, "variant": variant,
                                 "baseline": original, "corrected": corrected})
                    print(family, seed, variant, original["strict_success"],
                          corrected["strict_success"], corrected["first_anomaly"], flush=True)
                    (args.out / "summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
