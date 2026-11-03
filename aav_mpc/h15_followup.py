"""Small, explicitly paired H15 mechanism experiments.

Run after the baseline screen. These are diagnostics, not a full-factorial
claim about every combination of path, plant and disturbance.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from aav_mpc.diagnose import run


ROOT = Path("data/datasets/h15_mpc_challenge")


def cases(suite, split="development", seeds=None, families=None, variants=None):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    paths = manifest["splits"][split]
    if families is not None:
        paths = [p for p in paths if p["family"] in families]
    if suite in ("horizon", "lookahead"):
        paths = [p for p in paths if p["family"] in ("reversal", "coupled")]
        for path in paths:
            for seed in (seeds if seeds is not None else (101026, 101027)):
                if suite == "horizon":
                    for horizon in (15, 20, 25, 30, 40, 50):
                        yield path, seed, f"N{horizon}", dict(
                            method="endpoint40", horizon_override=horizon,
                            randomize_plant=True, current_max=.4, process_noise=True)
                else:
                    for distance in (0., 5., 10.):
                        yield path, seed, f"L{int(distance)}", dict(
                            method="endpoint40", lookahead_offset_m=distance,
                            randomize_plant=True, current_max=.4, process_noise=True)
    elif suite == "condensed_horizon":
        paths = [p for p in paths if p["family"] == "coupled"]
        for path in paths:
            for seed in (seeds if seeds is not None else range(101026, 101031)):
                for horizon in (20, 25, 30, 35, 40):
                    yield path, seed, f"condensed_N{horizon}", dict(
                        method="condensed_endpoint40", horizon_override=horizon,
                        randomize_plant=True, current_max=.4, process_noise=True)
    elif suite == "lag":
        for path in paths:
            for seed in (seeds if seeds is not None else range(101026, 101031)):
                for no_lag in (False, True):
                    yield path, seed, "oracle_no_lag" if no_lag else "nominal_lag", dict(
                        method="endpoint20", oracle_no_lag=no_lag,
                        randomize_plant=True, current_max=.4, process_noise=True)
    elif suite == "constraints":
        for path in paths:
            for seed in (seeds if seeds is not None else range(101026, 101031)):
                for constrained in (False, True):
                    yield path, seed, "path_slew_qp" if constrained else "box_qp", dict(
                        method="rti_endpoint25", qp_path_constraints=constrained,
                        randomize_plant=True, current_max=.4, process_noise=True)
    elif suite == "soft_constraints":
        for path in paths:
            for seed in (seeds if seeds is not None else range(101026, 101031)):
                for variant, path_constraints, penalty in (
                        ("box_qp", False, None),
                        ("hard_path_qp", True, None),
                        ("soft_path_qp", True, 100.)):
                    if variants is not None and variant not in variants:
                        continue
                    yield path, seed, variant, dict(
                        method="rti_endpoint25", qp_path_constraints=path_constraints,
                        qp_soft_path_penalty=penalty,
                        randomize_plant=True, current_max=.4, process_noise=True)
    else:
        raise ValueError(suite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("horizon", "lookahead", "condensed_horizon", "lag", "constraints",
                                             "soft_constraints"), required=True)
    parser.add_argument("--split", choices=("development", "validation", "test"), default="development")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--families", nargs="+", choices=("tight_turn", "reversal", "coupled", "late_entry"))
    parser.add_argument("--variants", nargs="+", choices=("box_qp", "hard_path_qp", "soft_path_qp"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    paths = tuple(Path(p["path"]) for p in manifest["splits"][args.split]
                  if args.families is None or p["family"] in args.families)
    files = (ROOT / "manifest.json", Path("aav_mpc/configs/h15.json"),
             Path("aav_mpc/diagnose.py"), Path("aav_mpc/h15_followup.py"),
             Path("aav_mpc/controller.py"), Path("simulator/rti_mpc.py"), *paths)
    config = {"suite": args.suite, "split": args.split, "seeds": args.seeds,
              "families": args.families, "variants": args.variants,
              "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    run_file = args.out / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()) != config:
        parser.error("output directory has a different configuration or source version")
    run_file.write_text(json.dumps(config, indent=2) + "\n")
    summaries = []
    with (args.out / "console.log").open("a") as log:
        for path, seed, variant, kwargs in cases(args.suite, args.split, args.seeds,
                                                args.families, args.variants):
            file = args.out / f"{path['family']}_s{seed}_{variant}.csv"
            if file.exists() and file.with_suffix(".json").exists():
                summary = json.loads(file.with_suffix(".json").read_text())
            else:
                method = kwargs.pop("method")
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    summary = run("H15", method, file, path_override=path["path"],
                                  seed_override=seed, **kwargs)
            summary["family"] = path["family"]
            summary["variant"] = variant
            summaries.append(summary)
            print(path["family"], variant, summary["strict_success"],
                  summary["first_anomaly"], flush=True)
            (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
