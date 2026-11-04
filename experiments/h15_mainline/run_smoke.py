"""Three paired H15 reference-chain smoke cases; no learned policy or filter."""
import argparse
import contextlib
import hashlib
import json
import os
import platform
from pathlib import Path

from aav_mpc.diagnose import run


ROOT = Path(__file__).resolve().parents[2]
CASES = (
    ("simple", "train", "train_turn_wide", 31001, False),
    ("coupled", "dev", "dev_coupled", 32001, False),
    ("mismatch", "dev", "dev_reversal", 32002, True),
)


def read_manifest(split):
    path = ROOT / f"experiments/h15_mainline/manifests/{split}.json"
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(args.freeze.read_text())
    for split, item in freeze["manifests"].items():
        if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {split}")
    sources = ("aav_mpc/diagnose.py", "simulator/batch_path_env.py",
               "experiments/h15_mainline/progress_reference.py",
               "experiments/h15_mainline/run_smoke.py")
    run_record = {
        "schema": 1, "method": "h15_reference_chain_smoke",
        "freeze": str(args.freeze),
        "freeze_sha256": hashlib.sha256(args.freeze.read_bytes()).hexdigest(),
        "cases": CASES, "intents_mps": [15., 14.],
        "stage_profiling": True,
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in sources},
        "python": platform.python_version(),
        "threads": {key: os.environ.get(key) for key in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "limitations": ["no candidate filter", "no state-constrained MPC",
                        "no estimator", "no RL", "diagnostic timing includes substep logging"],
    }
    run_file = args.out / "run.json"
    raw = json.dumps(run_record, indent=2) + "\n"
    if run_file.exists() and run_file.read_text() != raw:
        raise RuntimeError("run.json differs from the frozen run configuration")
    run_file.write_text(raw)
    if args.review_only:
        summaries = json.loads((args.out / "summary.json").read_text())
        if len(summaries) != 2 * len(CASES):
            raise RuntimeError("incomplete smoke set")
        for result in summaries:
            path = ROOT / result["trace"]
            if not path.exists() or not path.with_name(path.stem + "_substeps.csv").exists():
                raise RuntimeError(f"missing trajectory: {path}")
        print(f"reviewed {len(summaries)} smoke trajectories")
        return
    summaries = []
    with (args.out / "console.log").open("a") as log:
        for label, split, case_id, seed, mismatch in CASES:
            entry = next(case for case in read_manifest(split)["cases"]
                         if case["id"] == case_id)
            file = ROOT / entry["path"]
            if hashlib.sha256(file.read_bytes()).hexdigest() != entry["path_sha256"]:
                raise RuntimeError(f"path hash mismatch: {file}")
            for intent in (15., 14.):
                stem = f"{label}_s{seed}_intent{int(intent)}"
                trace = args.out / f"{stem}.csv"
                if trace.exists():
                    raise RuntimeError(f"refusing to overwrite {trace}")
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = run(
                        "H15", "condensed_endpoint35", trace,
                        path_override=entry["path"], seed_override=seed,
                        randomize_plant=mismatch, process_noise=mismatch,
                        current_max=1. if mismatch else 0.,
                        current_profile="harmonic" if mismatch else "steady",
                        target_domain=mismatch,
                        target_components=("fin", "integrator", "actuator") if mismatch else None,
                        prediction_fin_gain_scale=.85, prediction_fin_tau_scale=2.,
                        reference_intent_mps=intent, log_substeps=True,
                        profile_stages=True)
                result.update(case_label=label, intent_mps=intent,
                              diagnostic_only=True, candidate_filter="absent")
                summaries.append(result)
                (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
                print(f"{stem}: steps={result['steps']} hist={result['strict_hist']} "
                      f"substep={result['strict_substep_success']} "
                      f"first={result['first_anomaly']} "
                      f"p99={result['control_period_p99_ms']:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
