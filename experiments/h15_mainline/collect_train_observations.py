"""Collect train-only H15 response histories for normalization and fitting."""

import argparse
import contextlib
import hashlib
import json
import os
import platform
from pathlib import Path

from aav_mpc.diagnose import run
from experiments.h15_mainline.observation_features import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]
MODES = {
    31001: {"target_domain": False, "process_noise": False, "current_max": 0.},
    31002: {"target_domain": True, "target_fin_gain_scale": .9,
            "process_noise": True, "current_max": .5},
    31003: {"target_domain": True, "target_fin_gain_scale": .8,
            "process_noise": True, "current_max": 1.},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    train_entry = freeze["manifests"]["train"]
    manifest_path = ROOT / train_entry["path"]
    if sha(manifest_path) != train_entry["sha256"]:
        raise RuntimeError("training manifest differs from freeze")
    manifest = json.loads(manifest_path.read_text())
    if tuple(manifest["seeds"]) != tuple(MODES):
        raise RuntimeError("unexpected training seed set")
    for case in manifest["cases"]:
        if sha(ROOT / case["path"]) != case["path_sha256"]:
            raise RuntimeError("training path differs from manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    sources = ["aav_mpc/diagnose.py", "simulator/batch_path_env.py",
               "experiments/h15_mainline/progress_reference.py",
               "experiments/h15_mainline/observation_features.py",
               "experiments/h15_mainline/collect_train_observations.py"]
    record = {"schema": 1, "method": "h15_train_observation_collection",
              "freeze_sha256": sha(args.freeze), "train_manifest_sha256": sha(manifest_path),
              "h15_config_sha256": sha(ROOT / "aav_mpc/configs/h15.json"),
              "source_sha256": {p: sha(ROOT / p) for p in sources},
              "cases": [c["id"] for c in manifest["cases"]],
              "seed_modes": MODES, "observation_features": FEATURE_NAMES,
              "python_executable": "/bin/python3", "python": platform.python_version(),
              "threads": {key: os.environ.get(key) for key in
                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              "scope": "train only; no dev or test observation files"}
    raw = json.dumps(record, indent=2) + "\n"
    run_path = args.out / "run.json"
    if run_path.exists() and run_path.read_text() != raw:
        raise RuntimeError("existing run configuration differs")
    run_path.write_text(raw)
    summaries = []
    summary_path = args.out / "summary.json"
    if summary_path.exists():
        summaries = json.loads(summary_path.read_text())
    done = {(item["case_id"], item["seed"]) for item in summaries}
    with (args.out / "console.log").open("a") as log:
        for case in manifest["cases"]:
            for seed in manifest["seeds"]:
                if (case["id"], seed) in done:
                    continue
                trace = args.out / f"{case['id']}_s{seed}.csv"
                if trace.exists():
                    raise RuntimeError(f"unindexed trace exists: {trace}")
                mode = MODES[seed]
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = run("H15", "condensed_endpoint35", trace,
                                 path_override=case["path"], seed_override=seed,
                                 randomize_plant=seed != 31001,
                                 current_profile="harmonic" if seed != 31001 else "steady",
                                 prediction_fin_gain_scale=.85,
                                 prediction_fin_tau_scale=2.,
                                 reference_intent_mps=15., log_substeps=False,
                                 **mode)
                result["case_id"] = case["id"]
                result["trace_sha256"] = sha(trace)
                summaries.append(result)
                summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
                print(f"{case['id']} s{seed}: {result['steps']} periods", flush=True)


if __name__ == "__main__":
    main()
