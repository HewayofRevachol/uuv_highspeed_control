"""Freeze H15 quickstart inputs without running or inspecting test outcomes.

Run from the repository root with /bin/python3. This script refuses to
overwrite a frozen manifest or path with different bytes.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import osqp
import scipy
import torch

from aav_mpc.h15_challenge_paths import generate


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/datasets/h15_mainline_quickstart"
MANIFESTS = ROOT / "experiments/h15_mainline/manifests"
PROFILES = {
    "train": [
        ("train_turn_wide", "tight_turn", {"radius_m": 105., "entry_m": 38.}),
        ("train_reversal_wide", "reversal", {"radius_m": 105., "reversal_m": 195., "transition_m": 42.}),
        ("train_coupled_shallow", "coupled", {"radius_m": 100., "depth_amplitude_m": 4., "start_m": 80., "entry_m": 35., "wave_length_m": 220.}),
        ("train_late_entry", "late_entry", {"radius_m": 95., "entry_m": 145., "transition_m": 30.}),
    ],
    "dev": [
        ("dev_reversal", "reversal", {"radius_m": 78., "reversal_m": 165., "transition_m": 28.}),
        ("dev_coupled", "coupled", {"radius_m": 82., "depth_amplitude_m": 6.2, "start_m": 66., "entry_m": 23., "wave_length_m": 190.}),
    ],
    "test": [
        (f"test_coupled_{i:02d}", "coupled", p) for i, p in enumerate((
            {"radius_m": 68., "depth_amplitude_m": 7.1, "start_m": 57., "entry_m": 22., "wave_length_m": 185.},
            {"radius_m": 74., "depth_amplitude_m": 7.7, "start_m": 64., "entry_m": 18., "wave_length_m": 205.},
            {"radius_m": 80., "depth_amplitude_m": 8.3, "start_m": 72., "entry_m": 26., "wave_length_m": 175.},
            {"radius_m": 86., "depth_amplitude_m": 8.9, "start_m": 59., "entry_m": 31., "wave_length_m": 225.},
            {"radius_m": 71., "depth_amplitude_m": 9.1, "start_m": 68., "entry_m": 24., "wave_length_m": 195.},
            {"radius_m": 77., "depth_amplitude_m": 7.5, "start_m": 77., "entry_m": 20., "wave_length_m": 215.},
            {"radius_m": 83., "depth_amplitude_m": 8.1, "start_m": 62., "entry_m": 29., "wave_length_m": 180.},
            {"radius_m": 89., "depth_amplitude_m": 8.7, "start_m": 74., "entry_m": 27., "wave_length_m": 200.},
        ))
    ],
}
SEEDS = {"train": [31001, 31002, 31003],
         "dev": [32001, 32002, 32003],
         "test": list(range(33001, 33021))}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"frozen input differs: {path}")
    else:
        path.write_bytes(data)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def cpu_model():
    try:
        return next(line.split(":", 1)[1].strip() for line in
                    Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    entries = {}
    for split, profiles in PROFILES.items():
        cases = []
        for name, family, parameters in profiles:
            points, events = generate(family, parameters)
            path = DATA / f"{name}.csv"
            with __import__("io").StringIO(newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("north_m", "east_m", "depth_m", "target_tangent_speed_mps"))
                writer.writerows((*map(float, point), 15.) for point in points)
                raw = stream.getvalue().encode()
            write_once(path, raw)
            cases.append({"id": name, "generator_family": family,
                          "parameters": parameters, "geometry_events_m": events,
                          "path": str(path.relative_to(ROOT)), "path_sha256": sha256(raw)})
        manifest = {"schema": 1, "split": split,
                    "generator": "aav_mpc.h15_challenge_paths.generate",
                    "seed_role": "paired plant/process/current RNG seed",
                    "seeds": SEEDS[split], "cases": cases}
        path = MANIFESTS / f"{split}.json"
        raw = json_bytes(manifest)
        write_once(path, raw)
        if split == "test":
            path.chmod(0o444)
            for case in cases:
                (ROOT / case["path"]).chmod(0o444)
        entries[split] = {"path": str(path.relative_to(ROOT)),
                          "sha256": sha256(raw), "cases": len(cases),
                          "seeds": len(SEEDS[split])}
    config_path = ROOT / "aav_mpc/configs/h15.json"
    task_path = ROOT / "experiments/h15_mainline/task_contract.json"
    run = {
        "schema": 1, "method": "quickstart_freeze", "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "task_config": str(config_path.relative_to(ROOT)),
        "task_config_sha256": sha256(config_path.read_bytes()),
        "task_contract_sha256": sha256(task_path.read_bytes()),
        "manifests": entries, "candidate_grid_mps": [13., 14., 15., 16., 17.],
        "candidate_limit_k": 5, "screen_budget_ms": 5.,
        "intent_update_periods": 5, "normalization": {"status": "unfitted; train only"},
        "evaluation": {"control_period_s": .1, "integration_period_s": .02,
                       "strict_hist": "sampled 0.1 s legacy H15",
                       "strict_mainline": "all 0.02 s substeps plus fallback, empty set, deadline"},
        "environment": {"python_executable": os.path.realpath("/bin/python3"),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "osqp": osqp.__version__, "cpu": cpu_model(),
                        "threads": {key: os.environ.get(key) for key in
                                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "test_use": "Frozen before evaluation; no test threshold or checkpoint selection",
    }
    write_once(args.out / "quick_run_manifest.json", json_bytes(run))
    print(json.dumps({"out": str(args.out), "manifests": entries}, indent=2))


if __name__ == "__main__":
    main()
