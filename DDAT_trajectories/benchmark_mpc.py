"""Evaluate the zero-residual DDAT controller on the held-out paths."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DDAT_train import evaluate_paths, summarize_paths
from simulator.batch_path_env import PathBank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=102600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--admm-iterations", type=int, default=400)
    parser.add_argument("--current-max-mps", type=float, default=1.)
    parser.add_argument("--attitude-pitch-weight", type=float, default=100.)
    parser.add_argument("--attitude-yaw-weight", type=float, default=30.)
    parser.add_argument("--fsp-enabled", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--fsp-filter-time", type=float, default=.4)
    parser.add_argument("--fsp-preview-time", type=float, default=None)
    args = parser.parse_args()
    args.backend = "admm"
    args.eval_envs = 5
    args.adaptation_time = .5
    args.current_filter_time = 1.
    torch.set_num_threads(2)
    files = sorted((Path(__file__).resolve().parent/"val").glob("*.csv"))
    bank = PathBank(files, args.device)
    rows = evaluate_paths(None, bank, args, args.seed)
    for row in rows:
        row["path"] = Path(row["path"]).name
    print(json.dumps({"seed": args.seed, "fsp_enabled": args.fsp_enabled,
                      "fsp_filter_time": args.fsp_filter_time,
                      "fsp_preview_time": args.fsp_preview_time,
                      "summary": summarize_paths(rows),
                      "paths": rows}, indent=2))


if __name__ == "__main__":
    main()
