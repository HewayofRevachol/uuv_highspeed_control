"""Frozen critic evaluation on paired randomized-plant/current trajectories."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from aav_mpc.tail_learning import ROOT, TASKS, TailNet, row_features, tail_labels, summarize
from simulator.model import LoopPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
        default=Path("aav_mpc/data/stage3/model_aligned/critic_seed31.pt"))
    parser.add_argument("--calibration", type=Path,
        default=Path("aav_mpc/data/stage3/model_aligned/report_seed31.json"))
    parser.add_argument("--paired-dir", type=Path, default=Path("aav_mpc/data/stage4/paired"))
    parser.add_argument("--out", type=Path, default=Path("aav_mpc/data/stage3/ood_randomized.json"))
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = TailNet(state["feature_count"])
    net.load_state_dict(state["state_dict"])
    net.eval()
    threshold = json.loads(args.calibration.read_text())["calibration_shift"]
    report = {"checkpoint": str(args.checkpoint), "calibration": str(args.calibration),
              "groups": {}}
    for task_name in ("T1", "T3"):
        xs, ys, ms, hs, groups = [], [], [], [], []
        for file in sorted(args.paired_dir.glob(f"{task_name}_*_condensed25.csv")):
            meta = json.loads(file.with_suffix(".json").read_text())
            task = dict(TASKS[task_name])
            task["target_speed_mps"] = meta["target_speed_mps"]
            path = LoopPath(ROOT / meta["path"])
            with file.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            values, margins = tail_labels(rows, task)
            for i, row in enumerate(rows):
                before = dict(row)
                before["progress_m"] = row.get("progress_before_m", 0. if i == 0 else rows[i-1]["progress_m"])
                features, heuristic = row_features(before, path, task)
                xs.append(features); ys.append(values[i]); ms.append(margins[i])
                hs.append(heuristic); groups.append(file.stem)
        if not xs:
            continue
        x = np.array(xs); y = np.array(ys); m = np.array(ms)
        with torch.no_grad():
            predicted = net(torch.tensor((x-state["mean"])/state["std"])).numpy()
        value = np.expm1(np.clip(predicted[:, 0]*5, 0, 12))
        metrics = summarize(y, m, -predicted[:, 1], value, predicted[:, 1],
                            np.array(groups), threshold)
        metrics["heuristic_failure_auc"] = (float(roc_auc_score(m < 0, hs))
            if len(np.unique(m < 0)) == 2 else None)
        report["groups"][task_name] = metrics
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
