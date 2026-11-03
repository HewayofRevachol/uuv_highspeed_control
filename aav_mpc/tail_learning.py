"""Policy-conditional tail value and lower-quantile margin experiment.

Geometry families, not episodes or seeds, determine the train/validation/test
split in run_stage3_collection.sh. The learned outputs are descriptive labels
under the fixed condensed-MPC continuation policy, not optimal values or
formal viability certificates.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from simulator.model import LoopPath

ROOT = Path(__file__).resolve().parents[1]
TASKS = json.loads((Path(__file__).parent / "configs/tasks.json").read_text())
OFFSETS = (10., 20., 40., 80.)
GAMMA = .98
TAIL_STEPS = 50


def row_features(row, path, task):
    s = float(row["progress_m"])
    x = np.array(json.loads(row["state_before_json"]))
    actual = np.array(json.loads(row["actuator_before_json"]))
    queue = np.array(json.loads(row["queue_before_json"]))[:3]
    current = np.array(json.loads(row.get("current_estimate_before_json") or "[0,0,0]"))
    xyz = path.position_at(s)
    tangent = path.tangent_at(s)
    yaw = float(np.interp(s, path.distances, path.yaw))
    pitch = path.pitch_at(s)
    offset = x[:3] - xyz
    ahead_yaw = np.interp(np.minimum(path.length, s + np.array((0.,) + OFFSETS)),
                          path.distances, path.yaw)
    ahead_pitch = np.interp(np.minimum(path.length, s + np.array((0.,) + OFFSETS)),
                            path.distances, path.pitch)
    yaw_changes = np.diff(ahead_yaw)
    pitch_changes = np.diff(ahead_pitch)
    scale = np.array([1_500_000., np.pi/4, np.pi/4])
    features = np.r_[np.dot(tangent, offset)/10,
        np.dot([-np.sin(yaw), np.cos(yaw)], offset[:2])/task["lateral_corridor_m"],
        offset[2]/task["depth_corridor_m"],
        np.arctan2(np.sin(x[4]-yaw), np.cos(x[4]-yaw))/np.pi,
        (x[3]-pitch)/.7, x[5:8]/[18, 6, 6], x[8:10]/.5,
        actual/scale, (queue/scale).ravel(), current/2,
        task["target_speed_mps"]/18, s/path.length,
        (path.length-s)/1000, yaw_changes/.5, pitch_changes/.5]
    heuristic = (task["target_speed_mps"]**2 * max(abs(yaw_changes / np.array(OFFSETS))) / 10
                 + max(abs(actual[1:]/scale[1:]))
                 + abs(features[1]) + abs(features[2]))
    return features.astype(np.float32), float(heuristic)


def tail_labels(rows, task):
    cost = []
    margin = []
    scale = np.array([1_500_000., np.pi/4, np.pi/4])
    for row in rows:
        actual = np.array([float(row["thrust_actual_n"]),
            math.radians(float(row["elevator_actual_deg"])),
            math.radians(float(row["rudder_actual_deg"]))])
        command = np.array(json.loads(row["command_json"]))
        lateral = abs(float(row["lateral_error_m"])) / task["lateral_corridor_m"]
        depth = abs(float(row["depth_error_m"])) / task["depth_corridor_m"]
        speed = abs(float(row["speed_error_mps"])) / task["speed_tolerance_mps"]
        pitch = abs(float(row["pitch_deg"])) / task["pitch_limit_deg"]
        cost.append(lateral*lateral + depth*depth + speed*speed +
                    .02 * sum((command/scale)**2) +
                    100 * (row["failed"] == "True" or row["timeout"] == "True"))
        margin.append(min(1-lateral, 1-depth, 1-speed, 1-pitch,
                          1-actual[0]/scale[0],
                          1-abs(actual[1])/scale[1],
                          1-abs(actual[2])/scale[2]))
    value = []
    lower = []
    for i in range(len(rows)):
        stop = min(len(rows), i+TAIL_STEPS)
        value.append(sum(GAMMA**j * cost[i+j] for j in range(stop-i)))
        lower.append(min(margin[i:stop]))
    return np.asarray(value, np.float32), np.asarray(lower, np.float32)


def load_split(raw_dir, split):
    x, y, m, heuristic, groups = [], [], [], [], []
    for file in sorted(raw_dir.glob(f"{split}_*.csv")):
        meta = json.loads(file.with_suffix(".json").read_text())
        task = dict(TASKS[meta["task"]])
        task["target_speed_mps"] = meta["target_speed_mps"] if "target_speed_mps" in meta else float(file.stem.split("_v")[-1].split("_")[0])
        path = LoopPath(ROOT / meta["path"])
        with file.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or "state_before_json" not in rows[0]:
            raise ValueError(f"{file} lacks measured-state fields")
        values, margins = tail_labels(rows, task)
        for j, row in enumerate(rows):
            before_row = dict(row)
            before_row["progress_m"] = row.get("progress_before_m", 0. if j == 0 else rows[j-1]["progress_m"])
            features, score = row_features(before_row, path, task)
            x.append(features); y.append(values[j]); m.append(margins[j])
            heuristic.append(score); groups.append(file.stem)
    if not x:
        raise ValueError(f"no {split} episodes in {raw_dir}")
    return (np.array(x), np.array(y), np.array(m),
            np.array(heuristic), np.array(groups))


class TailNet(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(features, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 2))

    def forward(self, x):
        return self.layers(x)


def summarize(y, m, score, value_pred, lower_pred, groups, threshold):
    event = (m < 0).astype(int)
    auc = (float(roc_auc_score(event, score)) if len(np.unique(event)) == 2 else None)
    lower = lower_pred - threshold
    safe = lower >= 0
    return {"samples": int(len(y)), "episodes": int(len(np.unique(groups))),
        "value_mae": float(np.mean(abs(value_pred-y))),
        "margin_mae": float(np.mean(abs(lower_pred-m))),
        "failure_auc": auc,
        "lower_margin_coverage": float(np.mean(m >= lower)),
        "predicted_safe_count": int(safe.sum()),
        "false_safe_count": int(np.sum(safe & (m < 0))),
        "false_safe_rate_among_predicted_safe": float(np.mean(m[safe] < 0)) if safe.any() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("aav_mpc/data/stage3/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("aav_mpc/data/stage3/model"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--log-value", action="store_true")
    parser.add_argument("--balanced-episodes", action="store_true")
    parser.add_argument("--select-risk", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data = {split: load_split(args.raw_dir, split) for split in ("train", "val", "test")}
    train_x = data["train"][0]
    mean = train_x.mean(0); std = train_x.std(0).clip(.05)
    net = TailNet(train_x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.tensor((train_x-mean)/std)
    y = torch.tensor(np.log1p(data["train"][1])/5 if args.log_value else data["train"][1]/100)
    m = torch.tensor(data["train"][2].clip(-10, 1))
    if args.balanced_episodes:
        groups = data["train"][4]
        sizes = {group: np.sum(groups == group) for group in np.unique(groups)}
        weights = torch.tensor(np.array([len(groups)/(len(sizes)*sizes[group])
                                          for group in groups], dtype=np.float32))
    else:
        weights = torch.ones(len(x))
    best, best_state = float("inf"), None
    history = []
    for epoch in range(args.epochs):
        net.train()
        permutation = torch.randperm(len(x))
        for ids in permutation.split(256):
            prediction = net(x[ids])
            value_loss = torch.nn.functional.huber_loss(
                prediction[:, 0], y[ids], delta=2., reduction="none")
            residual = m[ids]-prediction[:, 1]
            quantile_loss = torch.maximum(.1*residual, -.9*residual)
            loss = ((value_loss + quantile_loss) * weights[ids]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vx, vy, vm, _, _ = data["val"]
            vp = net(torch.tensor((vx-mean)/std)).numpy()
            if args.select_risk:
                residual = vm-vp[:, 1]
                validation = float(np.mean(np.maximum(.1*residual, -.9*residual)))
            elif args.log_value:
                validation = (float(np.mean(abs(vp[:, 0]*5-np.log1p(vy)))) +
                              float(np.mean(abs(vp[:, 1]-vm))))
            else:
                validation = (float(np.mean(abs(vp[:, 0]*100-vy))) +
                              float(np.mean(abs(vp[:, 1]-vm))))
        history.append({"epoch": epoch+1, "validation_score": validation})
        if validation < best:
            best = validation
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    outputs = {}
    with torch.no_grad():
        for split, (sx, sy, sm, heuristic, groups) in data.items():
            predictions = net(torch.tensor((sx-mean)/std)).numpy()
            outputs[split] = (sx, sy, sm, heuristic, groups, predictions)
    # Choose the lower-bound correction only on the held-out validation family.
    calibration = float(np.quantile(outputs["val"][5][:, 1]-outputs["val"][2], .9))
    report = {"seed": args.seed, "epochs": args.epochs,
              "log_value": args.log_value, "balanced_episodes": args.balanced_episodes,
              "select_risk": args.select_risk,
              "selected_epoch": int(np.argmin([r["validation_score"] for r in history])+1),
              "calibration_shift": calibration, "feature_count": int(train_x.shape[1]),
              "splits": {}, "inference_latency_us_per_sample": None}
    for split, (sx, sy, sm, heuristic, groups, predictions) in outputs.items():
        value_prediction = (np.expm1(np.clip(predictions[:, 0]*5, 0, 12))
                            if args.log_value else predictions[:, 0]*100)
        learned = summarize(sy, sm, -predictions[:, 1], value_prediction,
                            predictions[:, 1], groups, calibration)
        baseline_auc = (float(roc_auc_score((sm < 0).astype(int), heuristic))
                        if len(np.unique(sm < 0)) == 2 else None)
        learned["heuristic_failure_auc"] = baseline_auc
        learned["groups"] = {str(group): int(np.sum(groups == group)) for group in np.unique(groups)}
        report["splits"][split] = learned
    sample = torch.tensor((data["test"][0][:1]-mean)/std)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(1000): net(sample)
    report["inference_latency_us_per_sample"] = (time.perf_counter()-start)*1000
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std,
                "seed": args.seed, "feature_count": len(mean)}, args.out_dir/f"critic_seed{args.seed}.pt")
    (args.out_dir/f"report_seed{args.seed}.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.out_dir/f"history_seed{args.seed}.json").write_text(json.dumps(history, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
