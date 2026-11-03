"""Plot a failure timeline from one H15 per-period trace."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simulator.model import LoopPath


LIMITS = (("lateral_error_m", 8., "Lateral error (m)"),
          ("depth_error_m", 5., "Depth error (m)"),
          ("speed_error_mps", 2., "Along-path speed error (m/s)"),
          ("pitch_deg", 40., "Pitch (deg)"))


def plot(trace, out):
    with trace.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    meta = json.loads(trace.with_suffix(".json").read_text())
    path_file = Path(meta["path"])
    path_length = LoopPath(path_file).length
    manifest_file = path_file.parent / "manifest.json"
    events = []
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        events = [event for paths in manifest["splits"].values()
                  for item in paths if item["path"] == str(path_file)
                  for event in item.get("geometry_events_m", [])]
    s = np.array([float(row["progress_m"]) for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for axis, (field, limit, label) in zip(axes, LIMITS):
        value = np.array([float(row[field]) for row in rows])
        axis.plot(s, value, lw=1.2, color="#145a86")
        axis.axhline(limit, ls="--", color="#bd6326", lw=.8)
        axis.axhline(-limit, ls="--", color="#bd6326", lw=.8)
        axis.set_ylabel(label)
        axis.grid(alpha=.18)
    failed_qp = np.array([row["mpc_ok"].lower() != "true" for row in rows])
    if failed_qp.any():
        for axis in axes:
            axis.scatter(s[failed_qp], np.zeros(failed_qp.sum()), marker="x",
                         s=12, color="red", label="QP fallback")
    for axis in axes:
        for event in events:
            axis.axvline(event, color="#a0a0a0", lw=.55, ls=":")
        axis.axvline(path_length, color="#555", lw=.8, ls=":")
    for name, color in (("saturation", "#8f5aa9"), ("solver", "#a11")):
        event = meta.get("first_by_type", {}).get(name)
        if event:
            for axis in axes:
                axis.axvline(event["progress_m"], color=color, lw=.9, ls="--")
    axes[-1].set_xlabel("Path progress (m)")
    fig.suptitle(f"{Path(meta['path']).stem} | {meta['method']} | seed {meta['seed']}\n"
                 f"strict={meta['strict_success']} first={meta['first_anomaly']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.trace, args.out)
