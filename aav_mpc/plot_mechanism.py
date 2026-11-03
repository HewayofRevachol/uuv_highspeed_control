"""Export the frozen T1 paired timeline used in the stage-0/2 report."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator.model import LoopPath

ROOT = Path(__file__).resolve().parents[1]


def rows(file):
    with file.open(newline="") as stream:
        return list(csv.DictReader(stream))


def series(data, name):
    return np.array([float(row[name]) for row in data])


def main(out):
    path = LoopPath(ROOT / "DDAT_trajectories/val/constant_turn_01.csv")
    short = rows(ROOT / "aav_mpc/data/stage0/T1_osqp25_v2.csv")
    long = rows(ROOT / "aav_mpc/data/stage2/T1_endpoint40.csv")
    s = np.linspace(0, path.length, 1500)
    curvature = np.gradient(np.interp(s, path.distances, path.yaw), s)
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)
    axes[0].plot(s, curvature, color="black", lw=1.6)
    axes[0].set(xlabel="Path arc length (m)", ylabel="Curvature (1/m)")
    for data, label, color in ((short, "FSP + OSQP, 25 steps", "#b0473b"),
                               (long, "FSP + OSQP, 40 steps + endpoint extension", "#1b6e91")):
        t = series(data, "time_s")
        axes[1].plot(t, series(data, "rudder_command_deg"), label=label, color=color)
        axes[2].plot(t, series(data, "lateral_error_m"), label=label, color=color)
        axes[3].plot(t, series(data, "speed_error_mps"), label=label, color=color)
    axes[1].set(ylabel="Issued rudder (deg)")
    axes[2].axhline(5, color="gray", ls="--", lw=1)
    axes[2].axhline(-5, color="gray", ls="--", lw=1)
    axes[2].set(ylabel="Lateral error (m)")
    axes[3].axhline(1, color="gray", ls="--", lw=1)
    axes[3].axhline(-1, color="gray", ls="--", lw=1)
    axes[3].set(xlabel="Control time (s)", ylabel="Path speed error (m/s)")
    axes[1].legend(loc="best", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("aav_mpc/data/figures/t1_window_mechanism.png"))
    args = parser.parse_args()
    main(args.out)
