#!/usr/bin/env python3
"""Create compact figures from the frozen analysis tables."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(__import__("os").environ.get("GS12_WEIGHT_OUT", HERE/"output")).resolve()
FIG = OUT/"figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    with (OUT/name).open() as f: return list(csv.DictReader(f))


def scan_heatmap():
    rows = read_csv("dev_scan_metrics.csv")
    policies = [(d, r) for d in (.7, 1., 1.4) for r in (.7, 1., 1.4)]
    task_ids = ["scan_depth_k8", "scan_depth_k32"]
    baselines = {t: next(float(x["J_track"]) for x in rows if x["task_id"] == t and
                         x["controller"] == "C4" and x["weight_policy"] == "1_1") for t in task_ids}
    values = np.full((9, 2), np.nan)
    for i, (d, rate) in enumerate(policies):
        for j, tid in enumerate(task_ids):
            found = next((x for x in rows if x["task_id"] == tid and x["controller"] == "C4" and
                          float(x["a_depth"]) == d and float(x["a_rudder_rate"]) == rate), None)
            if found and found["J_track"]:
                values[i, j] = 100*(baselines[tid]-float(found["J_track"]))/baselines[tid]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-50, vmax=50, aspect="auto")
    ax.set_xticks([0, 1], ["k=8", "k=32"])
    ax.set_yticks(range(9), [f"({d:g}, {r:g})" for d, r in policies])
    ax.set_xlabel("64 s single-frequency task")
    ax.set_ylabel(r"$(a_D, a_{\Delta\delta})$")
    ax.set_title("dev_scan J_track improvement vs C4 baseline (%)")
    for i in range(9):
        for j in range(2):
            ax.text(j, i, f"{values[i,j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="relative reduction in J_track")
    fig.tight_layout(); fig.savefig(FIG/"dev_scan_heatmap.png", dpi=180); plt.close(fig)


def locked_plot():
    rows = read_csv("test_locked_metrics.csv")
    tasks = ["locked_depth_k6", "locked_east_k30", "locked_mixed"]
    labels = ["C4", "C4 (1.4, 0.7)"]
    vals = np.zeros((3, 2))
    for i, task in enumerate(tasks):
        base = next(r for r in rows if r["task_id"] == task and r["controller"] == "C4" and
                    float(r["a_depth"]) == 1. and float(r["a_rudder_rate"]) == 1.)
        candidate = next(r for r in rows if r["task_id"] == task and r["controller"] == "C4" and
                         float(r["a_depth"]) == 1.4 and float(r["a_rudder_rate"]) == .7)
        vals[i] = [0., 100*(float(base["J_track"])-float(candidate["J_track"]))/float(base["J_track"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(3); width = .36
    ax.bar(x-width/2, vals[:, 0], width, label=labels[0])
    ax.bar(x+width/2, vals[:, 1], width, label=labels[1])
    ax.set_xticks(x, ["depth k=6", "east k=30", "mixed"])
    ax.set_ylabel("relative reduction in J_track (%)")
    ax.set_title("Locked paired tasks, common 32–64 s window")
    ax.axhline(0., color="black", linewidth=.8); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/"locked_test_jtrack.png", dpi=180); plt.close(fig)


def long_plot():
    data = json.loads((OUT/"long_matrix"/"long_summary.json").read_text())
    rows = [x for x in data["matrix_metrics"] if x["amplitude"] == .006]
    bands = ["B1", "B4"]; methods = ["C4", "C5", "C4_weighted"]
    x = np.arange(2); width = .23
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, method in enumerate(methods):
        values = [next(r["mean_sigma1"] for r in rows if r["band"] == b and r["method"] == method) for b in bands]
        ax.bar(x+(i-1)*width, values, width, label=method)
    ax.set_xticks(x, bands); ax.set_ylabel("mean σ₁ across eight excited lines")
    ax.set_title("15 m/s long-protocol empirical response, amplitude 0.006")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG/"long_matrix_sigma1.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    scan_heatmap(); locked_plot(); long_plot()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
