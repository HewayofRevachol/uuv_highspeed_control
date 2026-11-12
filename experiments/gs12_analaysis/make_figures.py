"""Plot completed paired_mismatch.py results, without rerunning control."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/"paired_results"
summary = json.loads((RESULTS/"summary.json").read_text())
names = ["design", "fin_0p9", "second_order", "combined", "conservative_combined"]
labels = ["Matched design", "Fin gain x0.9", "Second-order actuator",
          "Both mismatches", "Both + conservative predictor"]
colors = ["#34495e", "#2874a6", "#d68910", "#c0392b", "#148f77"]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
for row, geometry in enumerate(("gentle", "coupled")):
    for name, label, color in zip(names, labels, colors):
        case = json.loads((RESULTS/f"{geometry}_{name}.json").read_text())
        rows = case["records"]
        t = [r["time_s"] for r in rows]
        axes[row, 0].plot(t, [r["error"][2] for r in rows], label=label, color=color)
        axes[row, 1].plot(t, np.rad2deg([r["state"][3] for r in rows]), label=label, color=color)
        if case["first_diagnostic_limit_violation"]:
            for col in range(2):
                ax = axes[row, col]
                y = rows[-1]["error"][2] if col == 0 else np.rad2deg(rows[-1]["state"][3])
                ax.plot(t[-1], y, "x", color=color, markersize=8, mew=2)
    axes[row, 0].set_ylabel("Depth tracking error (m)")
    axes[row, 1].set_ylabel("Actual pitch (deg)")
    axes[row, 0].set_title(f"{geometry.title()}: common timed reference")
    axes[row, 1].set_title(f"{geometry.title()}: x = first limit violation, then stop")
    for ax in axes[row]:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=.25)
axes[1, 1].axhline(40, color="gray", lw=.8, ls="--")
axes[1, 1].axhline(-40, color="gray", lw=.8, ls="--")
axes[0, 0].legend(fontsize=8)
fig.suptitle("GS-12 | Actual repository MPC, deterministic 24 s diagnosis\n"
    "North speed 15 m/s; no current/noise/low-level compensation; not an H15 acceptance run", fontsize=12)
fig.savefig(ROOT/"mismatch_effects.png", dpi=180)
fig.savefig(ROOT/"mismatch_effects.pdf")

# Equal-length comparisons between different controllers on the SAME plant.
common = {}
for geometry in ("gentle", "coupled"):
    cases = {name: json.loads((RESULTS/f"{geometry}_{name}.json").read_text()) for name in names}
    n = min(case["periods"] for case in cases.values())
    common[geometry] = {"periods": n, "end_s": n*.1, "metrics": {}}
    for name, case in cases.items():
        errors = np.asarray([r["error"] for r in case["records"][:n]])
        common[geometry]["metrics"][name] = {"RMSE": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            "mean_error": errors.mean(axis=0).tolist(), "variance": errors.var(axis=0).tolist()}
(ROOT/"common_window_metrics.json").write_text(json.dumps(common, indent=2)+"\n")
print(json.dumps({g: {"end_s": v["end_s"], "depth_RMSE_by_case": {
    k: m["RMSE"][2] for k, m in v["metrics"].items()}} for g, v in common.items()}, indent=2))
