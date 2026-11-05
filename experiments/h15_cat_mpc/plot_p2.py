"""Create the P2 pilot diagnostics figure from saved run artifacts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--ppo-summary", type=Path, required=True)
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = args.out.parent/".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training = json.loads(args.training.read_text())
    ppo = json.loads(args.ppo_summary.read_text())["methods"]
    fixed = json.loads(args.fixed_summary.read_text())["methods"]
    ticks = ["0000", "1000", "2000", "4000"]
    decisions = [0, 1000, 2000, 4000]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    ax[0, 0].plot([0]+[x["decisions"] for x in training],
                  [0]+[x["mean_decision_reward"] for x in training], "o-")
    ax[0, 0].set(title="Training reward", xlabel="gate decisions", ylabel="discounted return")

    means = [x["alpha_mean"] for x in training]
    stds = [x["alpha_std"] for x in training]
    for j, label in enumerate((r"$\alpha_\theta$", r"$\alpha_d$")):
        ax[0, 1].errorbar([x["decisions"] for x in training],
            [v[j] for v in means], yerr=[v[j] for v in stds], marker="o", label=label)
    ax[0, 1].set(title="Applied gate mean ± SD", xlabel="gate decisions", ylim=(0, 1))
    ax[0, 1].legend()

    raw_std = []
    for tick in ticks:
        checkpoint = torch.load(args.checkpoints/f"ckpt_{tick}.pt",
                                map_location="cpu", weights_only=False)
        raw_std.append(torch.exp(checkpoint["policy"]["log_std"]).numpy())
    for j, label in enumerate((r"$z_\theta$", r"$z_d$")):
        ax[0, 2].plot(decisions, [x[j] for x in raw_std], "o-", label=label)
    ax[0, 2].set(title="Raw Gaussian action SD", xlabel="gate decisions")
    ax[0, 2].legend()

    method_names = ["A0", "A4"]+["PPO_"+x for x in ticks]
    method_data = [fixed[x] for x in ("A0", "A4")]+[ppo["PPO_"+x] for x in ticks]
    ax[1, 0].bar(method_names, [x["strict_mainline_completed"] for x in method_data])
    ax[1, 0].set(title="Strict mainline completions", ylabel="episodes / 6")
    ax[1, 0].tick_params(axis="x", rotation=25)

    fallback = [x["fallback_periods"]+x["backup_periods"] for x in method_data]
    violations = [sum(x["physical_violations_0p02s"].values()) for x in method_data]
    x = np.arange(len(method_names))
    ax[1, 1].bar(x-.18, fallback, width=.36, label="fallback + backup")
    ax[1, 1].bar(x+.18, violations, width=.36, label="0.02 s violations")
    ax[1, 1].set(title="Interventions and substep events", ylabel="counts")
    ax[1, 1].set_xticks(x, method_names, rotation=25)
    ax[1, 1].legend()

    p95 = [x["period_ms"]["p95"] for x in method_data]
    maximum = [x["period_ms"]["max"] for x in method_data]
    ax[1, 2].plot(method_names, p95, "o-", label="p95")
    ax[1, 2].plot(method_names, maximum, "x--", label="max")
    ax[1, 2].axhline(100., color="red", linestyle=":", label="100 ms deadline")
    ax[1, 2].set(title="Full control-period time", ylabel="ms")
    ax[1, 2].tick_params(axis="x", rotation=25)
    ax[1, 2].legend()
    for item in ax.flat:
        item.grid(alpha=.2)
    fig.savefig(args.out, dpi=160)
    print(str(args.out))


if __name__ == "__main__":
    main()
