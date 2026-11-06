"""Paired GS-12 episode and empirical frequency summary."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gs12_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.gs12_gate.data_io import file_hash, read_cases, read_json, write_npz
from experiments.gs12_gate.run import _load_run
from simulator.model import rotation_ned


def _episode_path(run, split, case_id, method):
    return Path(run)/"episodes"/split/case_id/method


def _normalized_errors(substeps):
    states, refs = substeps[:, 1:11], substeps[:, 11:19]
    vn = np.array([(rotation_ned(x[3], x[4])@x[5:8])[0] for x in states])
    return np.column_stack(((states[:, 1]-refs[:, 1])/8,
                            (states[:, 2]-refs[:, 2])/5,
                            (vn-refs[:, 5])/2,
                            (states[:, 0]-refs[:, 0])/8))


def _paired_rows(run, split, methods, output_split=None):
    _, cases = _load_run(run)
    selected = [case for case in cases if case["split"] == split]
    output_split = output_split or split
    rows = []
    for case in selected:
        summaries = {}
        for method in methods:
            path = _episode_path(run, output_split, case["case_id"], method)
            if not (path/"episode.json").exists():
                raise FileNotFoundError(f"missing paired episode: {path}")
            summaries[method] = read_json(path/"episode.json")
        prefix = min((s["duration_actual_s"] if s["first_failure_s"] is None else
                      max(0., s["first_failure_s"]-.02)) for s in summaries.values())
        for method, summary in summaries.items():
            path = _episode_path(run, output_split, case["case_id"], method)
            with np.load(path/"substeps.npz") as substep_file:
                substeps = substep_file["values"]
            count_substeps = min(len(substeps), int(round(prefix/.02)))
            sample = substeps[:count_substeps]
            if count_substeps:
                normalized = _normalized_errors(sample)
                rmse = (float(np.sqrt(np.mean(np.sum(normalized[:, :3]**2, axis=1))))
                        if np.isfinite(normalized).all() else None)
                north_rmse = (float(np.sqrt(np.mean(normalized[:, 3]**2)))*8
                              if np.isfinite(normalized).all() else None)
                vn_rmse = (float(np.sqrt(np.mean(normalized[:, 2]**2)))*2
                           if np.isfinite(normalized).all() else None)
                position_rmse = (float(np.sqrt(np.mean(np.sum(
                    (sample[:, 1:4]-sample[:, 11:14])**2, axis=1))))
                    if np.isfinite(sample).all() else None)
            else:
                rmse = north_rmse = vn_rmse = position_rmse = None
            full = _normalized_errors(substeps) if summary["complete"] else None
            full_rmse = (float(np.sqrt(np.mean(np.sum(full[:, :3]**2, axis=1))))
                         if full is not None and np.isfinite(full).all() else None)
            with np.load(path/"control.npz") as data:
                alpha = data["alpha"]
                clip_fraction = float(np.mean(data["clip"]))
                qp_failures = int(np.sum(~data["qp_ok"]))
                saturation_fraction = float(np.mean(data["saturation_fraction"]))
                deadline_misses = int(np.sum(data["deadline_miss"]))
                p95_period_ms = float(np.quantile(data["total_ms"], .95))
            rows.append({"case_id": case["case_id"], "split": split,
                         "family": case["family"], "speed_mps": case["speed"],
                         "maneuver": case["maneuver"],
                         "capacity_limited": case["capacity_limited"],
                         "method": method, "complete": summary["complete"],
                         "first_failure_s": summary["first_failure_s"],
                         "failure_reason": ";".join(summary["failure_reason"] or []),
                         "common_prefix_s": prefix, "common_prefix_rmse": rmse,
                         "full_success_rmse": full_rmse,
                         "north_lag_rmse_m": north_rmse,
                         "position_rmse_m": position_rmse,
                         "north_velocity_rmse_mps": vn_rmse,
                         "peak_abs_pitch_deg": float(np.max(np.abs(substeps[:, 4]))*180/np.pi),
                         "tracking_cost_integral": summary["tracking_cost_integral"],
                         "qp_failures": qp_failures, "deadline_misses": deadline_misses,
                         "period_p95_ms": p95_period_ms,
                         "saturation_fraction": saturation_fraction,
                         "clip_fraction": clip_fraction,
                         "alpha_mean": json.dumps(np.mean(alpha, axis=0).tolist())})
    return rows


def _mean_or_nan(values):
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _plot_or_note(axis, xs, ys, title, ylabel):
    if len(xs):
        axis.bar(xs, ys)
        axis.tick_params(axis="x", labelrotation=45)
    else:
        axis.text(.5, .5, "No valid data", ha="center", va="center",
                  transform=axis.transAxes)
    axis.set_title(title); axis.set_ylabel(ylabel)


def _charts(out, rows, run, split, methods):
    families = sorted(set(row["family"] for row in rows))
    fig, axis = plt.subplots(figsize=(10, 4))
    width = .8/max(len(methods), 1)
    for i, method in enumerate(methods):
        values = [_mean_or_nan([r["common_prefix_rmse"] for r in rows
                           if r["family"] == family and r["method"] == method])
                  for family in families]
        axis.bar(np.arange(len(families))-.4+(i+.5)*width, values, width, label=method)
    axis.set_xticks(np.arange(len(families)), families, rotation=45)
    axis.set_ylabel("Paired common-prefix normalized RMSE")
    axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"families.png", dpi=150); plt.close(fig)

    fig, (axis, failure_axis) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    positions = [_mean_or_nan([r["position_rmse_m"] for r in rows if r["method"] == m])
                 for m in methods]
    velocities = [_mean_or_nan([r["north_velocity_rmse_mps"] for r in rows
                               if r["method"] == m]) for m in methods]
    axis.bar(np.arange(len(methods))-.18, positions, .36, label="position RMSE (m)")
    axis.bar(np.arange(len(methods))+.18, velocities, .36, label="VN RMSE (m/s)")
    axis.legend(fontsize=8)
    for i, method in enumerate(methods):
        times = [r["first_failure_s"] for r in rows if r["method"] == method
                 and r["first_failure_s"] is not None]
        failure_axis.scatter(np.full(len(times), i), times, s=18)
    failure_axis.set_xticks(np.arange(len(methods)), methods, rotation=30)
    failure_axis.set_ylabel("First boundary time (s)")
    fig.tight_layout(); fig.savefig(out/"tracking_and_failures.png", dpi=150); plt.close(fig)

    fig, (axis, distortion) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    found = False
    for method in methods:
        file = Path(run)/"frequency"/"confirm"/method/"frequency.npz"
        if file.exists():
            with np.load(file) as data:
                if bool(data["valid"]):
                    axis.plot(data["omega"], data["singular_values"][:, 0], marker="o", label=method)
                    distortion.plot(data["omega"], np.sum(data["nonexcited_energy"], axis=1),
                                    marker="o", label=method)
                    found = True
    if found:
        axis.legend(); axis.set_ylabel("Diagnostic max singular value")
        distortion.set_ylabel("Nonexcited energy")
    else:
        axis.text(.5, .5, "Frequency confirmation unavailable", ha="center", va="center",
                  transform=axis.transAxes)
    distortion.set_xlabel("rad/s")
    fig.tight_layout(); fig.savefig(out/"frequency.png", dpi=150); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4))
    representative = next((r for r in rows if r["method"] in methods and
                           _episode_path(run, split, r["case_id"], r["method"]).exists()), None)
    if representative:
        file = _episode_path(run, split, representative["case_id"], representative["method"])/"control.npz"
        with np.load(file) as data:
            for j, label in enumerate(("theta", "u", "vertical", "lateral")):
                axis.plot(data["t"], data["alpha"][:, j], label=label)
            axis.plot(data["t"], np.linalg.norm(data["d_raw"], axis=1),
                      color="black", alpha=.5, label="innovation norm")
            actual = data["actual"]/[1_500_000., np.pi/4, np.pi/4]
            margin = np.min(np.minimum(actual-[0., -1., -1.],
                                       [1., 1., 1.]-actual), axis=1)
            axis.plot(data["t"], margin, color="tab:gray", alpha=.6,
                      label="minimum actuator margin")
        axis.legend(fontsize=8); axis.set_xlabel("s")
    else:
        axis.text(.5, .5, "Gate trace unavailable", ha="center", va="center",
                  transform=axis.transAxes)
    fig.tight_layout(); fig.savefig(out/"gate_and_innovation.png", dpi=150); plt.close(fig)


def _seed_cluster_interval(rows, methods, run):
    policies = [method for method in methods if method.startswith("ppo_")]
    baselines = [method for method in ("fixed_best", "rule_best") if method in methods]
    if len(policies) < 2 or not baselines:
        return None
    cases = sorted(set(row["case_id"] for row in rows))
    lookup = {(row["case_id"], row["method"]): row for row in rows}
    cases = [case for case in cases if all(lookup[(case, policy)]["common_prefix_rmse"] is not None
             for policy in policies+baselines)]
    if not cases: return None
    selection = read_json(Path(run)/"selected_baselines.json")
    baseline = min(baselines, key=lambda name: tuple(selection["scores"][selection[name]]))
    differences = np.array([[lookup[(case, policy)]["common_prefix_rmse"]-
                             lookup[(case, baseline)]["common_prefix_rmse"]
                             for case in cases] for policy in policies])
    generator = np.random.default_rng(41001)
    draws = []
    for _ in range(2000):
        seed_indices = generator.integers(len(policies), size=len(policies))
        case_indices = generator.integers(len(cases), size=len(cases))
        draws.append(float(np.mean(differences[np.ix_(seed_indices, case_indices)])))
    return {"baseline": baseline,
            "per_seed_mean_delta": {policy: float(np.mean(differences[i]))
                                    for i, policy in enumerate(policies)},
            "mean_delta": float(np.mean(differences)),
            "bootstrap_95_interval": np.quantile(draws, [.025, .975]).tolist(),
            "resampling": "training seed clusters and paired scenario indices"}


def _decision(rows, methods, run, stage):
    selection_file = Path(run)/"selected_baselines.json"
    if not selection_file.exists():
        return "基线尚未筛选，不能判定 RL 增益。"
    selection = read_json(selection_file)
    baseline = min(("fixed_best", "rule_best"),
                   key=lambda name: tuple(selection["scores"][selection[name]]))
    feasible = [row for row in rows if not row["capacity_limited"]]
    base = [row for row in feasible if row["method"] == baseline]
    policies = [name for name in methods if name == "ppo" or name.startswith("ppo_")]
    if not base or not policies:
        return "没有同场景、同信息的 RL 与强基线配对结果。"
    baseline_cost = _mean_or_nan([row["tracking_cost_integral"] for row in base])
    baseline_failures = sum(not row["complete"] for row in base)
    policy_costs, policy_failures = [], []
    for name in policies:
        subset = [row for row in feasible if row["method"] == name]
        policy_costs.append(_mean_or_nan([row["tracking_cost_integral"] for row in subset]))
        policy_failures.append(sum(not row["complete"] for row in subset))
    improvement = 1.-float(np.mean(policy_costs))/max(baseline_cost, 1e-12)
    no_extra_failures = all(value <= baseline_failures for value in policy_failures)
    if improvement >= .05 and no_extra_failures:
        return (f"可行场景上相对开发集选出的 `{baseline}`，平均跟踪代价改善 "
                f"{improvement:.1%}，且观察到的失败数未增加；这是晋级信号，仍需同时核对频域和跨 seed 结果。")
    if stage == "pilot":
        return (f"首轮未达到约 5% 且不增失败的晋级条件（代价变化 {improvement:.1%}）。"
                "协议最多允许一次明确修正；当前不声称 RL 有额外价值。")
    return (f"确认阶段未达到晋级条件（代价变化 {improvement:.1%}）；"
            "保留开发集选出的固定/规则门控，不继续扩大 RL 训练。")


def _ablation_intervals(rows, methods):
    variants = ("ppo", "tied_residual", "no_history")
    seed_sets = []
    for variant in variants:
        prefix = variant+"_"
        seed_sets.append({name[len(prefix):] for name in methods if name.startswith(prefix)})
    seeds = sorted(set.intersection(*seed_sets))
    if len(seeds) < 2:
        return None
    lookup = {(row["case_id"], row["method"]): row for row in rows}
    cases = sorted({row["case_id"] for row in rows})
    cases = [case for case in cases if all(
        lookup.get((case, f"{variant}_{seed}"), {}).get("common_prefix_rmse") is not None
        for variant in variants for seed in seeds)]
    if not cases:
        return None
    matrices = {variant: np.array([[lookup[(case, f"{variant}_{seed}")]["common_prefix_rmse"]
                                    for case in cases] for seed in seeds])
                for variant in variants}
    generator = np.random.default_rng(42001)
    results = {}
    for variant in variants[1:]:
        delta = matrices[variant]-matrices["ppo"]
        draws = []
        for _ in range(2000):
            seed_index = generator.integers(len(seeds), size=len(seeds))
            case_index = generator.integers(len(cases), size=len(cases))
            draws.append(float(np.mean(delta[np.ix_(seed_index, case_index)])))
        results[variant] = {"mean_delta": float(np.mean(delta)),
                            "bootstrap_95_interval": np.quantile(draws, [.025, .975]).tolist(),
                            "per_seed_delta": {seed: float(np.mean(delta[i]))
                                               for i, seed in enumerate(seeds)}}
    return {"seeds": seeds, "paired_cases": len(cases), "results": results}


def report(run, stage="pilot", tag=None, split=None):
    default_split = "dev_quick" if stage == "pilot" else "test"
    split = split or default_split
    output_split = split if tag is None else f"{split}_{tag}"
    evaluation = read_json(Path(run)/f"evaluation_{output_split}.json")
    methods = evaluation["methods"]
    rows = _paired_rows(run, split, methods, output_split)
    folder = stage if split == default_split else f"{stage}_{split}"
    if tag is not None:
        folder += f"_{tag}"
    out = Path(run)/"reports"/folder
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    with (out/"paired_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    frequency_files = list((Path(run)/"frequency"/"confirm").glob("*/frequency.npz"))
    if frequency_files:
        arrays = {}
        for file in frequency_files:
            with np.load(file) as data:
                for key in data.files:
                    arrays[f"{file.parent.name}_{key}"] = data[key]
        write_npz(out/"frequency_metrics.npz", **arrays)
    else:
        write_npz(out/"frequency_metrics.npz", valid=np.asarray(False))
    _charts(out, rows, run, output_split, methods)
    manifest = read_json(Path(run)/"run_manifest.json")
    total_control = sum(item["control_steps"] for item in evaluation["results"])
    total_decisions = sum(item["decisions"] for item in evaluation["results"])
    total_substeps = sum(item["substeps"] for item in evaluation["results"])
    total_wall = sum(item["wall_s"] for item in evaluation["results"])
    missing_prefix = sum(row["common_prefix_rmse"] is None for row in rows)
    capacity_cases = len({row["case_id"] for row in rows if row["capacity_limited"]})
    lines = [f"# GS-12 {stage} report", "",
             f"Protocol: `{manifest['protocol']}`. Evaluation split: `{split}` (tag: `{tag}`).",
             f"Paired cases: {len(rows)//len(methods)}. Methods: {', '.join(methods)}.", "",
             f"Executed: {total_decisions} gate decisions, {total_control} MPC periods, "
             f"{total_substeps} plant substeps; episode wall time {total_wall:.1f} s.",
             f"Capacity-limited cases: {capacity_cases}; missing common-prefix RMSE rows: {missing_prefix}.", "",
             "## Paired results", "",
             "| Method | Failures | Common-prefix RMSE | Mean cost integral | QP failures |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        lines.append(f"| {method} | {sum(not r['complete'] for r in subset)} | "
                      f"{_mean_or_nan([r['common_prefix_rmse'] for r in subset]):.5f} | "
                     f"{np.mean([r['tracking_cost_integral'] for r in subset]):.5f} | "
                     f"{sum(r['qp_failures'] for r in subset)} |")
    interval = _seed_cluster_interval(rows, methods, run)
    if interval is not None:
        lines.extend(["", "## Across training seeds", "",
                      f"Paired RMSE difference versus `{interval['baseline']}`: "
                      f"{interval['mean_delta']:.5f}; clustered 95% resampling interval "
                      f"[{interval['bootstrap_95_interval'][0]:.5f}, "
                      f"{interval['bootstrap_95_interval'][1]:.5f}].",
                      f"Per-seed deltas: `{json.dumps(interval['per_seed_mean_delta'])}`.",
                      "The five training seeds do not guarantee precise inference."])
    attribution = _ablation_intervals(rows, methods)
    if attribution is not None:
        lines.extend(["", "## Retrained ablations", "",
                      f"Paired seeds: {', '.join(attribution['seeds'])}; "
                      f"paired cases: {attribution['paired_cases']}."])
        for variant, result in attribution["results"].items():
            lower, upper = result["bootstrap_95_interval"]
            lines.append(f"- `{variant}` minus standard PPO normalized RMSE: "
                         f"{result['mean_delta']:.5f} "
                         f"(clustered 95% interval [{lower:.5f}, {upper:.5f}]); "
                         f"per-seed deltas `{json.dumps(result['per_seed_delta'])}`.")
    lines.extend(["", "## Decision", "", _decision(rows, methods, run, stage), "",
                  "The compared gates share the same prediction structure and causal estimator. "
                  "Their paired difference can support a gate-decision claim only; this report "
                  "does not assign the hybrid-model or estimator benefit to RL."])
    if any(method.startswith("ppo_") for method in methods):
        lines.extend(["", "## Training seeds and worst observed cases", ""])
        for method in (name for name in methods if name.startswith("ppo_")):
            subset = [row for row in rows if row["method"] == method]
            finite = [row for row in subset if row["common_prefix_rmse"] is not None]
            worst = max(finite, key=lambda row: row["common_prefix_rmse"]) if finite else None
            lines.append(f"- `{method}`: {sum(not row['complete'] for row in subset)} failures; "
                         f"worst observed `{worst['case_id']}` "
                         f"(RMSE {worst['common_prefix_rmse']:.5f})." if worst else
                         f"- `{method}`: no finite paired prefix.")
    lines.extend(["", "Common-prefix RMSE uses each case's shortest observed duration across methods.",
                  "Failures and capacity-limited cases remain in the table.",
                  "Dynamic-gate frequency matrices assembled from separate axes are diagnostic empirical responses.",
                  "They are not a single LTI MIMO FRF or a stability guarantee.", "",
                  "## Artifacts", "",
                  "- `paired_metrics.csv`: all paired case-method metrics.",
                  "- `frequency_metrics.npz`: available complex-response diagnostics.",
                  "- `families.png`, `tracking_and_failures.png`, `frequency.png`, `gate_and_innovation.png`."])
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"report": str(out/"report.md"), "paired_rows": len(rows),
            "frequency_files": len(frequency_files)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "final"), required=True)
    parser.add_argument("--tag")
    parser.add_argument("--split", choices=("dev_quick", "test", "ablation_holdout"))
    args = parser.parse_args()
    print(json.dumps(report(args.run, args.stage, args.tag, args.split), ensure_ascii=False))


if __name__ == "__main__":
    main()
