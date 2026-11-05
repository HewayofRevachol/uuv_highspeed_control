"""Execute the prespecified E2 fixed-gate and compensation/predictor matrix."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from experiments.h15_cat_mpc.analyze_ppo import normalized_rmse, speed_mae
from experiments.h15_cat_mpc.residual_predictor import ResidualAR1
from experiments.h15_cat_mpc.run_episode import ROOT, run_episode, sha256


def execute(job):
    torch.set_num_threads(1)
    return run_episode(**job)


def physical_pair(candidate, baseline):
    both = bool(candidate["success"] and baseline["success"] and
                candidate["no_fallback_physical"] and baseline["no_fallback_physical"])
    cr = normalized_rmse(candidate) if both else None
    br = normalized_rmse(baseline) if both else None
    return {"physical_improved": not baseline["physical_strict"] and candidate["physical_strict"],
        "physical_worsened": baseline["physical_strict"] and not candidate["physical_strict"],
        "no_fallback_improved": not baseline["no_fallback_physical"] and candidate["no_fallback_physical"],
        "no_fallback_worsened": baseline["no_fallback_physical"] and not candidate["no_fallback_physical"],
        "deadline_improved": not baseline["deadline_clean"] and candidate["deadline_clean"],
        "deadline_worsened": baseline["deadline_clean"] and not candidate["deadline_clean"],
        "common_no_intervention_success": both,
        "candidate_R": cr, "baseline_R": br,
        "R_relative_change": (cr/br-1.) if both and br not in (None, 0.) else None,
        "R_absolute_change": (cr-br) if both and br == 0. else None,
        "candidate_speed_mae": speed_mae(candidate) if both else None,
        "baseline_speed_mae": speed_mae(baseline) if both else None,
        "candidate_periods": candidate["periods"], "baseline_periods": baseline["periods"]}


def offline_forecast_metrics(episode, predictor):
    rows = episode["completed_rows"]
    valid = [np.asarray(row["residual_body_acc"], dtype=float)
             if row.get("residual_valid") else None for row in rows]
    result = {}
    for horizon in (1, 5, 25):
        errors = []
        for start in range(len(rows)):
            target_i = start+horizon-1
            if target_i >= len(rows) or valid[target_i] is None:
                continue
            prior = [x for x in valid[:start] if x is not None]
            history = np.asarray(prior, dtype=float).reshape(-1, 5)
            forecast = predictor.predict(history, horizon)[horizon-1]
            errors.append(np.abs(valid[target_i]-forecast))
        result[str(horizon)] = {"count": len(errors),
            "mae": np.mean(errors, axis=0).tolist() if errors else None}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization-on", type=Path, required=True)
    parser.add_argument("--normalization-off", type=Path, required=True)
    parser.add_argument("--predictor-on", type=Path, required=True)
    parser.add_argument("--predictor-off", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for mode, norm_path, predictor_path in (
            ("on", args.normalization_on, args.predictor_on),
            ("off", args.normalization_off, args.predictor_off)):
        norm = json.loads(norm_path.read_text())
        pred = json.loads(predictor_path.read_text())
        if norm.get("split") != "train" or norm.get("compensation_mode") != mode:
            raise ValueError(f"{mode} normalization is not a matching train artifact")
        if pred.get("manifest", {}).get("compensation_mode") != mode:
            raise ValueError(f"{mode} predictor artifact is mismatched")
        if pred.get("manifest", {}).get("label_version") != "nominal_5x20ms_issued_v2":
            raise ValueError(f"{mode} predictor has an incompatible label version")
        artifacts[mode] = {"normalization_path": norm_path.resolve(),
            "calibration_path": None, "predictor": predictor_path.resolve()}
    cfg = json.loads(args.config.read_text())
    dev_manifest_path = ROOT/cfg["dev_manifest"]
    dev = json.loads(dev_manifest_path.read_text())
    units = [(case, int(seed)) for case in dev["cases"] for seed in dev["seeds"]]
    if len(units) != 6 or len({(c["id"], s) for c, s in units}) != 6:
        raise ValueError("E2 D_diag requires the frozen six geometry x seed units")
    jobs, reusable = [], {}

    def add(group, case, seed, method, mode, predictor_mode):
        predictor = (artifacts[mode]["predictor"]
                     if predictor_mode == "ar_or_persistence" else None)
        norm = artifacts[mode]["normalization_path"]
        stem = f"{case['id']}_s{seed}_{method}"
        key = (group, mode, predictor_mode, case["id"], seed, method)
        if group == "main":
            path = out/"main"/stem
        else:
            path = out/"ablation"/mode/predictor_mode/stem
        job = {"path": ROOT/case["path"], "seed": seed, "out": path,
            "method": method, "target_domain": True,
            "adapter_artifacts": {"normalization_path": norm,
                                   "calibration_path": None},
            "predictor_checkpoint": predictor,
            "predictor_mode": predictor_mode,
            "compensation_mode": mode}
        jobs.append(job)
        reusable[key] = path

    for case, seed in units:
        for method in ("A0", "A1", "A2", "A3", "A4"):
            add("main", case, seed, method, "on", "ar_or_persistence")
        for mode in ("on", "off"):
            for predictor_mode in (("zero", "persistence") if mode == "on"
                                   else ("zero", "persistence", "ar_or_persistence")):
                add("ablation", case, seed, "A4", mode, predictor_mode)
        add("ablation", case, seed, "A0", "off", "zero")
    if len(jobs) != 66:
        raise AssertionError(f"E2 matrix must contain exactly 66 episodes; got {len(jobs)}")
    protocol = {"schema": 1, "stage": "E2", "D_diag_manifest_sha256": sha256(dev_manifest_path),
        "units": [{"geometry": c["id"], "seed": s} for c, s in units],
        "episodes": 66, "main_fixed_gate_runs": 30,
        "additional_ablation_runs": 36,
        "fixed_gates": {"A0": [0., 0.], "A1": [1., 0.], "A2": [0., 1.],
                        "A3": [1., 1.], "A4": [.5, .5]},
        "compensation_modes": ["on", "off"],
        "predictors": ["zero", "persistence", "ar_or_persistence"],
        "holdout_used": False}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    run_record = {"schema": 1, "status": "running", "protocol": protocol,
        "workers": args.workers, "artifact_sha256": {mode: {
            "normalization": sha256(artifacts[mode]["normalization_path"]),
            "predictor": sha256(artifacts[mode]["predictor"])} for mode in artifacts}}
    (out/"run.json").write_text(json.dumps(run_record, indent=2)+"\n")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=mp.get_context("spawn")) as pool:
        list(pool.map(execute, jobs))

    episodes = []
    for job in jobs:
        episode_path = job["out"]/"episode.json"
        row = json.loads(episode_path.read_text())
        row["group"] = "main" if "main" in job["out"].parts else "ablation"
        episodes.append(row)
    keyed = {(row["group"], row["compensation_mode"], row["predictor_mode"],
              Path(row["path"]).stem, row["seed"], row["method"]): row for row in episodes}
    main_rows = [r for r in episodes if r["group"] == "main"]
    pairs = []
    for method in ("A1", "A2", "A3", "A4"):
        for case, seed in units:
            base = next(r for r in main_rows if Path(r["path"]).stem == Path(case["path"]).stem
                        and r["seed"] == seed and r["method"] == "A0")
            cand = next(r for r in main_rows if Path(r["path"]).stem == Path(case["path"]).stem
                        and r["seed"] == seed and r["method"] == method)
            pairs.append({"group": "main", "compensation_mode": "on",
                "predictor_mode": "ar_or_persistence", "geometry": case["id"],
                "seed": seed, "method": method, **physical_pair(cand, base)})
    ablations = []
    for mode, predictor_mode in (("on", "zero"), ("on", "persistence"),
            ("on", "ar_or_persistence"), ("off", "zero"),
            ("off", "persistence"), ("off", "ar_or_persistence")):
        for case, seed in units:
            if mode == "on" and predictor_mode == "ar_or_persistence":
                cand = next(r for r in main_rows if Path(r["path"]).stem == Path(case["path"]).stem
                            and r["seed"] == seed and r["method"] == "A4")
                base = next(r for r in main_rows if Path(r["path"]).stem == Path(case["path"]).stem
                            and r["seed"] == seed and r["method"] == "A0")
            else:
                candidates = [r for r in episodes if r["group"] == "ablation" and
                    r["compensation_mode"] == mode and r["predictor_mode"] == predictor_mode and
                    r["seed"] == seed and Path(r["path"]).stem == Path(case["path"]).stem]
                cand = next(r for r in candidates if r["method"] == "A4")
                base = (next(r for r in main_rows if Path(r["path"]).stem == Path(case["path"]).stem
                              and r["seed"] == seed and r["method"] == "A0") if mode == "on"
                        else next(r for r in episodes if r["group"] == "ablation" and
                                  r["compensation_mode"] == "off" and r["seed"] == seed and
                                  Path(r["path"]).stem == Path(case["path"]).stem and r["method"] == "A0"))
            ablations.append({"group": "ablation", "compensation_mode": mode,
                "predictor_mode": predictor_mode, "geometry": case["id"],
                "seed": seed, "method": "A4", **physical_pair(cand, base)})
    pairs_path = out/"fixed_gate_pairs.csv"
    with pairs_path.open("w", newline="") as stream:
        fields = list(pairs[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(pairs)
    ablation_path = out/"ablation_pairs.csv"
    with ablation_path.open("w", newline="") as stream:
        fields = list(ablations[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(ablations)
    summary = {"schema": 1, "protocol": protocol,
        "episodes_completed": len(episodes), "main_physical_strict": {
            m: sum(r["physical_strict"] for r in main_rows if r["method"] == m)
            for m in ("A0", "A1", "A2", "A3", "A4")},
        "main_no_fallback_physical": {
            m: sum(r["no_fallback_physical"] for r in main_rows if r["method"] == m)
            for m in ("A0", "A1", "A2", "A3", "A4")},
        "main_deadline_clean": {
            m: sum(r["deadline_clean"] for r in main_rows if r["method"] == m)
            for m in ("A0", "A1", "A2", "A3", "A4")},
        "main_pairs": pairs, "ablation_pairs": ablations,
        "pair_csv_sha256": sha256(pairs_path), "ablation_csv_sha256": sha256(ablation_path),
        "backup_supported": False, "backup_periods": None}
    offline = {}
    for mode in ("on", "off"):
        baseline_rows = ([r for r in main_rows if r["method"] == "A0"] if mode == "on"
            else [r for r in episodes if r["group"] == "ablation" and
                  r["compensation_mode"] == "off" and r["method"] == "A0"])
        mode_predictor = ResidualAR1.load(artifacts[mode]["predictor"],
            expected_label_version="nominal_5x20ms_issued_v2",
            expected_compensation_mode=mode)
        predictor_set = {"zero": ResidualAR1(np.zeros(5), np.zeros(5), "zero_v1"),
                         "persistence": ResidualAR1(version="persistence_v1"),
                         "ar": mode_predictor}
        per_episode = []
        for episode in baseline_rows:
            if episode["command_schema"] != "issued_v2":
                raise RuntimeError("offline predictor analysis refuses legacy command schema")
            per_episode.append({"geometry": Path(episode["path"]).stem,
                "seed": episode["seed"], "forecasts": {name:
                    offline_forecast_metrics(episode, predictor)
                    for name, predictor in predictor_set.items()}})
        offline[mode] = {"episodes": per_episode,
            "horizons": {str(h): {name: {
                "episodes": sum(1 for ep in per_episode
                    if ep["forecasts"][name][str(h)]["count"]),
                "count": sum(ep["forecasts"][name][str(h)]["count"] for ep in per_episode),
                "mae": (np.mean([ep["forecasts"][name][str(h)]["mae"]
                    for ep in per_episode if ep["forecasts"][name][str(h)]["mae"] is not None],
                    axis=0).tolist() if any(ep["forecasts"][name][str(h)]["mae"] is not None
                        for ep in per_episode) else None),
                "h25_count_at_least_50_each": (all(ep["forecasts"][name][str(h)]["count"] >= 50
                    for ep in per_episode) if h == 25 else None)}
                for name in predictor_set} for h in (1, 5, 25)}}
    summary["offline_predictor_metrics_same_A0_trace"] = offline
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    run_record.update({"status": "completed", "episodes_completed": len(episodes),
                       "summary_sha256": sha256(out/"summary.json")})
    (out/"run.json").write_text(json.dumps(run_record, indent=2)+"\n")
    print(json.dumps({"episodes_completed": len(episodes),
        "physical_strict": summary["main_physical_strict"],
        "output": str(out.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
