#!/usr/bin/env python3
"""Paired external-objective analysis for the GS-12 weight route."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(__import__("os").environ.get("GS12_WEIGHT_OUT", HERE/"output")).resolve()
ERROR_SCALE = np.array([8., 5., 2.])


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records():
    result = []
    for path in sorted((OUT/"runs").glob("*.json")):
        rec = json.loads(path.read_text())
        trace = OUT/"traces"/(rec["run_id"]+".npz")
        if trace.exists() and sha(trace) == rec.get("trace_sha256"):
            result.append((rec, trace))
    return result


def summarize(rec, trace, common_end=None):
    d = np.load(trace)
    time_s = d["time_substep_s"]
    end = min(float(rec["completion"]), common_end or float("inf"))
    mask = (time_s >= 32.-1e-9) & (time_s <= min(64., end)+1e-9)
    if not np.any(mask):
        return {"J_track": None, "rmse_E": None, "rmse_D": None, "rmse_V": None,
                "window_start_s": None, "window_end_s": None}
    ref, output = d["reference_substep"][mask], d["output_substep"][mask]
    err = np.column_stack((ref[:, 1]-output[:, 1], ref[:, 2]-output[:, 2],
                           ref[:, 5]-output[:, 5]))
    norm = err/ERROR_SCALE
    return {"J_track": float(np.mean(np.sum(norm**2, axis=1))),
            "rmse_E": float(np.sqrt(np.mean(err[:, 0]**2))),
            "rmse_D": float(np.sqrt(np.mean(err[:, 1]**2))),
            "rmse_V": float(np.sqrt(np.mean(err[:, 2]**2))),
            "peak_E": float(np.max(np.abs(err[:, 0]))),
            "peak_D": float(np.max(np.abs(err[:, 1]))),
            "peak_V": float(np.max(np.abs(err[:, 2]))),
            "window_start_s": float(time_s[mask][0]),
            "window_end_s": float(time_s[mask][-1]),
            "max_command_change_normalized": float(np.max(np.abs(np.diff(d["command"]/np.array([1500000., .7853981633974483, .7853981633974483]), axis=0)))) if len(d["command"]) > 1 else 0.}


def analyze(stage):
    all_records = [(r, t) for r, t in records() if r["stage"] == stage]
    by_task = {}
    for r, t in all_records: by_task.setdefault(r["task_id"], []).append((r, t))
    rows = []
    for task_id, methods in sorted(by_task.items()):
        common_end = min(float(r["completion"]) for r, _ in methods)
        for rec, trace in methods:
            metrics = summarize(rec, trace, common_end)
            row = {k: rec.get(k) for k in ("run_id", "task_id", "split", "plant", "controller",
                   "weight_policy", "a_depth", "a_rudder_rate", "status", "completion",
                   "qp_failures", "command_saturation_fraction", "actual_saturation_fraction",
                   "solve_p95_ms", "wall_s", "trace_sha256")}
            row.update(metrics); row["common_prefix_end_s"] = common_end
            rows.append(row)
    fields = sorted({k for row in rows for k in row})
    with (OUT/f"{stage}_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    expected_tasks = [t["task_id"] for t in json.loads((OUT/"protocol.json").read_text())["tasks"]
                      if (t["split"] == stage or (stage == "dev_scan" and t["split"] == "dev_scan"))]
    missing = [t for t in expected_tasks if t not in by_task]
    return rows, missing


def choose_scan(rows, missing):
    scan = [r for r in rows if r["controller"] == "C4"]
    baseline = {r["task_id"]: r for r in scan if r["a_depth"] == 1. and r["a_rudder_rate"] == 1.}
    policies = sorted({(r["a_depth"], r["a_rudder_rate"]) for r in scan})
    usable = []
    for depth, rate in policies:
        candidates = [r for r in scan if r["a_depth"] == depth and r["a_rudder_rate"] == rate]
        if len(candidates) != 2 or any(r["J_track"] is None or r["status"].startswith("early") or r["qp_failures"] for r in candidates):
            continue
        if any(r["task_id"] not in baseline or baseline[r["task_id"]]["J_track"] is None for r in candidates):
            continue
        improvements = [(baseline[r["task_id"]]["J_track"]-r["J_track"])/baseline[r["task_id"]]["J_track"] for r in candidates]
        usable.append({"a_depth": depth, "a_rudder_rate": rate,
                       "mean_task_relative_improvement": float(np.mean(improvements)),
                       "per_task_relative_improvement": {r["task_id"]: float(v) for r, v in zip(candidates, improvements)}})
    usable.sort(key=lambda x: x["mean_task_relative_improvement"], reverse=True)
    positive = [x for x in usable if x["mean_task_relative_improvement"] > 0.]
    selected = []
    if positive:
        selected.append(positive[0])
        best_by_task = {}
        for task_id in baseline:
            options = [r for r in scan if r["task_id"] == task_id and r["J_track"] is not None and
                       not r["status"].startswith("early") and not r["qp_failures"]]
            best_by_task[task_id] = min(options, key=lambda r: r["J_track"])
        endpoints = {(r["a_depth"], r["a_rudder_rate"]) for r in best_by_task.values()}
        for policy in endpoints:
            if policy != (selected[0]["a_depth"], selected[0]["a_rudder_rate"]):
                item = next((x for x in positive if (x["a_depth"], x["a_rudder_rate"]) == policy), None)
                if item and len(selected) < 2: selected.append(item)
    selection = {"basis": "dev_scan", "missing_tasks": missing,
                 "policies_ranked": usable, "candidates": [[x["a_depth"], x["a_rudder_rate"]] for x in selected],
                 "decision": "candidate_selected" if selected else "no_positive_configuration",
                 "locked_test_status": "not_run"}
    (OUT/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    return selection


def finalize_candidates(scan_selection, rule_rows, missing):
    baseline = {r["task_id"]: r for r in rule_rows
                if r["controller"] == "C4" and r["weight_policy"] == "1_1"}
    approved = []
    checks = []
    for pair in scan_selection["candidates"]:
        d, rate = pair
        candidates = [r for r in rule_rows if r["controller"] == "C4" and
                      r["a_depth"] == d and r["a_rudder_rate"] == rate]
        relative = {}
        if len(candidates) == 2 and len(baseline) == 2 and not missing:
            for row in candidates:
                base = baseline.get(row["task_id"])
                if base and base["J_track"] and row["J_track"] is not None:
                    relative[row["task_id"]] = (base["J_track"]-row["J_track"])/base["J_track"]
        ok = (len(relative) == 2 and all(v > 0 for v in relative.values()) and
              all(r["status"] == "complete_valid" and r["qp_failures"] == 0 for r in candidates))
        checks.append({"a_depth": d, "a_rudder_rate": rate,
                       "per_task_relative_improvement": relative,
                       "mean_task_relative_improvement": float(np.mean(list(relative.values()))) if relative else None,
                       "retained": bool(ok)})
        if ok: approved.append(pair)
    selection = dict(scan_selection)
    selection.update({"dev_rule_checks": checks, "candidates": approved,
                      "decision": "candidate_frozen" if approved else "no_candidate_survived_dev_rule",
                      "candidate_frozen_before_test_locked": True,
                      "locked_test_status": "not_run"})
    (OUT/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    return selection


def evaluate_locked(rows, missing, selection):
    candidates = selection["candidates"]
    results = []
    for d, rate in candidates:
        subset = [r for r in rows if r["controller"] == "C4" and
                  r["a_depth"] == d and r["a_rudder_rate"] == rate]
        baseline = {r["task_id"]: r for r in rows if r["controller"] == "C4" and
                    r["a_depth"] == 1. and r["a_rudder_rate"] == 1.}
        pairs = []
        for row in subset:
            base = baseline.get(row["task_id"])
            if base and base["J_track"] is not None and row["J_track"] is not None:
                pairs.append({"task_id": row["task_id"], "J_C4": base["J_track"],
                              "J_candidate": row["J_track"],
                              "relative_improvement": (base["J_track"]-row["J_track"])/base["J_track"],
                              "candidate_status": row["status"], "C4_status": base["status"],
                              "candidate_qp_failures": row["qp_failures"],
                              "solve_p95_ms": row["solve_p95_ms"]})
        mean_base = float(np.mean([p["J_C4"] for p in pairs])) if pairs else None
        mean_candidate = float(np.mean([p["J_candidate"] for p in pairs])) if pairs else None
        improvement = ((mean_base-mean_candidate)/mean_base) if mean_base else None
        no_worse = bool(pairs) and all(p["J_candidate"] <= p["J_C4"] for p in pairs)
        no_added_failures = bool(pairs) and all(
            p["candidate_status"] == p["C4_status"] and p["candidate_qp_failures"] == 0
            for p in pairs)
        passed = (not missing and len(pairs) == 3 and improvement is not None and
                  improvement >= .05 and no_worse and no_added_failures)
        results.append({"a_depth": d, "a_rudder_rate": rate, "task_pairs": pairs,
                        "mean_J_C4": mean_base, "mean_J_candidate": mean_candidate,
                        "relative_improvement_of_mean_J": improvement,
                        "all_tasks_nonworse": no_worse,
                        "no_added_failures": no_added_failures, "formal_gate_pass": passed})
    summary = {"compared_against": "C4 baseline", "expected_tasks": 3,
               "missing_tasks": missing, "candidates": results,
               "formal_engineering_gate": "mean J improvement >= 5%, each task nonworse, no added failures",
               "result": "pass" if any(x["formal_gate_pass"] for x in results) else "not_pass"}
    (OUT/"test_locked_decision.json").write_text(json.dumps(summary, indent=2)+"\n")
    selection["locked_test_status"] = summary["result"]
    selection["locked_test_decision_sha256"] = sha(OUT/"test_locked_decision.json")
    (OUT/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    return summary


def main():
    scan_rows, missing = analyze("dev_scan")
    selection = choose_scan(scan_rows, missing)
    print(json.dumps({"stage": "dev_scan", "rows": len(scan_rows), "missing_tasks": missing,
                      "decision": selection["decision"], "candidates": selection["candidates"]}))
    selection = json.loads((OUT/"selection.json").read_text())
    for stage in ("dev_rule", "test_locked"):
        if any(r["stage"] == stage for r, _ in records()):
            rows, missing = analyze(stage)
            print(json.dumps({"stage": stage, "rows": len(rows), "missing_tasks": missing}))
            if stage == "dev_rule":
                selection = finalize_candidates(selection, rows, missing)
                print(json.dumps({"candidate_decision": selection["decision"],
                                  "frozen_candidates": selection["candidates"]}))
            if stage == "test_locked":
                decision = evaluate_locked(rows, missing, selection)
                print(json.dumps({"locked_decision": decision["result"],
                                  "candidates": decision["candidates"]}))


if __name__ == "__main__": main()
