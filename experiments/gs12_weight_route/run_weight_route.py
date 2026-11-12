#!/usr/bin/env python3
"""Prepare and execute the frozen GS-12 weight route."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get("GS12_WEIGHT_OUT", HERE/"output")).resolve()
WEIGHTS = [(d, r) for d in (.7, 1., 1.4) for r in (.7, 1., 1.4)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_list():
    tasks = []
    for k, phase in ((8, 0.), (32, .73)):
        tasks.append({"task_id": f"scan_depth_k{k}", "split": "dev_scan", "kind": "single",
                      "axis": "depth", "k": k, "phase_rad": phase, "amplitude": .006,
                      "duration_s": 64., "speed_mps": 15., "window_s": [32., 64.]})
    for i, (phase, sw) in enumerate(((.21, (20., 40.)), (1.17, (22., 42.))), 1):
        tasks.append({"task_id": f"mixed_dev_{i}", "split": "dev_rule", "kind": "mixed",
                      "axis": "east_depth", "phase_rad": phase, "switch_times_s": list(sw),
                      "amplitude": .006, "duration_s": 64., "speed_mps": 15.,
                      "window_s": [32., 64.]})
    tasks += [
        {"task_id": "locked_depth_k6", "split": "test_locked", "kind": "single",
         "axis": "depth", "k": 6, "phase_rad": 2.04, "amplitude": .006,
         "duration_s": 64., "speed_mps": 15., "window_s": [32., 64.]},
        {"task_id": "locked_east_k30", "split": "test_locked", "kind": "single",
         "axis": "east", "k": 30, "phase_rad": -1.31, "amplitude": .006,
         "duration_s": 64., "speed_mps": 15., "window_s": [32., 64.]},
        {"task_id": "locked_mixed", "split": "test_locked", "kind": "mixed",
         "axis": "east_depth", "phase_rad": 2.61, "switch_times_s": [18., 39.],
         "amplitude": .006, "duration_s": 64., "speed_mps": 15.,
         "window_s": [32., 64.]},
    ]
    return tasks


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"runs").mkdir(exist_ok=True); (OUT/"traces").mkdir(exist_ok=True)
    tasks = task_list()
    files = [ROOT/"aav_mpc/condensed_mpc.py", ROOT/"simulator/batch_mpc.py",
             ROOT/"simulator/rti_symbolic.py", ROOT/"simulator/model.py",
             ROOT/"experiments/gs12_analaysis/frequency_extension_src/plant_harness.py",
             ROOT/"experiments/gs12_analaysis/frequency_extension_src/run_frequency_extension.py",
             HERE/"reference_tasks.py", HERE/"runner_core.py", HERE/"run_weight_route.py",
             HERE/"verify_route.py", HERE/"analyze_weight_route.py"]
    try: commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = None
    packages = {}
    for name in ("numpy", "torch", "scipy", "osqp", "casadi"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    protocol = {
        "id": "GS12-WEIGHT-ROUTE-2026-11-11", "version": 1,
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repo_commit": commit,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "plant": {"speed_mps": 15., "config": {"elevator": .9, "rudder": .9,
                   "order": "second_all"}, "true_plant": "C4", "dt_s": .1,
                   "substep_s": .02, "horizon": 35},
        "controllers": {"C4": {"model_gain": 1., "model_tau": 1.},
                         "C5": {"model_gain": .85, "model_tau": 2.}},
        "weight_grid": WEIGHTS, "tasks": tasks,
        "objective": "mean((Eref-E)^2/8^2 + (Dref-D)^2/5^2 + (VNref-VN)^2/2^2)",
        "primary_window_s": [32., 64.], "diagnostic_limits": [8., 5., 2., 0.6981317007977318],
        "selection_rule": "dev_scan equal task-weighted relative J_track; exclude new early stop or QP failure",
        "locked_test_rule": "mean J improvement >=5%, no task worse than C4, no added failures",
        "note": "test_locked tasks frozen before scan; no locked run until explicit selection freeze"
    }
    (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    (OUT/"tasks.jsonl").write_text("".join(json.dumps(t, sort_keys=True)+"\n" for t in tasks))
    print(json.dumps({"tasks": len(tasks), "protocol_sha256": sha(OUT/"protocol.json"), "out": str(OUT)}))


def _job(task, controller, depth, rate, stage):
    return {"task": task, "controller": controller, "a_depth": depth,
            "a_rudder_rate": rate, "stage": stage}


def _primary_key(job, protocol_digest):
    task = job["task"]
    amplitude = float(task.get("amplitude", .006 if task["kind"] in ("single", "mixed", "band") else 0.))
    phase_id = task.get("phase_id")
    if phase_id is None:
        phase_id = f"phase_{float(task['phase_rad']):g}" if "phase_rad" in task else "zero"
    split_id = job["stage"] if job["stage"].startswith("pilot_") else task.get("split", job["stage"])
    return [protocol_digest, split_id, task["task_id"], job["controller"],
            f"{job['a_depth']:g}_{job['a_rudder_rate']:g}", amplitude, phase_id]


def assert_unique_jobs(jobs, protocol_digest):
    keys = [json.dumps(_primary_key(job, protocol_digest), separators=(",", ":")) for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate frozen run primary key in batch")


def jobs_for(stage):
    tasks = task_list()
    if stage == "dev_scan":
        tasks = [t for t in tasks if t["split"] == "dev_scan"]
        jobs = [_job(t, "C4", d, r, stage) for t in tasks for d, r in WEIGHTS]
        jobs += [_job(t, "C5", 1., 1., stage) for t in tasks]
    elif stage in ("dev_rule", "test_locked"):
        split = stage
        tasks = [t for t in tasks if t["split"] == split]
        # Candidate parameters are loaded from a frozen selection.
        selection = json.loads((OUT/"selection.json").read_text())
        policies = [("C4", 1., 1.), ("C5", 1., 1.)]
        policies += [("C4", *p) for p in selection["candidates"]]
        if stage == "dev_rule": policies = policies[:4]
        jobs = [_job(t, c, d, r, stage) for t in tasks for c, d, r in policies]
    elif stage == "long_matrix":
        selection = json.loads((OUT/"selection.json").read_text())
        if selection.get("locked_test_status") != "pass":
            raise RuntimeError("long protocol requires a passing locked test")
        if len(selection.get("candidates", [])) != 1:
            raise RuntimeError("long protocol expects one frozen static candidate")
        d, rate = selection["candidates"][0]
        controllers = [("C4", 1., 1.), ("C5", 1., 1.), ("C4", d, rate)]
        jobs = []
        for band in ("B1", "B4"):
            for axis in ("east", "depth", "speed"):
                for amplitude in (.006, .003):
                    task = {"task_id": f"{band}_{axis}_a{amplitude:g}", "split": "long_matrix",
                            "kind": "band", "band": band, "axis": axis, "amplitude": amplitude,
                            "phase_id": "primary", "duration_s": 256., "speed_mps": 15.}
                    jobs += [_job(task, c, wd, wr, stage) for c, wd, wr in controllers]
        jobs += [_job({"task_id": f"zero_{c}", "split": "long_matrix", "kind": "zero",
                       "axis": "zero", "duration_s": 256., "speed_mps": 15.}, c, 1., 1., stage)
                 for c in ("C4", "C5", "C4_candidate")]
        # The predictor and policy are separate: identify the third zero run unambiguously.
        for j in jobs:
            if j["task"]["task_id"] == "zero_C4_candidate":
                j["controller"] = "C4"; j["a_depth"], j["a_rudder_rate"] = d, rate
    else:
        raise ValueError(stage)
    return jobs


def _execute(job, output_dir=OUT):
    # Set numerical-library threads before importing the model stack.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(HERE))
    from runner_core import run_one
    result = run_one(job["task"], job["controller"], job["a_depth"], job["a_rudder_rate"],
                     job["task"].get("duration_s", 64.))
    arrays, meta = result["arrays"], result["meta"]
    protocol_path = OUT/"long_protocol.json" if job["stage"] == "long_matrix" else OUT/"protocol.json"
    protocol_digest = sha(protocol_path)
    task = job["task"]
    primary_key = _primary_key(job, protocol_digest)
    split_id = primary_key[1]; amplitude = primary_key[5]; phase_id = primary_key[6]
    weight_policy = f"{job['a_depth']:g}_{job['a_rudder_rate']:g}"
    run_id = hashlib.sha256(json.dumps(primary_key, separators=(",", ":")).encode()).hexdigest()[:16]
    (output_dir/"traces").mkdir(parents=True, exist_ok=True)
    (output_dir/"runs").mkdir(parents=True, exist_ok=True)
    trace = output_dir/"traces"/(run_id+".npz")
    record = output_dir/"runs"/(run_id+".json")
    if trace.exists() or record.exists():
        if not (trace.exists() and record.exists()):
            raise FileExistsError(f"partial existing run {run_id}")
        old = json.loads(record.read_text())
        if (old.get("protocol_sha256") != protocol_digest or
                old.get("trace_sha256") != sha(trace) or
                old.get("source_sha256") != sha(HERE/"runner_core.py")):
            raise FileExistsError(f"existing run failed integrity checks: {run_id}")
        return old
    np.savez_compressed(trace, **arrays)
    rec = {"run_id": run_id, "task_id": job["task"]["task_id"],
           "split": split_id, "stage": job["stage"],
           "plant": "C4", "controller": job["controller"],
           "weight_policy": weight_policy,
           "a_depth": job["a_depth"], "a_rudder_rate": job["a_rudder_rate"],
           "amplitude": amplitude, "phase_id": phase_id,
           "protocol_sha256": protocol_digest, "source_sha256": sha(HERE/"runner_core.py"),
           "status": meta["status"], "completion": meta["duration_observed_s"],
           "first_violation": meta["first_violation"], "qp_failures": meta["qp_failures"],
           "command_saturation_fraction": meta["command_saturation_fraction"],
           "actual_saturation_fraction": meta["actual_saturation_fraction"],
           "solve_p50_ms": meta["solve_p50_ms"], "solve_p95_ms": meta["solve_p95_ms"],
           "solve_p99_ms": meta["solve_p99_ms"], "wall_s": meta["wall_s"],
           "worker_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
           "trace_sha256": sha(trace)}
    record.write_text(json.dumps(rec, indent=2)+"\n")
    return rec


def run_stage(stage, workers=1):
    jobs = jobs_for(stage)
    assert_unique_jobs(jobs, sha(OUT/"protocol.json"))
    print(json.dumps({"stage": stage, "jobs": len(jobs), "workers": workers,
                      "expected_max_s": len(jobs)*64/ max(workers, 1)}), flush=True)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_execute, j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            rec = f.result(); results.append(rec)
            print(json.dumps({"run_id": rec["run_id"], "task": rec["task_id"],
                              "policy": rec["weight_policy"], "controller": rec["controller"],
                              "status": rec["status"], "wall_s": rec["wall_s"]}), flush=True)
    manifest_path = OUT/"manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    known = {x["run_id"] for x in manifest["runs"]}
    manifest["runs"] += [{"run_id": r["run_id"], "status": r["status"], "stage": stage,
                          "trace_sha256": r["trace_sha256"]} for r in results if r["run_id"] not in known]
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"stage": stage, "completed": len(results), "manifest": str(manifest_path)}))


def run_pilot():
    tasks = [
        {"task_id": "pilot_depth_k8", "split": "pilot", "kind": "single", "axis": "depth", "k": 8, "phase_rad": .1},
        {"task_id": "pilot_depth_k32", "split": "pilot", "kind": "single", "axis": "depth", "k": 32, "phase_rad": .4},
        {"task_id": "pilot_east_k8", "split": "pilot", "kind": "single", "axis": "east", "k": 8, "phase_rad": .7},
        {"task_id": "pilot_east_k32", "split": "pilot", "kind": "single", "axis": "east", "k": 32, "phase_rad": 1.},
    ]
    summary = {"same_four_tasks": [t["task_id"] for t in tasks], "groups": []}
    for workers in (1, 2, 4):
        out = OUT/f"pilot_w{workers}"
        jobs = [_job(t, "C4", 1., 1., f"pilot_w{workers}") for t in tasks]
        assert_unique_jobs(jobs, sha(OUT/"protocol.json"))
        started = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_execute, j, out) for j in jobs]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed = time.perf_counter()-started
        group = {"workers": workers, "tasks": len(results), "batch_wall_s": elapsed,
                 "throughput_runs_per_s": len(results)/elapsed,
                 "peak_worker_rss_kb": max(r["worker_peak_rss_kb"] for r in results),
                 "statuses": {s: sum(r["status"] == s for r in results) for s in sorted({r["status"] for r in results})}}
        summary["groups"].append(group)
        print(json.dumps(group), flush=True)
    selected = max(summary["groups"], key=lambda g: g["throughput_runs_per_s"])
    summary["selected_workers"] = selected["workers"]
    (OUT/"pilot_throughput.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def run_long_matrix(workers=4):
    jobs = jobs_for("long_matrix")
    protocol = {"id": "GS12-WEIGHT-LONG-B1B4-2026-11-11", "base_protocol_sha256": sha(OUT/"protocol.json"),
                "selection_sha256": sha(OUT/"selection.json"), "jobs": [{"task_id": j["task"]["task_id"],
                "controller": j["controller"], "a_depth": j["a_depth"],
                "a_rudder_rate": j["a_rudder_rate"]} for j in jobs],
                "duration_s": 256., "windows_s": [[64., 128.], [128., 192.], [192., 256.]],
                "matrix": "three normalized outputs by three normalized reference inputs",
                "amplitudes_per_line": [.006, .003], "bands": {"B1": list(range(1, 9)), "B4": list(range(25, 33))},
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                    [ROOT/"aav_mpc/condensed_mpc.py", ROOT/"simulator/batch_mpc.py",
                     ROOT/"simulator/rti_symbolic.py", ROOT/"simulator/model.py",
                     ROOT/"experiments/gs12_analaysis/frequency_extension_src/plant_harness.py",
                     HERE/"reference_tasks.py", HERE/"runner_core.py", HERE/"run_weight_route.py",
                     HERE/"analyze_weight_route.py", HERE/"analyze_long_matrix.py"]},
                "controller_plant_contract": "same C4 true plant for C4, C5 and frozen static candidate"}
    (OUT/"long_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    assert_unique_jobs(jobs, sha(OUT/"long_protocol.json"))
    out = OUT/"long_matrix"; out.mkdir(exist_ok=True)
    # The first 256 s task is both a throughput pilot and a member of the frozen 39-run matrix.
    started = time.perf_counter()
    first = _execute(jobs[0], out)
    first["batch_wall_s"] = time.perf_counter()-started
    (out/"long_throughput_pilot.json").write_text(json.dumps(first, indent=2)+"\n")
    remaining = jobs[1:]
    print(json.dumps({"stage": "long_matrix", "pilot_wall_s": first["batch_wall_s"],
                      "remaining_jobs": len(remaining), "workers": workers,
                      "estimated_remaining_s": len(remaining)*first["batch_wall_s"]/workers}), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_execute, j, out) for j in remaining]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        for rec in results:
            print(json.dumps({"run_id": rec["run_id"], "task": rec["task_id"],
                              "controller": rec["controller"], "policy": rec["weight_policy"],
                              "status": rec["status"], "wall_s": rec["wall_s"]}), flush=True)
    all_results = [first, *results]
    (out/"manifest.json").write_text(json.dumps({"runs": [{"run_id": r["run_id"],
        "task_id": r["task_id"], "status": r["status"], "trace_sha256": r["trace_sha256"]}
        for r in all_results]}, indent=2)+"\n")
    print(json.dumps({"stage": "long_matrix", "completed": len(all_results), "out": str(out)}))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("pilot")
    r = sub.add_parser("run"); r.add_argument("stage", choices=("dev_scan", "dev_rule", "test_locked", "long_matrix")); r.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if a.cmd == "prepare": prepare()
    elif a.cmd == "pilot": run_pilot()
    elif a.stage == "long_matrix": run_long_matrix(a.workers)
    else: run_stage(a.stage, a.workers)


if __name__ == "__main__": main()
