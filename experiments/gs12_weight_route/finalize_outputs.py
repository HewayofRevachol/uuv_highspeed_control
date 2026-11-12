#!/usr/bin/env python3
"""Check run/trace integrity and freeze a final file/source manifest."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(__import__("os").environ.get("GS12_WEIGHT_OUT", HERE/"output")).resolve()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def audit_run_dir(run_dir):
    entries = []
    errors = []
    for p in sorted(run_dir.glob("*.json")):
        if p.name == "long_throughput_pilot.json": continue
        try: rec = json.loads(p.read_text())
        except Exception as exc:
            errors.append({"file": p.name, "error": f"invalid JSON: {exc}"}); continue
        trace = run_dir.parent/"traces"/(rec.get("run_id", "")+".npz")
        if not trace.exists(): errors.append({"run_id": rec.get("run_id"), "error": "missing trace"}); continue
        digest = sha(trace)
        if digest != rec.get("trace_sha256"):
            errors.append({"run_id": rec.get("run_id"), "error": "trace SHA-256 mismatch"})
        primary = [rec.get("protocol_sha256"), rec.get("split"), rec.get("task_id"),
                   rec.get("controller"), rec.get("weight_policy"), float(rec.get("amplitude", 0.)),
                   rec.get("phase_id")]
        expected_id = hashlib.sha256(json.dumps(primary, separators=(",", ":")).encode()).hexdigest()[:16]
        if expected_id != rec.get("run_id"):
            errors.append({"run_id": rec.get("run_id"), "error": "run ID does not match protocol/task primary key"})
        entries.append({"run_id": rec.get("run_id"), "task_id": rec.get("task_id"),
                        "stage": rec.get("stage"), "status": rec.get("status"),
                        "trace_sha256": digest})
    return entries, errors


def main():
    all_entries, errors = [], []
    main_entries, e = audit_run_dir(OUT/"runs"); all_entries += main_entries; errors += e
    long_entries, e = audit_run_dir(OUT/"long_matrix"/"runs"); all_entries += long_entries; errors += e
    pilot_entries = []
    for workers in (1, 2, 4):
        entries, e = audit_run_dir(OUT/f"pilot_w{workers}"/"runs")
        pilot_entries += entries; errors += e
    expected = {"dev_scan": 20, "dev_rule": 8, "test_locked": 9, "long_matrix": 39}
    counts = {s: sum(x.get("stage") == s for x in all_entries) for s in expected}
    if counts != expected: errors.append({"expected_stage_counts": expected, "observed": counts})
    if any(x["status"] != "complete_valid" for x in all_entries):
        errors.append({"error": "one or more primary runs are not complete_valid"})
    run_ids = [x["run_id"] for x in all_entries+pilot_entries]
    if len(run_ids) != len(set(run_ids)):
        errors.append({"error": "duplicate run IDs across primary and long protocol outputs"})
    source_files = [
        "aav_mpc/condensed_mpc.py", "simulator/batch_mpc.py", "simulator/rti_symbolic.py",
        "simulator/model.py", "experiments/gs12_analaysis/frequency_extension_src/plant_harness.py",
        "experiments/gs12_analaysis/frequency_extension_src/run_frequency_extension.py",
    ] + [f"experiments/gs12_weight_route/{name}" for name in
         ("reference_tasks.py", "runner_core.py", "run_weight_route.py", "verify_route.py",
          "analyze_weight_route.py", "analyze_long_matrix.py", "plot_results.py",
          "finalize_outputs.py", "rekey_runs.py")]
    source_manifest = {p: sha(ROOT/p) for p in source_files}
    (OUT/"source_manifest.json").write_text(json.dumps({"files_sha256": source_manifest}, indent=2)+"\n")
    artifact_files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "integrity_audit.json"]
    result = {"status": "pass" if not errors else "fail", "stage_run_counts": counts,
              "total_primary_runs": len(all_entries), "total_pilot_runs": len(pilot_entries),
              "trace_integrity_errors": errors,
              "protocol_sha256": sha(OUT/"protocol.json"),
              "long_protocol_sha256": sha(OUT/"long_protocol.json"),
              "selection_sha256": sha(OUT/"selection.json"),
              "verified_runs": all_entries+pilot_entries,
              "artifacts_sha256": {str(p.relative_to(OUT)): sha(p) for p in sorted(artifact_files)}}
    (OUT/"integrity_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "counts": counts, "errors": errors}))
    if errors: raise SystemExit(1)


if __name__ == "__main__": main()
