#!/usr/bin/env python3
"""Migrate frozen run artifacts to IDs that include protocol and task identity."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(__import__("os").environ.get("GS12_WEIGHT_OUT", HERE/"output")).resolve()


def run_id(rec, amplitude, phase_id):
    primary = [rec["protocol_sha256"], rec["split"], rec["task_id"], rec["controller"],
               rec["weight_policy"], float(amplitude), phase_id]
    return hashlib.sha256(json.dumps(primary, separators=(",", ":")).encode()).hexdigest()[:16]


def migrate_record_dir(run_dir):
    changed = {}
    for rp in list(run_dir.glob("*.json")):
        rec = json.loads(rp.read_text())
        if "run_id" not in rec: continue
        if rec.get("stage", "").startswith("pilot_"):
            rec["split"] = rec["stage"]
        tp = run_dir.parent/"traces"/(rec["run_id"]+".npz")
        task_kind = "zero" if rec["task_id"].startswith("zero_") else "single"
        amp = rec.get("amplitude")
        if amp is None:
            if task_kind == "zero": amp = 0.
            elif "_a" in rec["task_id"]: amp = float(rec["task_id"].rsplit("_a", 1)[1])
            else: amp = .006
        phase = rec.get("phase_id")
        if phase is None:
            if rec["task_id"].startswith("zero_"): phase = "zero"
            elif rec["task_id"].startswith("pilot_"):
                phase_map = {"pilot_depth_k8": .1, "pilot_depth_k32": .4,
                             "pilot_east_k8": .7, "pilot_east_k32": 1.}
                phase = f"phase_{phase_map[rec['task_id']]:g}"
            elif rec["task_id"].startswith("B1_") or rec["task_id"].startswith("B4_"):
                phase = "primary"
            else:
                tasks = [json.loads(line) for line in (OUT/"tasks.jsonl").read_text().splitlines()]
                task = next(t for t in tasks if t["task_id"] == rec["task_id"])
                phase = task.get("phase_id", f"phase_{float(task['phase_rad']):g}" if "phase_rad" in task else "zero")
        rec["amplitude"] = float(amp); rec["phase_id"] = phase
        new_id = run_id(rec, amp, phase)
        new_rp = run_dir/(new_id+".json")
        new_tp = tp.with_name(new_id+".npz")
        if new_id != rec["run_id"]:
            if new_rp.exists() or new_tp.exists(): raise FileExistsError(new_id)
            tp.rename(new_tp)
            rec["run_id"] = new_id
            rp.unlink()
        rec["trace_sha256"] = hashlib.sha256(new_tp.read_bytes()).hexdigest()
        new_rp.write_text(json.dumps(rec, indent=2)+"\n")
        changed[rp.stem] = rec
        changed[rec["run_id"]] = rec
    return changed


def rewrite_manifest(path, changed):
    if not path.exists(): return
    data = json.loads(path.read_text())
    for item in data.get("runs", []):
        old_id = item.get("run_id")
        if old_id in changed:
            rec = changed[old_id]
            item.update({"run_id": rec["run_id"], "status": rec["status"],
                         "trace_sha256": rec["trace_sha256"]})
    path.write_text(json.dumps(data, indent=2)+"\n")


def main():
    groups = {}
    groups.update(migrate_record_dir(OUT/"runs"))
    for workers in (1, 2, 4):
        base = OUT/f"pilot_w{workers}"
        groups.update(migrate_record_dir(base/"runs"))
    long_changed = migrate_record_dir(OUT/"long_matrix"/"runs")
    groups.update(long_changed)
    rewrite_manifest(OUT/"manifest.json", groups)
    rewrite_manifest(OUT/"long_matrix"/"manifest.json", long_changed)
    pilot = json.loads((OUT/"long_matrix"/"long_throughput_pilot.json").read_text())
    if pilot.get("run_id") in long_changed:
        pilot.update(long_changed[pilot["run_id"]])
        (OUT/"long_matrix"/"long_throughput_pilot.json").write_text(json.dumps(pilot, indent=2)+"\n")
    print(json.dumps({"migrated_runs": len(groups), "long_runs": len(long_changed)}))


if __name__ == "__main__": main()
