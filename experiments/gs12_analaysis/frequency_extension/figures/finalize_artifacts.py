#!/usr/bin/env python3
"""Generate final figures/report and rebuild a complete artifact hash manifest."""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=Path(__file__).resolve().parents[4]
FIG=ROOT/"figures"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def stage_for(r):
    if r["run_id"].startswith("bridge_"):return "bridge-7"
    s=r["speed_mps"]; amp=r.get("amplitude_per_line",0.); phase=r.get("phase_id","primary")
    if s==15 and phase=="secondary":return "phase-15"
    if s==15 and amp==.003:return "half-15"
    if s==15:return "main-15"
    if amp==.003:return "other-speeds-half-constrained"
    return "other-speeds"


def main():
    for name in ("generate_figures.py","build_report.py"):
        env=os.environ.copy();env["MPLCONFIGDIR"]="/tmp/gs12-mplconfig"
        subprocess.run([sys.executable,str(FIG/name)],cwd=REPO,env=env,check=True)
    run_files=sorted((ROOT/"runs").glob("*.json"),key=lambda p:(p.stat().st_mtime_ns,p.name))
    run_records=[json.loads(p.read_text()) for p in run_files]
    stage_runs=[];stage_counts=Counter()
    for p,r in zip(run_files,run_records):
        stage=stage_for(r);stage_counts[stage]+=1
        stage_runs.append({"run_id":r["run_id"],"stage":stage,"status":r["status"],
          "speed_mps":r.get("speed_mps"),"condition":r.get("condition"),"band":r.get("band"),
          "axis":r.get("axis"),"amplitude_per_line":r.get("amplitude_per_line"),"phase_id":r.get("phase_id"),
          "protocol_sha256":r.get("protocol_sha256"),"source_sha256":r.get("source_sha256"),
          "trace_sha256":r.get("trace_sha256"),"observed_write_order":len(stage_runs)+1})
    replay_files=sorted((ROOT/"direction_replays"/"runs").glob("*.json")) if (ROOT/"direction_replays"/"runs").exists() else []
    replays=[json.loads(p.read_text()) for p in replay_files]
    inventory={}
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p==ROOT/"manifest.json":continue
        rel=p.relative_to(ROOT).as_posix();inventory[rel]={"sha256":sha(p),"size_bytes":p.stat().st_size}
    summary=json.loads((ROOT/"summary.json").read_text())
    protocol=json.loads((ROOT/"protocol.json").read_text())
    verification=json.loads((ROOT/"input_verification.json").read_text())
    manifest={"manifest_version":1,"experiment_id":"GS12-FRF-EXT-2026-11-10",
      "protocol_sha256":sha(ROOT/"protocol.json"),"source_sha256":protocol["source_sha256"],
      "input_zip_sha256":{"code":verification["code_zip_sha256"],"data":verification["data_zip_sha256"]},
      "planned_full_matrix_groups":summary["expected_full_matrix_groups"],
      "full_matrix_groups_missing":summary["missing_expected_full_matrix_groups"],
      "analyzed_matrix_groups":summary["matrix_groups"],"analysis_missing":summary["missing"],
      "failed_runs":summary["failed_runs"],"constrained_runs":summary["constrained_runs"],
      "run_counts_by_stage":dict(stage_counts),"run_records_in_observed_file_write_order":stage_runs,
      "direction_replays":{"planned":15,"completed":len(replays),
        "run_ids":[r["run_id"] for r in replays],
        "prediction_error_relative":[r["prediction_error_relative"] for r in replays]},
      "superseded_protocol_count":len(protocol.get("superseded_protocols",[])),
      "artifact_count_excluding_manifest":len(inventory),
      "total_artifact_bytes_excluding_manifest":sum(x["size_bytes"] for x in inventory.values()),
      "files_sha256_and_sizes":inventory,
      "notes":["Parallel runs are ordered by observed run JSON file-write timestamps; simultaneous completion order is approximate.",
        "All artifact files under frequency_extension are hashed except manifest.json itself.",
        "Superseded runner-only protocol revisions are retained and explicitly listed in protocol.json; analysis accepts only matching protocol/source hash pairs."]}
    (ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"manifest":str(ROOT/"manifest.json"),"artifact_count":len(inventory),
      "bytes":manifest["total_artifact_bytes_excluding_manifest"],"runs":len(run_records),
      "direction_replays":len(replays),"stages":dict(stage_counts)}))


if __name__=="__main__":main()
