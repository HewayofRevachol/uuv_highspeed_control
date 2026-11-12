#!/usr/bin/env python3
"""Create side-slice manifests and verify trace hashes for final handoff."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def verify_folder(folder, write_manifest=False):
    run_dir, trace_dir = folder / "runs", folder / "traces"
    rows, errors = [], []
    for jp in sorted(run_dir.glob("*.json")):
        rec = json.loads(jp.read_text())
        rid = rec["run_id"]
        tp = trace_dir / f"{rid}.npz"
        actual = sha256(tp) if tp.exists() else None
        expected = rec.get("trace_sha256")
        if actual != expected:
            errors.append({"run_id": rid, "expected": expected, "actual": actual})
        rows.append({"run_id": rid, "status": rec.get("status"), "trace_sha256": expected,
                     "source_sha256": rec.get("source_sha256"), "reference_sha256": rec.get("reference_sha256")})
    if write_manifest:
        doc = {"runs": rows, "run_count": len(rows), "trace_hash_errors": errors}
        (folder / "manifest.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    return {"directory": folder.relative_to(OUT).as_posix(), "runs_checked": len(rows),
            "trace_hash_errors": errors}

results = []
main_manifest = json.loads((OUT / "manifest.json").read_text())
main_errors = []
main_records = 0
for item in main_manifest["runs"]:
    rid = item["run_id"]
    matches = list(OUT.glob(f"*/traces/{rid}.npz")) + list((OUT / "traces").glob(f"{rid}.npz"))
    if len(matches) != 1:
        main_errors.append({"run_id": rid, "trace_matches": len(matches)})
        continue
    main_records += 1
    actual = sha256(matches[0])
    if actual != item.get("trace_sha256"):
        main_errors.append({"run_id": rid, "expected": item.get("trace_sha256"), "actual": actual})
results.append({"directory": ".", "runs_checked": main_records, "trace_hash_errors": main_errors})
for name in ("paired22_exact", "ideal_diagnostic"):
    results.append(verify_folder(OUT / name, write_manifest=True))

doc = {"integrity_checks": results,
       "total_runs_checked": sum(x["runs_checked"] for x in results),
       "total_trace_hash_errors": sum(len(x["trace_hash_errors"]) for x in results)}
(OUT / "final_integrity.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(doc, ensure_ascii=False, indent=2))
if doc["total_trace_hash_errors"]:
    raise SystemExit(1)
