#!/usr/bin/env python3
"""Verify only the requested 10-state main and ideal-diagnostic trace sets."""
import hashlib,json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/"output"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

main=json.loads((OUT/"manifest.json").read_text())
psha=json.loads((OUT/"protocol.json").read_text())["source_sha256"]
protocol_sha=json.loads((OUT/"protocol.json").read_text())
protocol_sha=__import__("hashlib").sha256((OUT/"protocol.json").read_bytes()).hexdigest()
checked=[];errors=[]
prefix="p"+protocol_sha[:12]
for item in main["runs"]:
    if not item["run_id"].startswith(prefix):continue
    rid=item["run_id"];tp=OUT/"traces"/(rid+".npz")
    actual=sha(tp) if tp.exists() else None
    if actual!=item.get("trace_sha256"):
        errors.append({"run_id":rid,"expected":item.get("trace_sha256"),"actual":actual})
    checked.append(rid)
ideal=json.loads((OUT/"ideal_diagnostic/manifest.json").read_text())
for item in ideal["runs"]:
    rid=item["run_id"];tp=OUT/"ideal_diagnostic/traces"/(rid+".npz")
    actual=sha(tp) if tp.exists() else None
    if actual!=item.get("trace_sha256"):
        errors.append({"run_id":rid,"expected":item.get("trace_sha256"),"actual":actual})
    checked.append(rid)
doc={"scope":"model10 main protocol and ideal-immediate-actuator diagnostic",
     "main_protocol_sha256":protocol_sha,"main_runs_checked":len(checked)-len(ideal["runs"]),
     "ideal_diagnostic_runs_checked":len(ideal["runs"]),"runs_checked":len(checked),
     "trace_hash_errors":errors}
(OUT/"scope_10state_integrity.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(doc,ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
