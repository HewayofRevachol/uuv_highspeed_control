"""Freeze or refuse promotion of a tested dynamic development rule."""
import argparse,json,hashlib
from pathlib import Path
OUT=Path("experiments/gs12_predictor_route/output")
def freeze(policy):
 decision=json.loads((OUT/f"{policy.lower()}_development_decision.json").read_text());protocol=json.loads((OUT/"dynamic_protocol.json").read_text())
 if not decision.get("development_gate_pass"):
  result={"frozen":False,"policy":policy,"development_gate_pass":False,"reason":"rule did not beat pre-frozen F_dev by 5% with taskwise nonworse quality-valid results; test_dynamic_locked remains closed","dynamic_protocol_sha256":protocol["dynamic_protocol_sha256"],"decision_sha256":hashlib.sha256((OUT/f"{policy.lower()}_development_decision.json").read_bytes()).hexdigest()}
 else:
  result={"frozen":True,"policy":policy,"development_gate_pass":True,"F_dev_beta":json.loads((OUT/"selection_static.json").read_text())["F_dev"],"initial_beta":json.loads((OUT/"selection_static.json").read_text())["F_dev"],"decision_sha256":hashlib.sha256((OUT/f"{policy.lower()}_development_decision.json").read_bytes()).hexdigest(),"dynamic_protocol_sha256":protocol["dynamic_protocol_sha256"],"locked_tasks":[t["task_id"] for t in map(json.loads,(OUT/"tasks.jsonl").read_text().splitlines()) if t.get("split")=="test_dynamic_locked"]}
 path=OUT/"selection_dynamic.json"
 if path.exists():raise SystemExit("selection_dynamic.json already exists; refusing rewrite")
 path.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("cmd",choices=["freeze-dynamic"]);p.add_argument("--policy",choices=["R0","R1"],required=True);a=p.parse_args();freeze(a.policy)
