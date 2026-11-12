"""Execute only the pre-registered M2 static locked comparison."""
import concurrent.futures,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"experiments/gs12_weight_route"));sys.path.insert(0,str(ROOT/"experiments/gs12_joint_route"))
import numpy as np
from experiments.gs12_predictor_route.run_predictor import task_rows,digest,sha_file,read_protocol,read_tasks
from experiments.gs12_weight_route.runner_core import run_one
from joint_reference import joint_task_reference
def sources():
 ps=[HERE/"run_static_locked.py",ROOT/"experiments/gs12_weight_route/runner_core.py",ROOT/"experiments/gs12_joint_route/joint_reference.py",ROOT/"experiments/gs12_weight_route/reference_tasks.py",ROOT/"aav_mpc/condensed_mpc.py",ROOT/"simulator/rti_symbolic.py",ROOT/"simulator/batch_mpc.py",ROOT/"experiments/gs12_analaysis/frequency_extension_src/plant_harness.py"]
 return {str(p.relative_to(ROOT)):sha_file(p) for p in ps}
def worker(args):
 rid,t,b,out_s,protocol_sha=args;out=Path(out_s);res=run_one(t,"C4",1.4,.7,64.,joint_task_reference,(1.,1.),beta_policy=lambda tm,g,task,b=b:b)
 trace=out/"traces"/(rid+".npz");np.savez_compressed(trace,**res["arrays"]);m=res["meta"]
 rec={"run_id":rid,"protocol_sha256":protocol_sha,"task_sha256":digest(t),"source_sha256":sources(),"git_commit":read_protocol(out)["git_commit"],"split":"test_static_locked","task_id":t["task_id"],"method_id":f"beta_{b:g}","beta_or_policy":b,"model_gain":1-.15*b,"model_tau":1+b,"a_depth":1.4,"a_rudder_rate":.7,"speed_mps":t["speed_mps"],"amplitude":t["amplitude"],"phase_rad":t["phase_rad"],"completion_status":m["status"],"duration_observed_s":m["duration_observed_s"],"first_violation_time_s":(m["first_violation"] or {}).get("time_s"),"first_violation":m["first_violation"],"qp_failures":m["qp_failures"],"command_saturation":m["command_saturation_fraction"],"actual_saturation":m["actual_saturation_fraction"],"solve_p50_ms":m["solve_p50_ms"],"solve_p95_ms":m["solve_p95_ms"],"solve_p99_ms":m["solve_p99_ms"],"wall_s":m["wall_s"],"trace_sha256":sha_file(trace)}
 (out/"runs"/(rid+".json")).write_text(json.dumps(rec,indent=2)+"\n");return rec
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=1,choices=[1,2,4]);p.add_argument("--out",default="experiments/gs12_predictor_route/output");a=p.parse_args();out=Path(a.out);sel=json.loads((out/"selection_static.json").read_text());prot=read_protocol(out)
 if not sel.get("G1_A_pass") or sel.get("candidate_beta") in (0,1):raise SystemExit("G1-A non-endpoint candidate not frozen; static lock prohibited")
 if not (out/"m2_frozen.json").exists():raise SystemExit("M2 candidate, comparator, metrics and task list must be frozen first")
 freeze=json.loads((out/"m2_frozen.json").read_text()); betas=freeze["methods_beta"];tasks=[t for t in read_tasks(out) if t["split"]=="test_static_locked"]
 jobs=[]
 for t in tasks:
  for b in betas:
   rid=digest({"protocol":prot["protocol_sha256"],"split":"test_static_locked","task":t,"beta":b,"W1":[1.4,.7]})[:20];jobs.append((rid,t,b,str(out),prot["protocol_sha256"]))
 if len({j[0] for j in jobs})!=len(jobs):raise SystemExit("duplicate run_id")
 seen={x.stem for x in (out/"runs").glob("*.json")};jobs=[j for j in jobs if j[0] not in seen]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool: recs=list(pool.map(worker,jobs))
 mp=out/"manifest.json";man=json.loads(mp.read_text());known={x["run_id"] for x in man["runs"]}
 if any(x["run_id"] in known for x in recs):raise SystemExit("duplicate manifest run id")
 man["runs"].extend({k:r[k] for k in ("run_id","split","task_id","method_id")} for r in recs);mp.write_text(json.dumps(man,indent=2)+"\n");print(f"completed {len(recs)} M2 locked trajectories")
if __name__=="__main__":main()
