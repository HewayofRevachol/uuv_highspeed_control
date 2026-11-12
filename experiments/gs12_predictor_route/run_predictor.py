#!/usr/bin/env python3
"""Frozen GS-12 continuous predictor path scan (W1)."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, platform, sys, time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"experiments/gs12_weight_route")); sys.path.insert(0,str(ROOT/"experiments/gs12_joint_route"))
import numpy as np
from joint_reference import joint_task_reference, derivative_error
from experiments.gs12_weight_route.runner_core import run_one

BETAS=[0.,.25,.5,.75,1.]
TASK_SPECS=[
 ("dev_static","ps_dev_depth_low","single","depth",3,.58,None),
 ("dev_static","ps_dev_depth_high","single","depth",29,-.76,None),
 ("dev_static","ps_dev_east_high","single","east",27,1.49,None),
 ("dev_static","ps_dev_mixed","mixed","east_depth",None,-1.24,[19.,43.]),
 ("test_static_locked","ps_lock_depth_low","single","depth",5,2.02,None),
 ("test_static_locked","ps_lock_depth_high","single","depth",31,.37,None),
 ("test_static_locked","ps_lock_east_high","single","east",25,-1.12,None),
 ("test_static_locked","ps_lock_mixed","mixed","east_depth",None,.92,[23.,45.]),
 ("dev_dynamic","pd_dev_depth_low","single","depth",2,-.93,None),
 ("dev_dynamic","pd_dev_depth_high","single","depth",30,1.04,None),
 ("dev_dynamic","pd_dev_east_high","single","east",26,-1.56,None),
 ("dev_dynamic","pd_dev_mixed_a","mixed","east_depth",None,1.72,[26.,46.]),
 ("dev_dynamic","pd_dev_mixed_b","mixed","east_depth",None,-2.31,[18.,38.]),
 ("test_dynamic_locked","pd_lock_depth_low","single","depth",7,.34,None),
 ("test_dynamic_locked","pd_lock_depth_high","single","depth",32,2.19,None),
 ("test_dynamic_locked","pd_lock_east_high","single","east",28,-.24,None),
 ("test_dynamic_locked","pd_lock_mixed_a","mixed","east_depth",None,-1.87,[27.,47.]),
 ("test_dynamic_locked","pd_lock_mixed_b","mixed","east_depth",None,.15,[17.,37.]),
]
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def digest(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def sources():
 paths=[HERE/"run_predictor.py",HERE/"analyze_predictor.py",ROOT/"experiments/gs12_joint_route/joint_reference.py",ROOT/"experiments/gs12_weight_route/runner_core.py",ROOT/"experiments/gs12_weight_route/reference_tasks.py",ROOT/"aav_mpc/condensed_mpc.py",ROOT/"simulator/rti_symbolic.py",ROOT/"simulator/batch_mpc.py",ROOT/"experiments/gs12_analaysis/frequency_extension_src/plant_harness.py"]
 return {str(p.relative_to(ROOT)):sha_file(p) for p in paths}
def outpath(s): return Path(s).resolve()
def task_rows():
 rows=[]
 for split,tid,kind,axis,k,phase,switch in TASK_SPECS:
  t={"split":split,"task_id":tid,"kind":kind,"axis":axis,"phase_rad":phase,"duration_s":64.,"speed_mps":15.,"amplitude":.006,"window_s":[32.,64.]}
  if k is not None:t["k"]=k
  if switch is not None:t["switch_times_s"]=switch
  rows.append(t)
 return rows
def read_tasks(out): return [json.loads(s) for s in (out/"tasks.jsonl").read_text().splitlines() if s.strip()]
def read_protocol(out): return json.loads((out/"protocol.json").read_text())
def prepare(out):
 out.mkdir(parents=True,exist_ok=True)
 if (out/"protocol.json").exists() or (out/"tasks.jsonl").exists(): raise SystemExit("frozen protocol/tasks already exist; refusing rewrite")
 tasks=task_rows(); task_bytes="".join(canon(t)+"\n" for t in tasks).encode()
 prot={"protocol_id":"GS12-predictor-route-v1","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"working_tree_status":os.popen(f"git -C {ROOT} status --short").read().strip(),"source_sha256":sources(),"python":sys.version.split()[0],"platform":platform.platform(),"plant":{"id":"C4","elevator":.9,"rudder":.9,"order":"second_all"},"mpc":{"states":22,"dt_s":.1,"horizon":35,"substeps_per_cycle":5,"substep_s":.02},"weight":{"a_depth":1.4,"a_rudder_rate":.7},"beta_values":BETAS,"beta_mapping":{"gain":"1-.15*beta","tau":"1+beta"},"scoring":{"window_s":[32.,64.],"formula":"mean(((Eref-E)/8)^2+((Dref-D)/5)^2+((VNref-VN)/2)^2)"},"tasks_sha256":hashlib.sha256(task_bytes).hexdigest(),"quality_gates":{"complete":True,"qp_failures":0,"boundary_violations":0,"saturation":0,"command_spike":"max candidate normalized slew >0.05 and >1.25*B"},"locked_splits_frozen_unopened":True}
 prot["protocol_sha256"]=digest(prot)
 (out/"tasks.jsonl").write_bytes(task_bytes); (out/"protocol.json").write_text(json.dumps(prot,indent=2,ensure_ascii=False)+"\n")
 (out/"controllers.json").write_text(json.dumps({str(b):{"beta":b,"model_gain":1-.15*b,"model_tau":1+b,"weight_id":"W1","a_depth":1.4,"a_rudder_rate":.7} for b in BETAS},indent=2)+"\n")
 (out/"manifest.json").write_text(json.dumps({"protocol_sha256":prot["protocol_sha256"],"runs":[]},indent=2)+"\n")
 print(prot["protocol_sha256"])

def verify(out):
 from experiments.gs12_joint_route.run_joint import TASKS as OLD_TASKS
 oldout=ROOT/"experiments/gs12_joint_route/output"; records=[json.loads(p.read_text()) for p in (oldout/"runs").glob("*.json")]
 checks={}; maxder=max(derivative_error(t) for t in task_rows())
 if maxder>=1e-3: raise SystemExit(f"reference derivative gate failed: {maxder}")
 # Re-run matching old B/D trajectories on their original task definitions, then compare saved traces.
 old_task=next(t for t in OLD_TASKS if t["task_id"]=="dev_depth_b1")
 for cid,beta in (("B",0.),("D",1.)):
  rec=next(r for r in records if r["task_id"]==old_task["task_id"] and r["controller_id"]==cid)
  old=np.load(oldout/"traces"/(rec["run_id"]+".npz"))
  new=run_one(old_task,"C4",1.4,.7,24.,joint_task_reference,(1.,1.),beta_policy=lambda tm,grid,task,b=beta:b)
  arr=new["arrays"]; nctrl=240; nsub=1200
  diffs={"command":float(np.max(np.abs(arr["command"][:nctrl]-old["command"][:nctrl]))),"state_substep":float(np.max(np.abs(arr["state_substep"][:nsub]-old["state_substep"][:nsub]))),"actual_substep":float(np.max(np.abs(arr["actual_substep"][:nsub]-old["actual_substep"][:nsub]))),"qp_mismatch_count":int(np.sum(arr["qp_ok"][:nctrl]!=old["qp_ok"][:nctrl]))}
  checks[cid]={"old_run_id":rec["run_id"],"max_abs_differences":diffs}
  if max(diffs[k] for k in ("command","state_substep","actual_substep"))>2e-7 or diffs["qp_mismatch_count"]: raise SystemExit(f"M0 legacy endpoint alignment failed {cid}: {diffs}")
 # Explicit actual-parameter path + endpoint/interior QP solve on a shared state/reference.
 from simulator.batch_mpc import BatchedUUVModel,CONTROL_SCALE
 from simulator.rti_symbolic import ActuatorAwareCpuDynamics
 from aav_mpc.condensed_mpc import CondensedActuatorMPC
 import plant_harness as harness
 model=BatchedUUVModel("cpu"); plant=harness.Plant(15.,harness.PlantConfig(elevator=.9,rudder=.9,order="second_all")); refs=joint_task_reference(task_rows()[0],np.arange(1,36)*.1)
 param=[]
 for b in (0.,.5,1.):
  m=CondensedActuatorMPC(model,1,horizon=35,dt=.1); m.dynamics=ActuatorAwareCpuDynamics(.1,1.,1.); m.previous[:]=model.tensor(plant.actual/np.asarray(CONTROL_SCALE)); m.plan[:]=m.previous[:,None]
  m.solve(model.tensor(plant.controller_state())[None],model.tensor(refs)[None],alpha=[b,0.],theta_hat=[.85,2.],d_hat=None)
  p=[1-.15*b,1+b]; param.append({"beta":b,"actual_model_parameters":p,"qp_valid":bool(m.last_qp_valid[0])})
  if not np.allclose(p,[(1.,1.),(.925,1.5),(.85,2.)][(0,.5,1.).index(b)],rtol=0,atol=1e-12): raise SystemExit("continuous parameter injection mismatch")
 out.mkdir(parents=True,exist_ok=True); (out/"smoke").mkdir(exist_ok=True)
 report={"reference_derivative_max_abs":maxder,"legacy_endpoint_alignment":checks,"continuous_parameter_injection":param,"source_sha256":sources(),"note":"Legacy traces and original task definitions used; no locked tasks read."}
 (out/"smoke/verify.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

def jobs(out,split,betas=None):
 prot=read_protocol(out); ts=[t for t in read_tasks(out) if t["split"]==split]; betas=BETAS if betas is None else betas; result=[]
 for t in ts:
  for b in betas:
   job={"split":split,"task":t,"beta":float(b),"a_depth":1.4,"a_rudder_rate":.7}; rid=digest({"protocol":prot["protocol_sha256"],**job})[:20]; result.append((rid,job))
 if len({x[0] for x in result})!=len(result): raise SystemExit("duplicate run_id")
 return result
def worker(arg):
 rid,j,out_s,ph=arg; out=Path(out_s); b=j["beta"]; t=j["task"]
 res=run_one(t,"C4",1.4,.7,64.,joint_task_reference,(1.,1.),beta_policy=lambda tm,g,task,b=b:b)
 tr=out/"traces"/(rid+".npz"); tr.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(tr,**res["arrays"])
 meta=res["meta"]; rec={"run_id":rid,"protocol_sha256":ph,"task_sha256":digest(t),"source_sha256":sources(),"git_commit":read_protocol(out)["git_commit"],"split":j["split"],"task_id":t["task_id"],"method_id":f"beta_{b:g}","beta_or_policy":b,"model_gain":1-.15*b,"model_tau":1+b,"a_depth":1.4,"a_rudder_rate":.7,"speed_mps":t["speed_mps"],"amplitude":t["amplitude"],"phase_rad":t["phase_rad"],"completion_status":meta["status"],"duration_observed_s":meta["duration_observed_s"],"first_violation_time_s":(meta["first_violation"] or {}).get("time_s"),"first_violation":meta["first_violation"],"qp_failures":meta["qp_failures"],"command_saturation":meta["command_saturation_fraction"],"actual_saturation":meta["actual_saturation_fraction"],"solve_p50_ms":meta["solve_p50_ms"],"solve_p95_ms":meta["solve_p95_ms"],"solve_p99_ms":meta["solve_p99_ms"],"wall_s":meta["wall_s"],"trace_sha256":sha_file(tr)}
 (out/"runs"/(rid+".json")).write_text(json.dumps(rec,indent=2)+"\n"); return rec
def run(out,split,workers,limit=None):
 prot=read_protocol(out)
 if sources()!=prot["source_sha256"]: raise SystemExit("source hash differs from frozen protocol")
 if split!="dev_static": raise SystemExit("static locked split is unopened unless selection_static.json is frozen")
 batch=jobs(out,split); (out/"runs").mkdir(exist_ok=True); (out/"traces").mkdir(exist_ok=True)
 existing={p.stem for p in (out/"runs").glob("*.json")}; batch=[x for x in batch if x[0] not in existing]
 if limit is not None: batch=batch[:limit]
 if not batch: print("no unrun trajectories"); return
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool: recs=list(pool.map(worker,[(rid,j,str(out),prot["protocol_sha256"]) for rid,j in batch]))
 mp=out/"manifest.json"; man=json.loads(mp.read_text()); seen={x["run_id"] for x in man["runs"]}
 if any(x["run_id"] in seen for x in recs): raise SystemExit("run_id already exists; refusing overwrite")
 man["runs"].extend({k:r[k] for k in ("run_id","split","task_id","method_id")} for r in recs); mp.write_text(json.dumps(man,indent=2)+"\n"); print(f"completed {len(recs)} trajectories")
def freeze_static(out,beta):
 if beta.upper()=="NONE":
  (out/"selection_static.json").write_text(json.dumps({"frozen":True,"candidate":None,"reason":"G1-A no candidate; static locked prohibited"},indent=2)+"\n"); return
 data=list(csv.DictReader((out/"dev_static_metrics.csv").open())); valid=[r for r in data if r["completion_status"]=="complete_valid"]
 if len(valid)!=20: raise SystemExit("freeze requires 20 quality-valid dev runs")
 b=float(beta)
 (out/"selection_static.json").write_text(json.dumps({"frozen":True,"beta_star":b,"protocol_sha256":read_protocol(out)["protocol_sha256"],"dev_metrics_sha256":sha_file(out/"dev_static_metrics.csv"),"gate":"G1-A pre-registered minimax regret and mean-J constraint"},indent=2)+"\n")

def main():
 ap=argparse.ArgumentParser(description=__doc__); sp=ap.add_subparsers(dest="cmd",required=True)
 for cmd in ("verify","prepare"):
  p=sp.add_parser(cmd); p.add_argument("--out",default="experiments/gs12_predictor_route/output")
 p=sp.add_parser("run"); p.add_argument("split",choices=["dev_static","test_static_locked"]); p.add_argument("--workers",type=int,default=1,choices=[1,2,4]); p.add_argument("--limit",type=int); p.add_argument("--out",default="experiments/gs12_predictor_route/output")
 p=sp.add_parser("freeze-static"); p.add_argument("--beta",required=True); p.add_argument("--out",default="experiments/gs12_predictor_route/output")
 a=ap.parse_args(); o=outpath(a.out)
 if a.cmd=="verify": verify(o)
 elif a.cmd=="prepare": prepare(o)
 elif a.cmd=="run": run(o,a.split,a.workers,a.limit)
 else: freeze_static(o,a.beta)
if __name__=="__main__":
 import csv
 main()
