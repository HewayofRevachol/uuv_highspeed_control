"""Conditional M3 reference-feature gate, injection audit and dynamic runs."""
import argparse,concurrent.futures,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT_DEFAULT=HERE/"output"
sys.path[:0]=[str(ROOT),str(ROOT/"experiments/gs12_weight_route"),str(ROOT/"experiments/gs12_joint_route")]
import numpy as np
from experiments.gs12_predictor_route.run_predictor import read_protocol,read_tasks,read_tasks as _rt,canon,digest,sha_file,sources
from experiments.gs12_predictor_route.spectral_policy import features,SpectralBetaPolicy,StaircaseBetaPolicy
from experiments.gs12_weight_route.runner_core import run_one
from joint_reference import joint_task_reference
def _dynamic_worker(arg):
 rid,j,out_s,sel=arg;out=Path(out_s);t=j["task"];name,b=j["method"],j["beta"]
 policy=(lambda tm,g,task,b=b:b) if name=="fixed" else SpectralBetaPolicy(float(sel["F_dev"]),name)
 res=run_one(t,"C4",1.4,.7,64.,joint_task_reference,(1.,1.),beta_policy=policy,preview_s=16.)
 trace=out/"traces"/(rid+".npz");np.savez_compressed(trace,**res["arrays"]);m=res["meta"]
 rec={"run_id":rid,"protocol_sha256":read_protocol(out)["protocol_sha256"],"dynamic_protocol_sha256":json.loads((out/"dynamic_protocol.json").read_text())["dynamic_protocol_sha256"],"task_sha256":digest(t),"source_sha256":sources(),"policy_source_sha256":sha_file(HERE/"spectral_policy.py"),"git_commit":read_protocol(out)["git_commit"],"split":j["split"],"task_id":t["task_id"],"method_id":f"beta_{b:g}" if name=="fixed" else name,"beta_or_policy":b if name=="fixed" else name,"model_gain":"1-.15*beta_t","model_tau":"1+beta_t","a_depth":1.4,"a_rudder_rate":.7,"speed_mps":t["speed_mps"],"amplitude":t["amplitude"],"phase_rad":t["phase_rad"],"completion_status":m["status"],"duration_observed_s":m["duration_observed_s"],"first_violation_time_s":(m["first_violation"] or {}).get("time_s"),"qp_failures":m["qp_failures"],"command_saturation":m["command_saturation_fraction"],"actual_saturation":m["actual_saturation_fraction"],"solve_p50_ms":m["solve_p50_ms"],"solve_p95_ms":m["solve_p95_ms"],"solve_p99_ms":m["solve_p99_ms"],"wall_s":m["wall_s"],"trace_sha256":sha_file(trace)}
 (out/"runs"/(rid+".json")).write_text(json.dumps(rec,indent=2)+"\n");return rec
def save_feature_probe(out):
 out=Path(out); prot=read_protocol(out); rows=[]; med={}
 dynamic=[t for t in read_tasks(out) if t["split"]=="dev_dynamic"]
 for task in dynamic:
  grid=joint_task_reference(task,np.arange(0,80.0001,.1)); vals=[]
  for ts in np.arange(0,64,.5):
   i=round(ts/.1);f=features(grid[i:i+160]);vals.append((float(ts),f))
   rows.append({"task_id":task["task_id"],"window_start_s":ts,"window_end_s":ts+15.9,"E_total":f["E_total"],"rho_high":"" if f["rho_high"] is None else f["rho_high"],"low_energy":f["low_energy"],"omega_resolution_rad_s":f["omega_resolution_rad_s"]})
  def median(a,b):
   q=[f["rho_high"] for t,f in vals if a<=t<=b and not f["low_energy"]];return float(np.median(q)) if q else None
  if task["kind"]=="single":med[task["task_id"]]={"median_32_48":median(32,48)}
  elif task["task_id"].endswith("mixed_a"):med[task["task_id"]]={"high_platform_28_30":median(28,30),"low_platform_48_56":median(48,56)}
  else:med[task["task_id"]]={"high_platform_20_22":median(20,22),"low_platform_40_48":median(40,48)}
 with (out/"spectral_features.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 low=med["pd_dev_depth_low"]["median_32_48"];gates={"high_depth_delta":med["pd_dev_depth_high"]["median_32_48"]-low,"east_high_delta":med["pd_dev_east_high"]["median_32_48"]-low,"mixed_a_delta":med["pd_dev_mixed_a"]["high_platform_28_30"]-med["pd_dev_mixed_a"]["low_platform_48_56"],"mixed_b_delta":med["pd_dev_mixed_b"]["high_platform_20_22"]-med["pd_dev_mixed_b"]["low_platform_40_48"]}
 passed=gates["high_depth_delta"]>=.30 and gates["east_high_delta"]>=.30 and gates["mixed_a_delta"]>=.20 and gates["mixed_b_delta"]>=.20
 result={"protocol_sha256":prot["protocol_sha256"],"feature_definition":"3 channels normalized (E/8,(D-20)/5,(VN-15)/2); remove linear trend; Hann; rFFT n=160,d=.1; sum squared magnitude/n^2 over non-DC; high omega>=2.2 rad/s","frequency_resolution_rad_s":2*np.pi/16,"feature_medians":med,"gate_deltas":gates,"feature_gate_pass":bool(passed),"low_energy_windows":sum(bool(r["low_energy"]) for r in rows),"planned_reference_only":True}
 (out/"feature_gate.json").write_text(json.dumps(result,indent=2)+"\n");return result
def create_dynamic_protocol(out):
 out=Path(out);main=read_protocol(out);dyn=[t for t in read_tasks(out) if t["split"] in ("dev_dynamic","test_dynamic_locked")]
 core={k:v for k,v in main["source_sha256"].items() if k not in ("experiments/gs12_predictor_route/run_predictor.py","experiments/gs12_predictor_route/analyze_predictor.py")}
 core_now={k:v for k,v in sources().items() if k in core}
 if core_now!=core:raise SystemExit("M3 core source hashes differ from frozen M1 protocol")
 spec={"protocol_id":"GS12-predictor-route-M3-v1","parent_protocol_sha256":main["protocol_sha256"],"core_source_sha256":core,"spectral_policy_sha256":sha_file(HERE/"spectral_policy.py"),"policy_runner_sha256":sha_file(HERE/"run_dynamic.py"),"task_sha256":hashlib.sha256((out/"tasks.jsonl").read_bytes()).hexdigest(),"dev_tasks":[t["task_id"] for t in dyn if t["split"]=="dev_dynamic"],"locked_tasks":[t["task_id"] for t in dyn if t["split"]=="test_dynamic_locked"],"reference_preview_s":16.,"preview_samples":160,"sample_dt_s":.1,"feature":"3 normalized output-reference channels; detrend constant+linear; Hann window; rFFT; E=sum(|FFT|^2)/n^2 for omega>0; rho=sum high omega>=2.2 / E; E<1e-8 holds beta","R0":{"rho_linear_range":[.25,.75],"slew_per_update":.1,"update_s":.5,"initial_beta":.75},"R1":{"rho_linear_range":[.35,.65],"slew_per_update":.1,"update_s":.5,"initial_beta":.75,"allowed_only_if_R0_quality_valid_but_not_promoted":True},"development_gate":"relative to frozen F_dev beta=.75: mean J reduction >=5%, every task nonworse, no QP/boundary/saturation/spike blockers","locked_gate":"one shot; only if a rule passes development; relative to same frozen F_dev","locked_results_opened":False,"created_after_M2_failed_with_M1_endpoint_cross":True}
 spec["dynamic_protocol_sha256"]=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()).hexdigest();p=out/"dynamic_protocol.json"
 if p.exists():raise SystemExit("dynamic_protocol already exists; refusing rewrite")
 p.write_text(json.dumps(spec,indent=2)+"\n");return spec
def verify_dynamic(out):
 out=Path(out)
 import osqp
 from unittest.mock import patch
 from simulator.batch_mpc import CONTROL_SCALE
 task=next(t for t in read_tasks(out) if t["task_id"]=="pd_dev_depth_low")
 checks={}
 for b in (0.,1.):
  ref=run_one(task,"C4",1.4,.7,24.,joint_task_reference,(1.,1.),beta_policy=lambda tm,g,t,b=b:b)
  params=(1.,1.) if b==0 else (.85,2.)
  static=run_one(task,"C4",1.4,.7,24.,joint_task_reference,params)
  diffs={k:float(np.max(np.abs(ref["arrays"][k]-static["arrays"][k]))) for k in ("command","state_substep","actual_substep")};diffs["qp_mismatch_count"]=int(np.sum(ref["arrays"]["qp_ok"]!=static["arrays"]["qp_ok"]))
  if max(diffs[k] for k in ("command","state_substep","actual_substep"))>2e-7 or diffs["qp_mismatch_count"]:raise SystemExit(f"dynamic endpoint mismatch beta={b}: {diffs}")
  checks[str(b)]=diffs
 task=next(t for t in read_tasks(out) if t["task_id"]=="pd_dev_mixed_a");capt=[];original=osqp.OSQP.setup
 def capture(self,*a,**kw):
  P=kw.get("P",a[0] if a else None);q=kw.get("q",a[1] if len(a)>1 else None);capt.append({"P":P.toarray().copy(),"q":np.asarray(q).copy()});return original(self,*a,**kw)
 with patch.object(osqp.OSQP,"setup",capture):stair=run_one(task,"C4",1.4,.7,4.,joint_task_reference,(1.,1.),beta_policy=StaircaseBetaPolicy())
 aa=stair["arrays"];beta=aa["beta_t"];uniq=[int(np.argmin(np.abs(beta-v))) for v in (.0,.1,.2)]; qp_target_differences=[]
 for i,j in zip(uniq,uniq[1:]):qp_target_differences.append(float(np.max(np.abs(capt[i]["q"]-capt[j]["q"]))))
 expected=np.repeat(np.tile(np.array([0.,.1,.2,.1]),5),5)[:20]
 if not np.allclose(beta[:20],expected,atol=1e-12):raise SystemExit("staircase beta sequence mismatch")
 if not all(x>0 for x in qp_target_differences):raise SystemExit("staircase did not change QP gradient")
 scale=np.asarray(CONTROL_SCALE,dtype=float)
 result={"endpoint_checks":checks,"staircase_beta_first_20":beta[:20].tolist(),"staircase_model_gain_first_20":aa["model_gain_t"][:20].tolist(),"staircase_model_tau_first_20":aa["model_tau_t"][:20].tolist(),"qp_gradient_max_differences":qp_target_differences,"qp_failures":stair["meta"]["qp_failures"],"max_normalized_command_step":float(np.max(np.abs(np.diff(aa["command"]/scale,axis=0)))),"max_normalized_actual_fin_step":float(np.max(np.abs(np.diff(aa["actual_substep"]/scale,axis=0)))),"continuity":"single run_one invocation, no reset call between beta steps; state and actual fin traces remain continuous","note":"The policy only reads precomputed reference values. No plant state or task label is passed to feature calculation."}
 (Path(out)/"dynamic_interface_audit.json").write_text(json.dumps(result,indent=2)+"\n");return result
def run_jobs(out,split,workers,policy_name="R0"):
 out=Path(out);sel=json.loads((out/"selection_static.json").read_text());m2=json.loads((out/"m2_decision.json").read_text());feat=json.loads((out/"feature_gate.json").read_text());
 if not feat["feature_gate_pass"]:raise SystemExit("feature gate failed; dynamic closed-loop not permitted")
 if not (out/"dynamic_protocol.json").exists() or not (out/"dynamic_interface_audit.json").exists():raise SystemExit("dynamic protocol and interface audit required")
 if split=="test_dynamic_locked":
  ds=json.loads((out/"selection_dynamic.json").read_text()) if (out/"selection_dynamic.json").exists() else None
  if not ds or not ds.get("frozen") or not ds.get("development_gate_pass"):raise SystemExit("dynamic test locked requires a frozen, promoted dev policy")
 if split!="dev_dynamic":raise SystemExit("dynamic locked run requires frozen dev promotion; no locked task has been opened")
 tasks=[t for t in read_tasks(out) if t["split"]==split];betas=[0.,1.,float(sel["candidate_beta"]),float(sel["F_dev"])];methods=[("fixed",b) for b in sorted(set(betas))]+[(policy_name,None)];jobs=[]
 for t in tasks:
  for method,b in methods:
   job={"task":t,"method":method,"beta":b,"split":split};rid=digest({"parent_protocol":read_protocol(out)["protocol_sha256"],"dynamic_protocol":json.loads((out/"dynamic_protocol.json").read_text())["dynamic_protocol_sha256"],**job})[:20];jobs.append((rid,job))
 if len({x[0] for x in jobs})!=len(jobs):raise SystemExit("duplicate dynamic run id")
 existing={f.stem for f in (out/"runs").glob("*.json")};jobs=[x for x in jobs if x[0] not in existing]
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:recs=list(pool.map(_dynamic_worker,[(rid,j,str(out),sel) for rid,j in jobs]))
 mp=out/"manifest.json";man=json.loads(mp.read_text());known={r["run_id"] for r in man["runs"]}
 if any(r["run_id"] in known for r in recs):raise SystemExit("dynamic run id collision")
 man["runs"].extend({k:r[k] for k in ("run_id","split","task_id","method_id")} for r in recs);mp.write_text(json.dumps(man,indent=2)+"\n");print(f"completed {len(recs)} dynamic dev runs ({policy_name})")
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True)
 for cmd in ("feature-probe","verify-dynamic"):
  q=s.add_parser(cmd);q.add_argument("--out",default=str(OUT_DEFAULT))
 q=s.add_parser("run");q.add_argument("split",choices=["dev_dynamic","test_dynamic_locked"]);q.add_argument("--policy",choices=["R0","R1"],default="R0");q.add_argument("--workers",type=int,default=1,choices=[1,2,4]);q.add_argument("--out",default=str(OUT_DEFAULT))
 a=p.parse_args()
 if a.cmd=="feature-probe":
  if not (Path(a.out)/"dynamic_protocol.json").exists():create_dynamic_protocol(a.out)
  result=save_feature_probe(a.out)
  print(json.dumps(result,indent=2))
 elif a.cmd=="verify-dynamic":print(json.dumps(verify_dynamic(a.out),indent=2))
 else:run_jobs(a.out,a.split,a.workers,a.policy)
if __name__=="__main__":main()
