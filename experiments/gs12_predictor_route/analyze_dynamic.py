"""Audit and score conditional spectral-policy development runs."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"experiments/gs12_predictor_route/output"
import sys
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(policy="R0"):
 sel=json.loads((OUT/"selection_static.json").read_text());dyn=json.loads((OUT/"dynamic_protocol.json").read_text());tasks=[json.loads(s) for s in (OUT/"tasks.jsonl").read_text().splitlines() if s.strip() and json.loads(s)["split"]=="dev_dynamic"];records=[json.loads(f.read_text()) for f in (OUT/"runs").glob("*.json") if json.loads(f.read_text()).get("split")=="dev_dynamic" and json.loads(f.read_text()).get("method_id")==policy];
 methods=sorted({r["method_id"] for r in records});expected=["beta_0","beta_1",f"beta_{sel['candidate_beta']:g}",f"beta_{sel['F_dev']:g}",policy];rows=[];audit={"split":"dev_dynamic","policy":policy,"planned":len(tasks)*len(set(expected)),"actual":len(records),"missing":[],"sha_mismatch":[],"early_stop":[],"qp_failure":[],"boundary_violation":[],"saturation":[],"pairing_complete":False};all_records=[json.loads(f.read_text()) for f in (OUT/"runs").glob("*.json") if json.loads(f.read_text()).get("split")=="dev_dynamic" and json.loads(f.read_text()).get("method_id") in expected];lookup={(r["task_id"],r["method_id"]):r for r in all_records}
 from simulator.batch_mpc import CONTROL_SCALE
 scale=np.asarray(CONTROL_SCALE,dtype=float)
 for t in tasks:
  for method in expected:
   r=lookup.get((t["task_id"],method))
   if r is None:audit["missing"].append({"task_id":t["task_id"],"method_id":method});continue
   path=OUT/"traces"/(r["run_id"]+".npz")
   if not path.exists() or sha(path)!=r["trace_sha256"] or r["protocol_sha256"]!=json.loads((OUT/"protocol.json").read_text())["protocol_sha256"] or r["dynamic_protocol_sha256"]!=dyn["dynamic_protocol_sha256"]:audit["sha_mismatch"].append(r["run_id"])
   a=np.load(path);tm=a["time_substep_s"];mask=(tm>=32)&(tm<=64+1e-9);ref=a["reference_substep"][mask];out=a["output_substep"][mask];err=np.column_stack(((ref[:,1]-out[:,1])/8,(ref[:,2]-out[:,2])/5,(ref[:,5]-out[:,5])/2));mse=np.mean(err**2,axis=0);un=np.column_stack((ref[:,1]-out[:,1],ref[:,2]-out[:,2],ref[:,5]-out[:,5]));rm=np.sqrt(np.mean(un**2,axis=0));cmd=a["command"]/scale;act=a["actual_substep"]/scale;slew=float(np.max(np.abs(np.diff(cmd,axis=0))));actslew=float(np.max(np.abs(np.diff(act,axis=0))))
   rows.append({"run_id":r["run_id"],"task_id":t["task_id"],"method_id":method,"J_track":float(mse.sum()),"mse_E_norm":mse[0],"mse_D_norm":mse[1],"mse_VN_norm":mse[2],"rmse_E_m":rm[0],"rmse_D_m":rm[1],"rmse_VN_mps":rm[2],"beta_mean":float(np.mean(a["beta_t"])),"beta_min":float(np.min(a["beta_t"])),"beta_max":float(np.max(a["beta_t"])),"max_normalized_command_slew":slew,"max_normalized_actual_slew":actslew,"completion_status":r["completion_status"],"qp_failures":r["qp_failures"],"first_violation_time_s":r.get("first_violation_time_s"),"command_saturation":int(any(x>0 for x in r["command_saturation"])),"actual_saturation":int(any(x>0 for x in r["actual_saturation"]))})
   if r["completion_status"]!="complete_valid":audit["early_stop"].append(r["run_id"])
   if r["qp_failures"]:audit["qp_failure"].append(r["run_id"])
   if r.get("first_violation_time_s") is not None:audit["boundary_violation"].append(r["run_id"])
   if any(x>0 for x in r["command_saturation"]+r["actual_saturation"]):audit["saturation"].append(r["run_id"])
 audit["pairing_complete"]=len(rows)==audit["planned"] and not any(audit[k] for k in ("missing","sha_mismatch","early_stop","qp_failure","boundary_violation","saturation"))
 with (OUT/f"{policy.lower()}_dev_dynamic_metrics.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/f"integrity_audit_{policy.lower()}_dynamic.json").write_text(json.dumps(audit,indent=2)+"\n")
 by={(r["task_id"],r["method_id"]):r for r in rows};fmethod=f"beta_{sel['F_dev']:g}";meanJ={m:float(np.mean([by[(t["task_id"],m)]["J_track"] for t in tasks])) for m in expected};candidate_rows=[by[(t["task_id"],policy)] for t in tasks];base_rows=[by[(t["task_id"],fmethod)] for t in tasks];nonworse=all(c["J_track"]<=b["J_track"] for c,b in zip(candidate_rows,base_rows));reduction=1-meanJ[policy]/meanJ[fmethod]
 baselineB={r["task_id"]:by[(r["task_id"],"beta_0")]["max_normalized_command_slew"] for r in candidate_rows};spikes=[r["task_id"] for r in candidate_rows if r["max_normalized_command_slew"]>.05 and r["max_normalized_command_slew"]>1.25*baselineB[r["task_id"]]];quality=audit["pairing_complete"] and not spikes;promote=quality and reduction>=.05 and nonworse
 result={"policy":policy,"mean_J_by_method":meanJ,"mean_reduction_vs_F_dev":reduction,"taskwise_nonworse_vs_F_dev":nonworse,"quality_valid":quality,"command_spike_tasks":spikes,"development_gate_pass":promote,"F_dev":sel["F_dev"],"per_task":{t["task_id"]:{m:by[(t["task_id"],m)]["J_track"] for m in expected} for t in tasks},"integrity":audit}
 (OUT/f"{policy.lower()}_development_decision.json").write_text(json.dumps(result,indent=2)+"\n")
 # Plot time-varying beta for the selected policy and rho preview feature.
 policy_records=[r for r in all_records if r["method_id"]==policy];fig,axs=plt.subplots(len(policy_records),1,figsize=(10,2.3*len(policy_records)),sharex=True);axs=np.atleast_1d(axs)
 for ax,r in zip(axs,policy_records):
  a=np.load(OUT/"traces"/(r["run_id"]+".npz"));ax.step(a["time_control_s"],a["beta_t"],where="post");ax.set_ylabel("β");ax.set_title(r["task_id"]);ax.set_ylim(-.03,1.03)
 axs[-1].set_xlabel("time (s)");fig.tight_layout();fig.savefig(OUT/f"figures/{policy.lower()}_beta_time.png",dpi=160);plt.close(fig)
 append=("\n\n## M3 频谱规则开发结果："+policy+"\n\n"+f"开发配对 {len(rows)}/{audit['planned']}；质量有效：{'是' if quality else '否'}；相对冻结 F_dev=β={sel['F_dev']} 的平均 J 降幅 {reduction*100:.2f}%；逐任务不劣：{'是' if nonworse else '否'}；动态开发晋级：{'通过' if promote else '未通过'}。\n\n| 任务 | 规则 J | F_dev J | β 均值 | β 范围 |\n|---|---:|---:|---:|---:|\n"+"\n".join(f"| {t['task_id']} | {by[(t['task_id'],policy)]['J_track']:.8g} | {by[(t['task_id'],fmethod)]['J_track']:.8g} | {by[(t['task_id'],policy)]['beta_mean']:.4f} | {by[(t['task_id'],policy)]['beta_min']:.3f}–{by[(t['task_id'],policy)]['beta_max']:.3f} |" for t in tasks)+"\n")
 with (OUT/"REPORT.md").open("a") as f:f.write(append)
 print(json.dumps(result,indent=2))
if __name__=="__main__":
 p=__import__("argparse").ArgumentParser();p.add_argument("--stage",choices=["dev_dynamic","test_dynamic_locked"],default="dev_dynamic");p.add_argument("--policy",choices=["R0","R1"],default="R0");a=p.parse_args()
 if a.stage=="test_dynamic_locked":
  decision=OUT/"selection_dynamic.json"
  if not decision.exists() or not json.loads(decision.read_text()).get("frozen") or not json.loads(decision.read_text()).get("development_gate_pass"):
   raise SystemExit("test_dynamic_locked is closed: no dynamic policy passed the development gate")
  raise SystemExit("locked dynamic scoring is not activated by this experiment protocol; no test_dynamic_locked runs were authorized")
 main(a.policy)
