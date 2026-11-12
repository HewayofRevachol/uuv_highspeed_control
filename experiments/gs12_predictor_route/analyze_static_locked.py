"""Score and audit the frozen M2 test_static_locked split."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path("experiments/gs12_predictor_route/output"); BETAS=[0.,1.,.5,.75]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 prot=json.loads((P/"protocol.json").read_text()); freeze=json.loads((P/"m2_frozen.json").read_text()); tasks=[json.loads(s) for s in (P/"tasks.jsonl").read_text().splitlines() if s.strip() and json.loads(s)["split"]=="test_static_locked"]; rows=[]; audit={"split":"test_static_locked","planned":16,"actual":0,"missing":[],"duplicate_run_ids":[],"sha_mismatch":[],"early_stop":[],"qp_failure":[],"boundary_violation":[],"saturation":[],"pairing_complete":False}
 records=[json.loads(f.read_text()) for f in (P/"runs").glob("*.json") if json.loads(f.read_text()).get("split")=="test_static_locked"];audit["actual"]=len(records);byid={}
 for r in records:
  b=float(r["beta_or_policy"]);byid[(r["task_id"],b)]=r;trace=P/"traces"/(r["run_id"]+".npz")
  if not trace.exists() or sha(trace)!=r["trace_sha256"] or r["protocol_sha256"]!=prot["protocol_sha256"]:audit["sha_mismatch"].append(r["run_id"])
  if r["completion_status"]!="complete_valid":audit["early_stop"].append(r["run_id"])
  if r["qp_failures"]:audit["qp_failure"].append(r["run_id"])
  if r.get("first_violation_time_s") is not None:audit["boundary_violation"].append(r["run_id"])
  if any(x>0 for x in r["command_saturation"]+r["actual_saturation"]):audit["saturation"].append(r["run_id"])
  a=np.load(trace);m=(a["time_substep_s"]>=32)&(a["time_substep_s"]<=64+1e-9);ref=a["reference_substep"][m];out=a["output_substep"][m];e=np.column_stack(((ref[:,1]-out[:,1])/8,(ref[:,2]-out[:,2])/5,(ref[:,5]-out[:,5])/2));unscaled=np.column_stack((ref[:,1]-out[:,1],ref[:,2]-out[:,2],ref[:,5]-out[:,5]));rm=np.sqrt(np.mean(unscaled**2,axis=0));mse=np.mean(e**2,axis=0)
  rows.append({"run_id":r["run_id"],"task_id":r["task_id"],"beta":b,"J_track":mse.sum(),"mse_E_norm":mse[0],"mse_D_norm":mse[1],"mse_VN_norm":mse[2],"rmse_E_m":rm[0],"rmse_D_m":rm[1],"rmse_VN_mps":rm[2],"completion_status":r["completion_status"],"qp_failures":r["qp_failures"],"first_violation_time_s":r.get("first_violation_time_s"),"command_saturation":int(any(x>0 for x in r["command_saturation"])),"actual_saturation":int(any(x>0 for x in r["actual_saturation"]))})
 for t in tasks:
  for b in BETAS:
   if (t["task_id"],b) not in byid:audit["missing"].append({"task_id":t["task_id"],"beta":b})
 audit["pairing_complete"]=len(rows)==16 and not any(audit[k] for k in ("missing","sha_mismatch","early_stop","qp_failure","boundary_violation","saturation"))
 with (P/"test_static_metrics.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (P/"integrity_audit_static_locked.json").write_text(json.dumps(audit,indent=2)+"\n")
 d={(r["task_id"],r["beta"]):r for r in rows}; candidate=float(freeze["candidate_beta"]); comparator=float(freeze["F_dev_beta"]); taskids=[t["task_id"] for t in tasks]
 avg={b:float(np.mean([d[(t,b)]["J_track"] for t in taskids])) for b in BETAS}; relB=1-avg[candidate]/avg[0.];relF=1-avg[candidate]/avg[comparator]
 eachB=all(d[(t,candidate)]["J_track"]<=d[(t,0.)]["J_track"] for t in taskids);eachF=all(d[(t,candidate)]["J_track"]<=d[(t,comparator)]["J_track"] for t in taskids)
 bpass=audit["pairing_complete"] and relB>=.05 and eachB; fpass=audit["pairing_complete"] and comparator!=candidate and relF>=.05 and eachF
 detail=[]
 for t in taskids:
  x=d[(t,candidate)]["J_track"];b=d[(t,0.)]["J_track"];z=d[(t,1.)]["J_track"];f=d[(t,comparator)]["J_track"]
  detail.append({"task_id":t,"J_B":float(b),"J_D":float(z),"J_beta_star":float(x),"J_F_dev":float(f),"delta_vs_endpoint_envelope":float(x-min(b,z)),"RMSE_E_beta_star_m":float(d[(t,candidate)]["rmse_E_m"]),"RMSE_D_beta_star_m":float(d[(t,candidate)]["rmse_D_m"]),"RMSE_VN_beta_star_mps":float(d[(t,candidate)]["rmse_VN_mps"]),"nonworse_B":bool(x<=b),"nonworse_F_dev":bool(x<=f)})
 result={"candidate_beta":candidate,"F_dev_beta":comparator,"Jbar_by_method":{str(k):v for k,v in avg.items()},"relative_reduction_vs_B":relB,"relative_reduction_vs_F_dev":relF,"taskwise_nonworse_vs_B":eachB,"taskwise_nonworse_vs_F_dev":eachF,"strict_B_gate_pass":bpass,"best_fixed_comparator_gate_pass":fpass,"F_dev_same_as_candidate":comparator==candidate,"quality_valid":audit["pairing_complete"],"endpoint_envelope_and_rmse":detail,"integrity":audit}
 (P/"m2_decision.json").write_text(json.dumps(result,indent=2)+"\n")
 report=(P/"REPORT.md").read_text()+"\n\n## 本轮新锁定：M2 静态验证\n\n"+f"配对完成 {len(rows)}/16；质量完整性：{'通过' if audit['pairing_complete'] else '未通过'}。β*=0.50，相对 B 的平均 J 降幅 {relB*100:.2f}%，相对预先冻结 F_dev=0.75 的降幅 {relF*100:.2f}%。严格 B 门槛：{'通过' if bpass else '未通过'}；最佳固定比较器门槛：{'通过' if fpass else '未通过'}。\n\n| 锁定任务 | B J | D J | β*=0.50 J | F_dev=0.75 J | β*−端点包络 | RMSE_D (m) |\n|---|---:|---:|---:|---:|---:|---:|\n"+"\n".join(f"| {x['task_id']} | {x['J_B']:.8g} | {x['J_D']:.8g} | {x['J_beta_star']:.8g} | {x['J_F_dev']:.8g} | {x['delta_vs_endpoint_envelope']:.8g} | {x['RMSE_D_beta_star_m']:.8g} |" for x in detail)+"\n\n详细记录见 `test_static_metrics.csv`、`m2_decision.json` 与 `integrity_audit_static_locked.json`。\n"
 (P/"REPORT.md").write_text(report);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
