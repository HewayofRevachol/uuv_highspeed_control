#!/usr/bin/env python3
"""Audit and score the frozen GS-12 predictor route."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
BETAS=[0.,.25,.5,.75,1.]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def writecsv(path,rows):
 if not rows:return
 with Path(path).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def analyze(stage,out):
 out=Path(out); prot=readj(out/"protocol.json"); tasks=[json.loads(s) for s in (out/"tasks.jsonl").read_text().splitlines() if s.strip()]
 split="dev_static" if stage=="dev_static" else "test_static_locked"; expected=[t for t in tasks if t["split"]==split]
 man=readj(out/"manifest.json"); ids=[x["run_id"] for x in man["runs"]]; duplicates=sorted({x for x in ids if ids.count(x)>1})
 records=[]
 for p in (out/"runs").glob("*.json"):
  r=readj(p)
  if r.get("split")==split:records.append(r)
 lookup={(r["task_id"],float(r["beta_or_policy"])):r for r in records}
 metric=[]; channel=[]; quality=[]; audit={"split":split,"planned":len(expected)*5,"actual":len(records),"missing":[],"duplicate_run_ids":duplicates,"sha_mismatch":[],"early_stop":[],"qp_failure":[],"boundary_violation":[],"saturation":[],"pairing_complete":False}
 for t in expected:
  for b in [0.,.25,.5,.75,1.]:
   r=lookup.get((t["task_id"],b))
   if r is None:audit["missing"].append({"task_id":t["task_id"],"beta":b});continue
   rid=r["run_id"]; trace=out/"traces"/(rid+".npz")
   if not trace.exists() or sha(trace)!=r["trace_sha256"]:audit["sha_mismatch"].append(rid);continue
   if r["protocol_sha256"]!=prot["protocol_sha256"] or r["source_sha256"]!=prot["source_sha256"]:audit["sha_mismatch"].append(rid)
   a=np.load(trace); tm=a["time_substep_s"]; mask=(tm>=32)&(tm<=64+1e-9)
   if not np.all(mask):
    pass
   ref=a["reference_substep"][mask]; outv=a["output_substep"][mask]
   err=np.column_stack(((ref[:,1]-outv[:,1])/8,(ref[:,2]-outv[:,2])/5,(ref[:,5]-outv[:,5])/2))
   mse=np.mean(err**2,axis=0); rmse=np.sqrt(np.mean((np.column_stack((ref[:,1]-outv[:,1],ref[:,2]-outv[:,2],ref[:,5]-outv[:,5])))**2,axis=0)); peaks=np.max(np.abs(np.column_stack((ref[:,1]-outv[:,1],ref[:,2]-outv[:,2],ref[:,5]-outv[:,5]))),axis=0)
   j=float(mse.sum()); cmd=a["command"]/np.array([.35,.35,.35]); slew=float(np.max(np.abs(np.diff(cmd,axis=0)))) if len(cmd)>1 else 0.; act=a["actual_substep"]/np.array([.35,.35,.35]); actslew=float(np.max(np.abs(np.diff(act,axis=0)))) if len(act)>1 else 0.
   metric.append({"run_id":rid,"task_id":t["task_id"],"beta":b,"J_track":j,"mse_E_norm":float(mse[0]),"mse_D_norm":float(mse[1]),"mse_VN_norm":float(mse[2]),"rmse_E_m":float(rmse[0]),"rmse_D_m":float(rmse[1]),"rmse_VN_mps":float(rmse[2]),"peak_E_m":float(peaks[0]),"peak_D_m":float(peaks[1]),"peak_VN_mps":float(peaks[2]),"max_normalized_command_slew":slew,"max_normalized_actual_slew":actslew,"completion_status":r["completion_status"],"first_violation_time_s":r.get("first_violation_time_s"),"qp_failures":r["qp_failures"],"command_saturation":int(any(x>0 for x in r["command_saturation"])),"actual_saturation":int(any(x>0 for x in r["actual_saturation"]))})
   channel.append({"task_id":t["task_id"],"beta":b,"rmse_E_m":rmse[0],"rmse_D_m":rmse[1],"rmse_VN_mps":rmse[2],"peak_E_m":peaks[0],"peak_D_m":peaks[1],"peak_VN_mps":peaks[2]})
   if r["completion_status"] not in ("complete_valid","complete_constrained"):audit["early_stop"].append(rid)
   if r["qp_failures"]:audit["qp_failure"].append(rid)
   if r.get("first_violation_time_s") is not None:audit["boundary_violation"].append(rid)
   if int(any(x>0 for x in r["command_saturation"])) or int(any(x>0 for x in r["actual_saturation"])):audit["saturation"].append(rid)
   quality.append({"run_id":rid,"task_id":t["task_id"],"beta":b,"completion_status":r["completion_status"],"qp_failures":r["qp_failures"],"first_violation_time_s":r.get("first_violation_time_s"),"command_saturation":r["command_saturation"],"actual_saturation":r["actual_saturation"],"max_normalized_command_slew":slew,"max_normalized_actual_slew":actslew})
 audit["pairing_complete"]=not any(audit[k] for k in ("missing","duplicate_run_ids","sha_mismatch")) and len(metric)==audit["planned"]
 (out/"integrity_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
 writecsv(out/("dev_static_metrics.csv" if split=="dev_static" else "test_static_metrics.csv"),metric);writecsv(out/"beta_channel_rmse.csv",channel);writecsv(out/"beta_qp_quality.csv",quality)
 if split=="dev_static" and audit["pairing_complete"]:
  by={(x["task_id"],x["beta"]):x for x in metric}; tasks4=[t["task_id"] for t in expected]; means={b:float(np.mean([by[(t,b)]["J_track"] for t in tasks4])) for b in [0.,.25,.5,.75,1.]}
  regret={}
  for b in means:
   vals=[]
   for t in tasks4:
    js=[by[(t,0.)]["J_track"],by[(t,1.)]["J_track"]];lo=min(js);den=max(*js,1e-12);vals.append((by[(t,b)]["J_track"]-lo)/den)
   regret[b]=max(vals)
  ends=min(regret[0.],regret[1.]); mids=[b for b in [.25,.5,.75] if regret[b]<=ends-.05 and means[b]<=1.05*min(means[0.],means[1.])]
  mids.sort(key=lambda b:(regret[b],means[b],b)); star=mids[0] if mids else None
  b_low=by[("ps_dev_depth_low",0.)]["J_track"];d_low=by[("ps_dev_depth_low",1.)]["J_track"]
  cross=(b_low<=.95*d_low and sum(by[(t,1.)]["J_track"]<=.95*by[(t,0.)]["J_track"] for t in ("ps_dev_depth_high","ps_dev_east_high","ps_dev_mixed"))>=2)
  selection={"frozen":True,"protocol_sha256":prot["protocol_sha256"],"dev_metrics_sha256":sha(out/"dev_static_metrics.csv"),"candidate_beta":star,"F_dev":min(means,key=lambda b:(means[b],b)),"Jbar_by_beta":{str(k):v for k,v in means.items()},"worst_regret_by_beta":{str(k):v for k,v in regret.items()},"G1_A_pass":star is not None,"G1_B_pass":bool(star is None and cross),"G1_C_stop":bool(star is None and not cross),"endpoint_cross":cross,"taskwise_beta_leq_both_endpoints":({str(b):all(by[(t,b)]["J_track"]<=min(by[(t,0.)]["J_track"],by[(t,1.)]["J_track"]) for t in tasks4) for b in [.25,.5,.75]})}
  (out/"selection_static.json").write_text(json.dumps(selection,indent=2)+"\n")
  # Trade-off curves and endpoint Pareto panel.
  fig,axs=plt.subplots(2,2,figsize=(10,7));
  for ax,t in zip(axs.flat,tasks4):
   ys=[by[(t,b)]["J_track"] for b in BETAS];ax.plot(BETAS,ys,"o-");ax.set_yscale("log");ax.set_title(t);ax.set_xlabel("β");ax.set_ylabel("J_track")
  fig.suptitle("Dev static predictor scan (32–64 s)");fig.tight_layout();(out/"figures").mkdir(exist_ok=True);fig.savefig(out/"figures/beta_tradeoff.png",dpi=160);plt.close(fig)
  low="ps_dev_depth_low"; high=["ps_dev_depth_high","ps_dev_east_high","ps_dev_mixed"]
  fig,ax=plt.subplots(figsize=(7,5))
  for b in BETAS:
   x=by[(low,b)]["J_track"];y=float(np.mean([by[(t,b)]["J_track"] for t in high]));ax.scatter(x,y,label=f"β={b:g}");ax.annotate(f"{b:g}",(x,y))
  ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("Low-frequency depth J");ax.set_ylabel("Mean high-frequency/mixed J");ax.legend();fig.tight_layout();fig.savefig(out/"figures/pareto_low_high.png",dpi=160);plt.close(fig)
  # Human report and decision log.
  lines=["# GS-12 预测器路径实验报告","","## 本轮开发结果","",f"完整配对：{len(metric)}/{audit['planned']}。质量异常 run：{len(audit['qp_failure'])+len(audit['boundary_violation'])+len(audit['saturation'])+len(audit['early_stop'])}。","","| β | gain | tau | 等任务平均 J | 最差端点遗憾 R |","|---:|---:|---:|---:|---:|"]
  for b in BETAS:lines.append(f"| {b:g} | {1-.15*b:.4f} | {1+b:.2f} | {means[b]:.8g} | {regret[b]:.6g} |")
  lines += ["","### 每项任务的结果","","| 任务 | β=0 J | β=.25 J | β=.5 J | β=.75 J | β=1 J |","|---|---:|---:|---:|---:|---:|"]
  for t in tasks4:lines.append("| "+t+" | "+" | ".join(f"{by[(t,b)]['J_track']:.8g}" for b in BETAS)+" |")
  lines += ["","## 决策门","",f"G1-A 中间折中候选：{star if star is not None else '无'}。G1-B 端点交叉：{'通过' if cross else '未通过'}。", "静态锁定仅在 G1-A 通过时允许；动态频谱分支仅在 G1-B 成立或后续符合协议规定条件时考虑。","","## 范围与解释","","历史两轮结果仅作为背景。此处是新开发任务上的确定性配对结果，不构成独立统计样本、实艇部署或稳定性概率结论。低频深度同时列原始 J 和 RMSE，避免只看相对百分比。","","运行完整性详见 `integrity_audit.json`；轨迹和指标位于同目录。"]
  (out/"REPORT.md").write_text("\n".join(lines)+"\n");(out/"decisions.md").write_text("G1-A: "+("pass β="+str(star) if star is not None else "no intermediate candidate")+"\nG1-B: "+("pass" if cross else "fail")+"\nG1-C: "+("stop" if star is None and not cross else "not applicable")+"\n")
 print(f"stage={split}; planned={audit['planned']}; actual={len(metric)}; complete_pairs={audit['pairing_complete']}")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=["dev_static","test_static_locked"]);p.add_argument("--out",default="experiments/gs12_predictor_route/output");a=p.parse_args();analyze(a.stage,a.out)
