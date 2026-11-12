#!/usr/bin/env python3
"""Integrity audit and scoring for GS-12 joint route runs."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(HERE))
from run_joint import CONTROLLERS, TASKS, canonical, digest, sha, source_hashes

def write_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def analyze(out,stage):
    protocol=json.loads((out/"protocol.json").read_text())
    tasks=[json.loads(x) for x in (out/"tasks.jsonl").read_text().splitlines() if x.strip()]
    tasks=[t for t in tasks if t["split"]==stage]
    expected_cids=(list(CONTROLLERS) if stage=="dev" else ["A","B","C",json.loads((out/"selection.json").read_text())["candidate_id"]])
    records=[]; bad_sha=[]; missing=[]; duplicate=[]; early=[]; qpf=0; violations=0; sat=0
    seen={}
    for p in sorted((out/"runs").glob("*.json")):
        rec=json.loads(p.read_text())
        if rec.get("split")!=stage: continue
        key=(rec["task_id"],rec["controller_id"])
        if key in seen: duplicate.append(key)
        seen[key]=rec
        trace=out/"traces"/(rec["run_id"]+".npz")
        if not trace.exists() or sha(trace)!=rec.get("trace_sha256") or rec.get("protocol_sha256")!=protocol["protocol_sha256"] or rec.get("source_sha256")!=protocol["source_sha256"]:
            bad_sha.append(rec["run_id"]); continue
        if rec.get("completion_status","").startswith("early") or rec.get("duration_observed_s",0)<64.-.02: early.append(rec["run_id"])
        qpf+=int(rec["qp_failures"]>0); violations+=int(rec.get("first_violation_time_s") is not None)
        sat+=int(any(x>0 for x in rec["command_saturation"]+rec["actual_saturation"]))
        with np.load(trace) as z:
            t=z["time_substep_s"]; ref=z["reference_substep"]; outv=z["output_substep"]
            err=np.column_stack(((ref[:,1]-outv[:,1])/8.,(ref[:,2]-outv[:,2])/5.,(ref[:,5]-outv[:,5])/2.))
            mask=(t>=32.)&(t<=64.)
            row={"run_id":rec["run_id"],"task_id":rec["task_id"],"controller_id":rec["controller_id"],"predictor_id":rec["predictor_id"],"weight_id":rec["weight_id"],"completion_status":rec["completion_status"],"duration_observed_s":rec["duration_observed_s"],"window_s":"32-64"}
            row["J_track"]=float(np.mean(np.sum(err[mask]**2,axis=1))) if np.any(mask) else float("nan")
            names=["E","D","VN"]
            units=["m","m","mps"]
            for j,(name,unit) in enumerate(zip(names,units)):
                raw=(ref[:,1]-outv[:,1]) if j==0 else ((ref[:,2]-outv[:,2]) if j==1 else (ref[:,5]-outv[:,5]))
                row[f"RMSE_{name}"]=float(np.sqrt(np.mean(raw[mask]**2))) if np.any(mask) else float("nan")
                row[f"peak_{name}"]=float(np.max(np.abs(raw[mask]))) if np.any(mask) else float("nan")
            from experiments.gs12_weight_route.runner_core import SCALE
            cmds=z["command"]/SCALE
            row["max_norm_command_step"]=float(np.max(np.abs(np.diff(cmds,axis=0)))) if len(cmds)>1 else 0.
            row["max_actual_fin_step"]=float(np.max(np.abs(np.diff(z["actual_substep"][:,1:],axis=0)))) if len(z["actual_substep"])>1 else 0.
            row.update({"qp_failures":rec["qp_failures"],"first_violation_time_s":rec.get("first_violation_time_s"),"command_saturation_any":int(any(x>0 for x in rec["command_saturation"])),"actual_saturation_any":int(any(x>0 for x in rec["actual_saturation"])),"solve_p50_ms":rec["solve_p50_ms"],"solve_p95_ms":rec["solve_p95_ms"],"solve_p99_ms":rec["solve_p99_ms"]})
            records.append(row)
    for t in tasks:
        for c in expected_cids:
            if (t["task_id"],c) not in seen: missing.append((t["task_id"],c))
    # Report extra entries and content failures; never rank an incomplete matched set.
    complete=not missing and not duplicate and not bad_sha and not early and len(records)==len(tasks)*len(expected_cids)
    metrics={ (r["task_id"],r["controller_id"]):r for r in records }
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/("dev_metrics.csv" if stage=="dev" else "new_locked_metrics.csv"),records)
    quality=[{k:r[k] for k in ["run_id","task_id","controller_id","completion_status","duration_observed_s","qp_failures","first_violation_time_s","command_saturation_any","actual_saturation_any","max_norm_command_step","max_actual_fin_step","solve_p50_ms","solve_p95_ms","solve_p99_ms"]} for r in records]
    if stage=="dev": write_csv(out/"dev_quality.csv",quality)
    audit={"stage":stage,"protocol_sha256":protocol["protocol_sha256"],"planned_runs":len(tasks)*len(expected_cids),"actual_records":len(records),"complete":complete,"missing":[list(x) for x in missing],"duplicate_keys":[list(x) for x in duplicate],"sha_or_protocol_mismatch":bad_sha,"early_or_short":early,"runs_with_qp_failures":qpf,"runs_with_boundary_violations":violations,"runs_with_saturation":sat}
    (out/"integrity_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    if complete:
        if stage=="dev":
            effects=[]
            for t in tasks:
                a,b,c,d=[metrics[(t["task_id"],x)]["J_track"] for x in "ABCD"]
                effects.append({"task_id":t["task_id"],"A":a,"B":b,"C":c,"D":d,"W1_on_P0_B_minus_A":b-a,"P1_on_W0_C_minus_A":c-a,"W1_on_P1_D_minus_C":d-c,"interaction_I":d-c-b+a,"D_minus_best_ABC":d-min(a,b,c),"D_over_B":d/b})
            effects.append({"task_id":"TASK_EQUAL_MEAN","A":np.mean([x["A"] for x in effects]),"B":np.mean([x["B"] for x in effects]),"C":np.mean([x["C"] for x in effects]),"D":np.mean([x["D"] for x in effects]),"W1_on_P0_B_minus_A":np.mean([x["W1_on_P0_B_minus_A"] for x in effects]),"P1_on_W0_C_minus_A":np.mean([x["P1_on_W0_C_minus_A"] for x in effects]),"W1_on_P1_D_minus_C":np.mean([x["W1_on_P1_D_minus_C"] for x in effects]),"interaction_I":np.mean([x["interaction_I"] for x in effects]),"D_minus_best_ABC":np.mean([x["D_minus_best_ABC"] for x in effects]),"D_over_B":np.mean([x["D"] for x in effects])/np.mean([x["B"] for x in effects])})
            write_csv(out/"factor_effects.csv",effects)
            make_figures(out,records,effects)
            dmean=effects[-1]["D"]; bmean=effects[-1]["B"]
            dvals=[x["D"] for x in effects[:-1]]; bvals=[x["B"] for x in effects[:-1]]
            blocked=any(metrics[(t["task_id"],"D")]["qp_failures"] or metrics[(t["task_id"],"D")]["first_violation_time_s"] is not None or metrics[(t["task_id"],"D")]["command_saturation_any"] or metrics[(t["task_id"],"D")]["actual_saturation_any"] for t in tasks)
            d_ok=dmean<bmean and sum(d<=b for d,b in zip(dvals,bvals))>=3 and not blocked
            recommendation="D" if d_ok else "NONE"
            decisions=(f"# M1 决策记录\n\n- 开发集完整：是（{len(records)}/{len(tasks)*len(expected_cids)} 条）。\n- 等权平均 J：B={bmean:.9g}, D={dmean:.9g}；D 相对 B 改善={100*(1-dmean/bmean):.3f}%。\n- D 不劣任务数：{sum(d<=b for d,b in zip(dvals,bvals))}/{len(tasks)}。\n- D 质量阻断：{'是' if blocked else '否'}。\n- 按预注册第 6 节第 1 条自动建议：{recommendation}。\n- 补充开发集：未运行；本轮不追加搜索。\n- 依据：`factor_effects.csv`、`dev_metrics.csv`、`dev_quality.csv` 和协议 SHA `{protocol['protocol_sha256']}`。\n")
            (out/"decisions.md").write_text(decisions)
        else:
            selection=json.loads((out/"selection.json").read_text()); cand=selection["candidate_id"]
            cm=np.mean([metrics[(t["task_id"],cand)]["J_track"] for t in tasks]); bm=np.mean([metrics[(t["task_id"],"B")]["J_track"] for t in tasks]); pct=100*(1-cm/bm)
            per=[(metrics[(t["task_id"],cand)]["J_track"],metrics[(t["task_id"],"B")]["J_track"]) for t in tasks]
            blocked=any(metrics[(t["task_id"],cand)][k] for t in tasks for k in ("qp_failures","command_saturation_any","actual_saturation_any")) or any(metrics[(t["task_id"],cand)]["first_violation_time_s"] is not None for t in tasks)
            passed=pct>=5 and all(c<=b for c,b in per) and not blocked
            audit.update({"candidate_id":cand,"candidate_vs_B_improvement_pct":pct,"candidate_nonworse_tasks":sum(c<=b for c,b in per),"strict_gate_passed":passed})
            (out/"integrity_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))
    return records,audit

def make_figures(out,records,effects):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception: return
    figdir=out/"figures"; figdir.mkdir(exist_ok=True)
    tasks=list(dict.fromkeys(r["task_id"] for r in records)); ids=list("ABCD")
    fig,ax=plt.subplots(figsize=(8,4.8))
    for cid in ids: ax.plot(tasks,[next(r["J_track"] for r in records if r["task_id"]==t and r["controller_id"]==cid) for t in tasks],"o-",label=cid)
    ax.set_ylabel("J_track, 32–64 s"); ax.set_title("Dev factor grid (task paired; B is fixed-weight baseline)"); ax.tick_params(axis="x",rotation=20); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"dev_factor_grid.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(tasks,[e["interaction_I"] for e in effects[:-1]]); ax.axhline(0,color="black",lw=.8); ax.set_title("Dev predictor × weight interaction I"); ax.set_ylabel("J interaction"); fig.tight_layout(); fig.savefig(figdir/"dev_interaction.png",dpi=160); plt.close(fig)
    mixed=next((r for r in records if r["task_id"]=="dev_mixed_switch"),None)
    if mixed:
        # Trace overlay requires reading records by stable run IDs.
        fig,ax=plt.subplots(figsize=(9,4.5))
        for cid in ids:
            rid=next(r["run_id"] for r in records if r["task_id"]=="dev_mixed_switch" and r["controller_id"]==cid)
            z=np.load(out/"traces"/(rid+".npz")); t=z["time_substep_s"]; ax.plot(t,z["output_substep"][:,2],label=f"{cid} actual");
            if cid=="A": ax.plot(t,z["reference_substep"][:,2],"k--",label="reference")
        ax.set_xlim(0,64); ax.set_xlabel("time (s)"); ax.set_ylabel("depth (m)"); ax.set_title("Dev mixed switch trace; scoring window 32–64 s; baseline B"); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(figdir/"dev_mixed_trace.png",dpi=160); plt.close(fig)

def freeze(out,candidate):
    decisions=out/"decisions.md"; audit=json.loads((out/"integrity_audit.json").read_text())
    if not decisions.exists() or audit.get("stage")!="dev" or not audit.get("complete"): raise SystemExit("need complete dev analysis and decisions.md before freeze")
    if candidate not in CONTROLLERS and candidate!="NONE": raise SystemExit("candidate must be registered or NONE")
    if candidate!="NONE" and (candidate!="D" or "自动建议：D" not in decisions.read_text()):
        raise SystemExit("automatic M2 freeze requires the dev decision record to recommend D")
    protocol=json.loads((out/"protocol.json").read_text())
    records=[json.loads(p.read_text()) for p in (out/"runs").glob("*.json") if json.loads(p.read_text()).get("split")=="dev"]
    sel={"candidate_id":None if candidate=="NONE" else candidate,"parameters":None if candidate=="NONE" else CONTROLLERS[candidate],"frozen":candidate!="NONE","frozen_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"protocol_sha256":protocol["protocol_sha256"],"decisions_sha256":sha(decisions),"dev_run_ids":[r["run_id"] for r in records],"dev_metrics_sha256":sha(out/"dev_metrics.csv"),"dev_factor_effects_sha256":sha(out/"factor_effects.csv"),"constraint_or_channel_losses_observed":True if "质量阻断：是" in decisions.read_text() else False}
    (out/"selection.json").write_text(json.dumps(sel,indent=2,ensure_ascii=False)+"\n"); print(json.dumps(sel,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--stage",choices=["dev","new_locked"],required=True); ap.add_argument("--out",default="experiments/gs12_joint_route/output"); a=ap.parse_args(); analyze(Path(a.out).resolve(),a.stage)
if __name__=="__main__": main()
