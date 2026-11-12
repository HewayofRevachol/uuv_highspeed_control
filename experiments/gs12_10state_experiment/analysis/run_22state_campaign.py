#!/usr/bin/env python3
"""Run the preregistered 22-state comparator campaign in this experiment folder."""
from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, os, platform, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.update({"OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"})
HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "src"
ANALYSIS = HERE.parent / "gs12_analaysis"
OUT_DEFAULT = HERE / "output" / "paired22_campaign"
SCALE = [8., 5., 2.]
STRAIGHT_CONDITIONS = {
    "C0": {"plant": {}, "gain": 1., "tau": 1.},
    "C1": {"plant": {"elevator": .9, "rudder": .9}, "gain": 1., "tau": 1.},
    "C2": {"plant": {"tau_scale": [2., 2., 2.]}, "gain": 1., "tau": 1.},
    "C3": {"plant": {"order": "second_all"}, "gain": 1., "tau": 1.},
    "C4": {"plant": {"elevator": .9, "rudder": .9, "order": "second_all"}, "gain": 1., "tau": 1.},
    "C5": {"plant": {"elevator": .9, "rudder": .9, "order": "second_all"}, "gain": .85, "tau": 2.},
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def sha_bytes(data): return hashlib.sha256(data).hexdigest()

def write_json(path, doc):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+"\n")
    tmp.replace(path)

def sources(repo):
    root=Path(repo)
    names=[HERE/"analysis"/"run_22state_campaign.py", SRC/"reference10.py",
           ANALYSIS/"frequency_extension_src"/"plant_harness.py",
           root/"simulator"/"batch_mpc.py", root/"simulator"/"model.py",
           root/"simulator"/"rti_symbolic.py", root/"aav_mpc"/"condensed_mpc.py",
           root/"aav_mpc"/"h15_target_domain.py"]
    return {str(p.relative_to(root)):sha(p) for p in names}

def protocol(repo, main10):
    packages={}
    for name in ("numpy","torch","casadi","osqp","scipy"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    return {"experiment":"GS-12 22-state preregistered exact-reference campaign",
            "controller_state_dim":22,"main_10_protocol_sha256":sha(main10),
            "source_sha256":sources(repo),"packages":packages,"platform":platform.platform(),
            "duration_s":256.,"control_dt_s":.1,"plant_substep_s":.02,
            "straight_conditions":STRAIGHT_CONDITIONS,
            "new_trajectory_conditions":["15 m/s C0/C4 left/right turn, +/-3 degree slope, periodic S-bend/coupled"],
            "reference_source":"src/reference10.py; exact same deterministic reference generator as 10-state campaign",
            "planned_straight_runs":333,"planned_new_trajectory_runs":156}

def jobs():
    out=[]
    def add(speed, cond, kind="straight", base=None, band=None, axis=-1, amp=0., phase="primary"):
        out.append(dict(speed=speed, condition=cond, kind=kind, base=base or {}, band=band,
                        axis=axis, amp=amp, phase=phase))
    def full(speed, cond):
        add(speed,cond)
        for b in ("B1","B2","B3","B4"):
            for ax in range(3): add(speed,cond,band=b,axis=ax,amp=.006)
    for c in STRAIGHT_CONDITIONS: full(15.,c)
    for c in STRAIGHT_CONDITIONS:
        for b in ("B1","B2","B3","B4"):
            for ax in range(3): add(15.,c,band=b,axis=ax,amp=.003)
    for c in ("C0","C4","C5"):
        for b in ("B1","B3","B4"):
            for ax in range(3): add(15.,c,band=b,axis=ax,amp=.006,phase="secondary")
    for speed in (10.,8.,12.,18.):
        for c in ("C0","C4","C5"): full(speed,c)
    for c in ("C0","C4"):
        for kappa,label in ((1/120.,"_left"),(-1/120.,"_right")):
            base={"curvature":kappa,"label":label};add(15.,c,"turn",base)
            for b in ("B1","B2","B3","B4"):
                for ax in range(3):add(15.,c,"turn",base,b,ax,.006)
        for gamma,label in ((-__import__("numpy").deg2rad(3.),"_climb"),(__import__("numpy").deg2rad(3.),"_dive")):
            base={"gamma":float(gamma),"label":label};add(15.,c,"slope",base)
            for b in ("B1","B2","B3","B4"):
                for ax in range(3):add(15.,c,"slope",base,b,ax,.006)
        for kind in ("s_bend","coupled"):
            base={"label":"_periodic"};add(15.,c,kind,base)
            for k in (6,10,18,30):
                for ax in range(3):add(15.,c,kind,{"probe_k":k,"label":"_periodic"},axis=ax,amp=.006)
    return out

def worker(task):
    repo=Path(task["repo"]); out=Path(task["out"]); p=task["job"]
    sys.path[:0]=[str(repo),str(SRC),str(ANALYSIS/"frequency_extension_src")]
    os.environ["GS12_REPO"]=str(repo)
    import torch; torch.set_num_threads(1)
    import plant_harness
    from plant_harness import PlantConfig, run_closed_loop, SCALE as PH_SCALE
    from reference10 import straight,turn,slope,periodic,turn_errors,coordinate_errors,multi_signal,_probe
    from simulator.rti_symbolic import ActuatorAwareCpuDynamics
    if p["kind"]=="straight": ref=straight(p["speed"],p["band"],p["axis"],p["amp"],p["phase"])
    elif p["kind"]=="turn": ref=turn(p["speed"],p["base"]["curvature"],p["band"],p["axis"],p["amp"],p["phase"])
    elif p["kind"]=="slope": ref=slope(p["speed"],p["base"]["gamma"],p["band"],p["axis"],p["amp"],p["phase"])
    else: ref=periodic(p["speed"],p["kind"],p["base"].get("probe_k"),p["axis"],p["amp"])
    baseline=periodic(p["speed"],p["kind"]) if p["kind"] in ("s_bend","coupled") else None
    cond=STRAIGHT_CONDITIONS[p["condition"]] if p["kind"]=="straight" else {"plant":({} if p["condition"]=="C0" else {"elevator":.9,"rudder":.9,"order":"second_all"}),"gain":1.,"tau":1.}
    cfg=PlantConfig(**cond["plant"])
    original_plant=plant_harness.Plant
    if p["kind"]=="slope":
        gamma=p["base"]["gamma"]
        class InitializedPlant(original_plant):
            def __init__(self,speed,config):
                super().__init__(speed,config); self.x[2]=280.; self.x[3]=-gamma
        plant_harness.Plant=InitializedPlant
    setup=None
    if cond["gain"]!=1. or cond["tau"]!=1.:
        setup=lambda m:setattr(m,"dynamics",ActuatorAwareCpuDynamics(.1,cond["gain"],cond["tau"]))
    try: result=run_closed_loop(cfg,p["speed"],ref,256.,1.,1.,setup,True)
    finally: plant_harness.Plant=original_plant
    meta,ar=result["meta"],result["arrays"]
    idx=__import__("numpy").arange(4,len(ar["time"]),5);tc=ar["time"][idx]
    import numpy as np
    if len(tc) and not np.allclose(tc,np.arange(1,len(tc)+1)*.1,rtol=0,atol=1e-10): raise ValueError("misaligned control samples")
    refs=ar["reference"][idx];outs=ar["output"][idx]
    eglobal=np.column_stack(((refs[:,1]-outs[:,1])/8.,(refs[:,2]-outs[:,2])/5.,(refs[:,5]-outs[:,5])/2.))
    epath=eglobal.copy()
    if p["kind"]=="turn":
        epath=np.stack([turn_errors(t,r,y,p["speed"],p["base"]["curvature"]) for t,r,y in zip(tc,refs,outs)])
    elif p["kind"]=="slope":
        epath=np.stack([coordinate_errors(r,y,"slope",gamma=p["base"]["gamma"]) for r,y in zip(refs,outs)])
    elif p["kind"] in ("s_bend","coupled"):
        epath=[]
        for t,r,y in zip(tc,refs,outs):
            rb=baseline(t);yaw=rb[4];pitch=rb[3]
            tangent=np.array([np.cos(yaw)*np.cos(pitch),np.sin(yaw)*np.cos(pitch),-np.sin(pitch)])
            normal=np.array([-np.sin(yaw),np.cos(yaw),0.]);dp=r[:3]-y[:3];dv=r[5:8]-y[5:8]
            epath.append([normal@dp/8.,dp[2]/5.,tangent@dv/2.])
        epath=np.asarray(epath)
    zin=np.zeros((len(tc),3))
    if p["axis"]>=0 and p["band"]:
        zin[:,p["axis"]]=multi_signal(tc,p["band"],p["amp"],p["phase"])[0]
    elif p["axis"]>=0 and p["kind"] in ("s_bend","coupled"):
        zin[:,p["axis"]]=_probe(tc,[p["base"]["probe_k"]],[0.],p["amp"],True)[0]
    preft=np.arange(0.,256.+3.5+1e-9,.02)
    refsha=sha_bytes(np.asarray(ref(preft),dtype="<f8").tobytes())
    psha=sha(out/"protocol.json")
    label=p["base"].get("label","");tag=p["band"] or (f'k{p["base"]["probe_k"]}' if "probe_k" in p["base"] else "zero")
    rid=f'p{psha[:12]}_model22_v{p["speed"]:g}_{p["condition"]}_{p["kind"]}{label}_{tag}_axis{p["axis"]}_a{p["amp"]:g}_{p["phase"]}'
    trace_path=out/"traces"/(rid+".npz");run_path=out/"runs"/(rid+".json")
    trace={"time_substep_s":ar["time"],"state_substep":ar["state"],"reference_substep":ar["reference"],
           "output_substep":ar["output"],"error_substep":ar["error"],"diagnostic_substep":ar["diagnostic"],
           "actual_substep":ar["actual"],"queue_substep":ar["queue"],"command_substep":ar["subcommand"],
           "time_control_s":tc,"reference_control":refs,"output_control":outs,"error_normalized":eglobal,
           "error_path_normalized":epath,"input_normalized":zin,"command":ar["command"],
           "qp_ok":ar["qp_ok"],"qp_primal":ar["qp_primal"],"qp_dual":ar["qp_dual"],"one_step_error":ar["one_step_error"]}
    tmp=trace_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f: np.savez_compressed(f,**trace)
    tmp.replace(trace_path)
    if meta["first_diagnostic_limit_violation"]:status="early_violation"
    elif meta["qp_failures"]:status="qp_failure"
    elif len(ar["command"])!=2560:status="incomplete"
    elif any(v>0 for v in meta["command_saturation_fraction"]+meta["actual_saturation_fraction_substeps"]):status="complete_constrained"
    else:status="complete_valid"
    rec={"run_id":rid,"protocol_sha256":psha,"source_sha256":sources(repo),"reference_sha256":refsha,
         "controller_state_dim":22,"controller":"CondensedActuatorMPC","speed_mps":p["speed"],
         "condition":p["condition"],"operating_regime":p["kind"],"regime_parameters":p["base"],
         "plant_config":meta["config"],"band":p["band"],"probe_k":p["base"].get("probe_k"),"axis":p["axis"],
         "amplitude_per_line":p["amp"],"phase_id":p["phase"],"status":status,
         "observed_duration_s":meta["duration_observed_s"],"control_periods":len(ar["command"]),"substeps":len(ar["time"]),
         "first_violation":meta["first_diagnostic_limit_violation"],"qp_failures":meta["qp_failures"],
         "command_saturation_fraction":meta["command_saturation_fraction"],
         "actual_saturation_fraction":meta["actual_saturation_fraction_substeps"],
         "trace_sha256":sha(trace_path),"wall_s":meta["wall_s_not_realtime_benchmark"]}
    write_json(run_path,rec)
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--repo",type=Path,required=True);ap.add_argument("--out",type=Path,default=OUT_DEFAULT);ap.add_argument("--workers",type=int,default=4);ap.add_argument("--limit",type=int)
    a=ap.parse_args();repo=a.repo.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    for d in ("runs","traces","matrices"): (out/d).mkdir(exist_ok=True)
    main10=HERE/"output"/"protocol.json";doc=protocol(repo,main10);pp=out/"protocol.json"
    if pp.exists() and json.loads(pp.read_text())!=doc:raise FileExistsError("22-state campaign protocol differs")
    if not pp.exists():write_json(pp,doc)
    alljobs=jobs(); selected=alljobs[:a.limit] if a.limit else alljobs
    manifest_path=out/"manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs":[],"planned_runs":len(alljobs)}
    known={x["run_id"] for x in manifest["runs"]}
    pending=[]
    for ordinal,j in enumerate(selected,1):
        base=j["base"];tag=j["band"] or (f'k{base["probe_k"]}' if "probe_k" in base else "zero")
        expected=f'p{sha(pp)[:12]}_model22_v{j["speed"]:g}_{j["condition"]}_{j["kind"]}{base.get("label","")}_{tag}_axis{j["axis"]}_a{j["amp"]:g}_{j["phase"]}'
        if expected not in known:pending.append((ordinal,j))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(worker,{"repo":str(repo),"out":str(out),"job":j}):i for i,j in pending}
        for fut in as_completed(futures):
            rec=fut.result();manifest["runs"].append({"run_id":rec["run_id"],"status":rec["status"],"ordinal":futures[fut],"trace_sha256":rec["trace_sha256"]})
            write_json(manifest_path,manifest)
            print(json.dumps({"done":len(manifest["runs"]),"planned":len(selected),"status":rec["status"],"run_id":rec["run_id"],"wall_s":rec["wall_s"]}),flush=True)
    print(json.dumps({"completed":len(manifest["runs"]),"planned":len(alljobs),"output":str(out)}))

if __name__=="__main__":main()
