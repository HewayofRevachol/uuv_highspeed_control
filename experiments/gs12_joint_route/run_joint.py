#!/usr/bin/env python3
"""Prepare and execute the registered GS-12 joint predictor/weight runs."""
from __future__ import annotations

import argparse, concurrent.futures, hashlib, json, os, platform, sys, time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "gs12_weight_route"))

import numpy as np
from joint_reference import joint_task_reference, derivative_error
from experiments.gs12_weight_route.runner_core import run_one

TASKS = [
 {"split":"dev","task_id":"dev_depth_b1","kind":"single","axis":"depth","k":8,"phase_rad":.43},
 {"split":"dev","task_id":"dev_depth_b4","kind":"single","axis":"depth","k":32,"phase_rad":-.61},
 {"split":"dev","task_id":"dev_east_b4","kind":"single","axis":"east","k":28,"phase_rad":1.37},
 {"split":"dev","task_id":"dev_mixed_switch","kind":"mixed","switch_times_s":[21.,41.],"phase_rad":-.42},
 {"split":"new_locked","task_id":"locked_depth_b1","kind":"single","axis":"depth","k":4,"phase_rad":1.83},
 {"split":"new_locked","task_id":"locked_east_b4","kind":"single","axis":"east","k":26,"phase_rad":-.94},
 {"split":"new_locked","task_id":"locked_mixed_switch","kind":"mixed","switch_times_s":[24.,44.],"phase_rad":2.27},
]
for _task in TASKS:
    _task.update(duration_s=64., speed_mps=15., amplitude=.006, window_s=[32.,64.])
CONTROLLERS = {
 "A":{"plant_id":"C4","predictor_id":"P0","model_gain":1.,"model_tau":1.,"weight_id":"W0","a_depth":1.,"a_rudder_rate":1.},
 "B":{"plant_id":"C4","predictor_id":"P0","model_gain":1.,"model_tau":1.,"weight_id":"W1","a_depth":1.4,"a_rudder_rate":.7},
 "C":{"plant_id":"C4","predictor_id":"P1","model_gain":.85,"model_tau":2.,"weight_id":"W0","a_depth":1.,"a_rudder_rate":1.},
 "D":{"plant_id":"C4","predictor_id":"P1","model_gain":.85,"model_tau":2.,"weight_id":"W1","a_depth":1.4,"a_rudder_rate":.7},
}
OPTIONAL = {"pmid":{"E":{"predictor_id":"Pmid","model_gain":.925,"model_tau":1.5,"weight_id":"W1","a_depth":1.4,"a_rudder_rate":.7}},
 "pmid_w0":{"E0":{"predictor_id":"Pmid","model_gain":.925,"model_tau":1.5,"weight_id":"W0","a_depth":1.,"a_rudder_rate":1.}},
 "p1_weight_cross":{
  "F":{"predictor_id":"P1","model_gain":.85,"model_tau":2.,"weight_id":"F","a_depth":1.4,"a_rudder_rate":1.},
  "G":{"predictor_id":"P1","model_gain":.85,"model_tau":2.,"weight_id":"G","a_depth":1.,"a_rudder_rate":.7},
  "H":{"predictor_id":"P1","model_gain":.85,"model_tau":2.,"weight_id":"H","a_depth":.7,"a_rudder_rate":1.}},}
SOURCE_FILES = [HERE/"joint_reference.py", HERE/"run_joint.py", HERE/"analyze_joint.py",
                ROOT/"experiments/gs12_weight_route/runner_core.py",
                ROOT/"experiments/gs12_weight_route/reference_tasks.py",
                ROOT/"aav_mpc/condensed_mpc.py"]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=False)
def digest(obj): return hashlib.sha256(canonical(obj).encode()).hexdigest()
def outdir(s): return Path(s).resolve()
def source_hashes(): return {str(p.relative_to(ROOT)):sha(p) for p in SOURCE_FILES}
def tasks_path(out): return out/"tasks.jsonl"
def read_tasks(out): return [json.loads(x) for x in tasks_path(out).read_text().splitlines() if x.strip()]
def read_protocol(out): return json.loads((out/"protocol.json").read_text())

def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/"protocol.json").exists() or tasks_path(out).exists():
        raise SystemExit("protocol/tasks already exist; refusing to rewrite frozen inputs")
    tasks=[]
    for t in TASKS:
        t=dict(t); tasks.append(t)
    protocol={"protocol_id":"GS12-joint-route-v1","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
      "git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"source_sha256":source_hashes(),
      "python":sys.version.split()[0],"platform":platform.platform(),"plant":{"id":"C4","elevator":.9,"rudder":.9,"order":"second_all"},
      "mpc":{"states":22,"dt_s":.1,"horizon":35,"plant_substeps_per_cycle":5,"substep_s":.02},
      "duration_s":64.,"speed_mps":15.,"amplitude":.006,"window_s":[32.,64.],
      "controllers":CONTROLLERS,"optional_dev_candidates":OPTIONAL,
      "tasks_sha256":None,"selection_gate":{"candidate_vs_B_improvement_pct":5.,"all_tasks_nonworse":True,"no_quality_blockers":True}}
    protocol["tasks_sha256"]=hashlib.sha256(("\n".join(canonical(t) for t in tasks)+"\n").encode()).hexdigest()
    (tasks_path(out)).write_text("".join(canonical(t)+"\n" for t in tasks))
    protocol["protocol_sha256"]=digest(protocol)
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+"\n")
    (out/"controller_matrix.json").write_text(json.dumps(CONTROLLERS,indent=2)+"\n")
    (out/"manifest.json").write_text(json.dumps({"protocol_sha256":protocol["protocol_sha256"],"runs":[]},indent=2)+"\n")
    print(protocol["protocol_sha256"])

def verify(out):
    smoke=out/"smoke"; smoke.mkdir(parents=True,exist_ok=True)
    # Numerical position/velocity check including mixed startup and both switch neighborhoods.
    worst=max(derivative_error(t) for t in TASKS)
    if worst>=1e-3: raise SystemExit(f"reference derivative gate failed: {worst}")
    # Constructor-level acceptance: confirm predictor and both QP scales for A-D.
    from simulator.batch_mpc import BatchedUUVModel, CONTROL_SCALE
    from aav_mpc.condensed_mpc import CondensedActuatorMPC
    from simulator.rti_symbolic import ActuatorAwareCpuDynamics
    from experiments.gs12_analaysis.frequency_extension_src.plant_harness import Plant, PlantConfig
    from unittest.mock import patch
    import osqp
    model=BatchedUUVModel("cpu"); plant=Plant(15.,PlantConfig(elevator=.9,rudder=.9,order="second_all"))
    refs=joint_task_reference(TASKS[0],np.arange(1,36)*.1)
    original=osqp.OSQP.setup; objs={}
    for cid,c in CONTROLLERS.items():
        m=CondensedActuatorMPC(model,1,horizon=35,dt=.1); m.dynamics=ActuatorAwareCpuDynamics(.1,c["model_gain"],c["model_tau"])
        m.previous[:]=model.tensor(plant.actual/np.asarray(CONTROL_SCALE)); m.plan[:]=m.previous[:,None]
        m.set_tracking_weight_scales(c["a_depth"],c["a_rudder_rate"])
        if (m.dynamics.fin_gain_scale,m.dynamics.fin_tau_scale)!=(c["model_gain"],c["model_tau"]): raise AssertionError(cid+" predictor wiring")
        if (m.depth_weight_scale,m.rudder_rate_weight_scale)!=(c["a_depth"],c["a_rudder_rate"]): raise AssertionError(cid+" weight wiring")
        def cap(self,*a,**kw):
            P=kw.get("P",a[0] if a else None); q=kw.get("q",a[1] if len(a)>1 else None); objs[cid]=(P.toarray().copy(),np.asarray(q).copy()); return original(self,*a,**kw)
        with patch.object(osqp.OSQP,"setup",cap):
            m.solve(model.tensor(plant.controller_state())[None],model.tensor(refs)[None])
    if np.max(np.abs(objs["A"][0]-objs["B"][0]))<1e-8 or np.max(np.abs(objs["C"][0]-objs["D"][0]))<1e-8: raise AssertionError("weight QP wiring not observed")
    if np.max(np.abs(objs["A"][0]-objs["C"][0]))<1e-8: raise AssertionError("predictor QP dynamics not distinct")
    # D performs a real short closed-loop solve, not merely a constructor check.
    smoke_result=run_one(TASKS[0],"C4",1.4,.7,1.0,joint_task_reference,(.85,2.))
    np.savez_compressed(smoke/"D_smoke.npz",**smoke_result["arrays"])
    (smoke/"verify.json").write_text(json.dumps({"reference_derivative_max_abs":worst,"configs_checked":list(CONTROLLERS),"D_smoke_status":smoke_result["meta"]["status"],"D_smoke_steps":len(smoke_result["arrays"]["qp_ok"])},indent=2)+"\n")
    print((smoke/"verify.json").read_text())

def build_jobs(out, split, extra=None):
    tasks=[t for t in read_tasks(out) if t["split"]==split]
    matrix=dict(CONTROLLERS)
    if split=="dev" and extra:
        if extra not in OPTIONAL: raise SystemExit("unknown dev-extra set")
        matrix.update(OPTIONAL[extra])
    elif extra: raise SystemExit("dev-extra only supports dev")
    jobs=[]
    for t in tasks:
      for cid,c in matrix.items():
        job={"split":split,"task":t,"controller_id":cid,**c}
        rid=digest({"protocol":read_protocol(out)["protocol_sha256"],"job":job})[:16]
        jobs.append((rid,job))
    ids=[x[0] for x in jobs]
    if len(ids)!=len(set(ids)): raise SystemExit("duplicate run_id in batch")
    return jobs

def worker(args):
    run_id,job,out_s,protocol_sha=args
    out=Path(out_s); c=job
    result=run_one(job["task"],"C4",c["a_depth"],c["a_rudder_rate"],64.,joint_task_reference,(c["model_gain"],c["model_tau"]))
    trace=out/"traces"/(run_id+".npz"); trace.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trace,**result["arrays"])
    task=job["task"]
    record={"run_id":run_id,"protocol_sha256":protocol_sha,"task_sha256":digest(task),"git_commit":read_protocol(out)["git_commit"],"source_sha256":source_hashes(),
      "split":job["split"],"task_id":task["task_id"],"controller_id":job["controller_id"],"phase_rad":task["phase_rad"],"amplitude":task["amplitude"],"speed_mps":task["speed_mps"],"plant_id":"C4",
      "predictor_id":c["predictor_id"],"model_gain":c["model_gain"],"model_tau":c["model_tau"],"weight_id":c["weight_id"],"a_depth":c["a_depth"],"a_rudder_rate":c["a_rudder_rate"],
      "completion_status":result["meta"]["status"],"duration_observed_s":result["meta"]["duration_observed_s"],"first_violation_time_s":(result["meta"]["first_violation"] or {}).get("time_s"),"first_violation":result["meta"]["first_violation"],
      "qp_failures":result["meta"]["qp_failures"],"command_saturation":result["meta"]["command_saturation_fraction"],"actual_saturation":result["meta"]["actual_saturation_fraction"],
      "solve_p50_ms":result["meta"]["solve_p50_ms"],"solve_p95_ms":result["meta"]["solve_p95_ms"],"solve_p99_ms":result["meta"]["solve_p99_ms"],"wall_s":result["meta"]["wall_s"],"trace_sha256":sha(trace)}
    (out/"runs"/(run_id+".json")).write_text(json.dumps(record,indent=2)+"\n")
    return record

def run(out,split,workers,extra=None):
    protocol=read_protocol(out)
    if source_hashes()!=protocol["source_sha256"]: raise SystemExit("source hash differs from frozen protocol")
    if split=="new_locked":
        sel=json.loads((out/"selection.json").read_text()) if (out/"selection.json").exists() else None
        if not sel or not sel.get("frozen"): raise SystemExit("freeze candidate before accessing new_locked")
    jobs=build_jobs(out,split,extra)
    args=[(rid,j,str(out),protocol["protocol_sha256"]) for rid,j in jobs]
    (out/"runs").mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        records=list(pool.map(worker,args))
    manifest_path=out/"manifest.json"; manifest=json.loads(manifest_path.read_text())
    known={r["run_id"] for r in manifest["runs"]}
    if any(r["run_id"] in known for r in records): raise SystemExit("run_id already present; refusing overwrite")
    # Keep one manifest writer: append this finished batch in the parent only.
    manifest["runs"].extend({"run_id":r["run_id"],"split":r["split"],"task_id":r["task_id"],"predictor_id":r["predictor_id"],"weight_id":r["weight_id"]} for r in records)
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"completed {len(records)} runs for {split}" + (f" ({extra})" if extra else ""))

def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="cmd",required=True)
    for cmd in ("verify","prepare"):
      p=sub.add_parser(cmd); p.add_argument("--out",default="experiments/gs12_joint_route/output")
    p=sub.add_parser("run"); p.add_argument("split",choices=["dev","dev-extra","new_locked"]); p.add_argument("--set",choices=list(OPTIONAL)); p.add_argument("--workers",type=int,default=1,choices=[1,2,4]); p.add_argument("--out",default="experiments/gs12_joint_route/output")
    p=sub.add_parser("freeze"); p.add_argument("--candidate",required=True); p.add_argument("--out",default="experiments/gs12_joint_route/output")
    a=ap.parse_args(); out=outdir(a.out)
    if a.cmd=="verify": verify(out)
    elif a.cmd=="prepare": prepare(out)
    elif a.cmd=="freeze":
        from analyze_joint import freeze
        freeze(out,a.candidate)
    elif a.cmd=="run":
        split="dev" if a.split=="dev-extra" else a.split
        if a.split=="dev-extra" and not a.set: raise SystemExit("dev-extra requires --set")
        run(out,split,a.workers,a.set if a.split=="dev-extra" else None)

if __name__=="__main__": main()
