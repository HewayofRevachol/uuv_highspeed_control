#!/usr/bin/env python3
"""Reproducible GS-12 ten-state experiment runner."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import argparse, concurrent.futures, hashlib, importlib.metadata, json, math, os as _os, platform, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent
ANALYSIS=EXPERIMENT.parent/"gs12_analaysis"
PLAN=Path(__file__).resolve().parents[2]/"gs12_analaysis"/"GS-12_10状态局部频响与典型工况实验方案.md"
REPO=Path(__file__).resolve().parents[3]
DEFAULT_OUT=EXPERIMENT/"output"
BANDS={f"B{i+1}":list(range(8*i+1,8*i+9)) for i in range(4)}
CONDITIONS={
 "C0":{"plant":{},"fin_gain":1.},
 "C1":{"plant":{"elevator":.9,"rudder":.9},"fin_gain":1.},
 "C2":{"plant":{"tau_scale":(2.,2.,2.)},"fin_gain":1.},
 "C3":{"plant":{"order":"second_all"},"fin_gain":1.},
 "C4":{"plant":{"elevator":.9,"rudder":.9,"order":"second_all"},"fin_gain":1.},
 "C5g":{"plant":{"elevator":.9,"rudder":.9,"order":"second_all"},"fin_gain":.85}}

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def source_hash(): return {p.name:sha(p) for p in sorted(HERE.glob("*.py"))}
def _json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n");tmp.replace(path)

def make_protocol(repo,out):
 from reference10 import T0
 versions={}
 for n in ("numpy","torch","casadi","osqp","scipy","matplotlib"):
  try: versions[n]=importlib.metadata.version(n)
  except importlib.metadata.PackageNotFoundError: versions[n]=None
 try: commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
 except Exception: commit=None
 zips={}
 for f in (EXPERIMENT.parent/"gs12_analaysis"/"GS-12_report_code.zip",EXPERIMENT.parent/"gs12_analaysis"/"GS-12_frequency_data.zip"):
  zips[f.name]=sha(f) if f.exists() else None
 return {"protocol_version":1,"plan_path":str(PLAN.relative_to(repo)),"plan_sha256":sha(PLAN),"repo_commit":commit,
  "created_local":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"python":sys.version,"packages":versions,
  "platform":platform.platform(),"cpu_count_logical":_os.cpu_count(),
  "thread_limits":{k:_os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
  "source_sha256":source_hash(),"input_zip_sha256":zips,"duration_s":256.,"pilot_duration_s":64.,"control_dt_s":.1,"plant_substep_s":.02,
  "horizon":35,"sqp_iterations":1,"osqp":{"eps_abs":1e-5,"eps_rel":1e-5,"max_iter":10000,"polishing":False},
  "frequency":{"T0_s":T0,"bins":list(range(1,33)),"omega_radps":(2*np.pi*np.arange(1,33)/T0).tolist(),"bands":BANDS},
  "amplitudes":[.006,.003],"phases":{"primary":"-pi*i*(i-1)/8","secondary":"default_rng(9512+b)"},
  "windows_s":[[64,128],[128,192],[192,256]],"normalization":{"lateral_m":8.,"depth_m":5.,"speed_mps":2.},
  "conditions":CONDITIONS,"reference_modes":["straight","turn","slope","s_bend","coupled"],
  "quality":{"W3_vs_W2_relative_frobenius_max":.02,"half_amplitude_max":.05,"diagnostic_limits":[8.,5.,2.,float(np.deg2rad(40))]}}

def verify_inputs(args):
 repo=args.repo.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
 results={"plan_sha256":sha(PLAN),"zip_checks":{},"repository":str(repo)}
 for name in ("GS-12_report_code.zip","GS-12_frequency_data.zip"):
  p=EXPERIMENT.parent/"gs12_analaysis"/name
  if p.exists():
   with zipfile.ZipFile(p) as z: bad=z.testzip(); count=len(z.namelist())
   if bad: raise ValueError(f"ZIP CRC error in {name}: {bad}")
   results["zip_checks"][name]={"sha256":sha(p),"members":count,"crc":"passed"}
  else: results["zip_checks"][name]={"available":False}
 protocol=make_protocol(repo,out); pp=out/"protocol.json"
 if pp.exists():
  old=json.loads(pp.read_text())
  if old.get("source_sha256")!=protocol["source_sha256"] or old.get("plan_sha256")!=protocol["plan_sha256"]:
   raise FileExistsError("output protocol differs; use a fresh output directory")
 else: _json(pp,protocol)
 results.update({"source_sha256":protocol["source_sha256"],"python":sys.version,"packages":protocol["packages"],
                 "physical_cores":_physical_cores(),"memory_available_bytes":_available_memory()})
 _json(out/"input_verification.json",results); print(json.dumps(results,indent=2));return results

def _physical_cores():
 try:
  import psutil
  return psutil.cpu_count(logical=False)
 except Exception:return None
def _available_memory():
 try:
  import psutil
  return int(psutil.virtual_memory().available)
 except Exception:return None

def _reference(kind,speed,base,band=None,axis=-1,amp=0.,phase="primary"):
 from reference10 import straight,turn,slope,periodic
 if kind=="straight": return straight(speed,band,axis,amp,phase)
 if kind=="turn": return turn(speed,base["curvature"],band,axis,amp,phase)
 if kind=="slope": return slope(speed,base["gamma"],band,axis,amp,phase)
 if kind in ("s_bend","coupled"): return periodic(speed,kind,base.get("probe_k"),axis,amp)
 raise ValueError(kind)

def _case_run(job):
 """Worker writes unique run and compressed trace; returns summary."""
 os.environ.setdefault("GS12_REPO",str(job["repo"]))
 sys.path.insert(0,str(job["repo"]));sys.path.insert(0,str(HERE));sys.path.insert(0,str(ANALYSIS/"frequency_extension_src"))
 try:
  import torch;torch.set_num_threads(1)
 except ImportError:pass
 from controller10 import CondensedMPC10
 from reference10 import turn_errors,coordinate_errors
 from plant_harness import Plant,PlantConfig,diagnostic_values,SCALE,DT,PERIOD
 p=job; out=Path(p["out"]);proto=out/"protocol.json"; psha=sha(proto)
 base=p["base"]; kind=p["kind"]; speed=p["speed"]; condition=p["condition"]
 amp=p["amp"]; phase=p["phase"]; band=p.get("band"); axis=p["axis"]
 tag=band if band else (f"k{base['probe_k']}" if "probe_k" in base else "zero")
 rid=f"p{psha[:12]}_model10_v{speed:g}_{condition}_{kind}{base.get('label','')}_{tag}_axis{axis}_a{amp:g}_{phase}"
 jp=out/"runs"/(rid+".json");tp=out/"traces"/(rid+".npz")
 if jp.exists() or tp.exists():
  if not(jp.exists() and tp.exists()): raise FileExistsError(f"partial run exists: {rid}")
  old=json.loads(jp.read_text())
  if old.get("protocol_sha256")!=psha or old.get("source_sha256")!=source_hash(): raise FileExistsError(f"stale run exists: {rid}")
  return old
 cfg=PlantConfig(**CONDITIONS[condition]["plant"]); plant=Plant(speed,cfg)
 if kind=="slope": plant.x[2]=280.; plant.x[3]=-base["gamma"]
 mpc=CondensedMPC10(35,.1,CONDITIONS[condition]["fin_gain"])
 trim=cfg.parameters[1]*__import__("simulator.model",fromlist=["D_COEFF"]).D_COEFF[0,0]*speed**2/cfg.parameters[2]
 mpc.reset(np.array([trim,0.,0.])/SCALE)
 ref=_reference(kind,speed,base,band,axis,amp,phase)
 baseline_ref=_reference(kind,speed,{"label":base.get("label","")}) if kind in ("s_bend","coupled") else None
 duration=p.get("duration",256.); cycles=round(duration/PERIOD); start=time.perf_counter()
 arr={k:[] for k in ("time_substep_s","state_substep","actual_substep","queue_substep","command_substep","reference_substep","output_substep","diagnostic_substep","time_control_s","reference_control","output_control","error_normalized","error_path_normalized","input_normalized","command","qp_ok","qp_status","qp_primal","qp_dual","qp_iterations","qp_solve_time_s","model_propagation_s","qp_assembly_s","osqp_setup_s","osqp_solve_wall_s","controller_wall_s","plant_integration_s")}
 first_violation=None;qp_fail=0;iterations=0
 for k in range(cycles):
  refs=ref((k+np.arange(1,36))*PERIOD)
  command,_=mpc.solve(plant.x.copy(),refs)
  qp=mpc.last; qp_fail+=not qp["valid"];iterations+=qp["iterations"]
  arr["command"].append(command.copy());arr["qp_ok"].append(qp["valid"]);arr["qp_status"].append(qp["status"])
  arr["qp_primal"].append(qp["primal"]);arr["qp_dual"].append(qp["dual"]);arr["qp_iterations"].append(qp["iterations"]);arr["qp_solve_time_s"].append(qp["solve_time_s"])
  for nm in ("model_propagation_s","qp_assembly_s","osqp_setup_s","osqp_solve_wall_s","controller_wall_s"):arr[nm].append(qp[nm])
  plant_start=time.perf_counter()
  for j in range(5):
   plant.step(command); now=(k*5+j+1)*DT; rr=ref(now); yy=plant.output(); diag=diagnostic_values(yy,rr)
   bad=np.abs(diag)>[8.,5.,2.,np.deg2rad(40)]
   if first_violation is None and np.any(bad): first_violation={"time_s":now,"substep":k*5+j+1,"kinds":[n for n,b in zip(("lateral_8m","depth_5m","tangent_speed_2mps","pitch_40deg"),bad) if b],"values":diag.tolist()}
   if kind=="turn": epath=turn_errors(now,rr,yy,speed,base["curvature"])
   elif kind=="slope": epath=coordinate_errors(rr,yy,"slope",gamma=base["gamma"])
   elif kind in ("s_bend","coupled"):
    rb=baseline_ref(now);yaw=rb[4]; tangent=np.array([np.cos(yaw)*np.cos(rb[3]),np.sin(yaw)*np.cos(rb[3]),-np.sin(rb[3])]); normal=np.array([-np.sin(yaw),np.cos(yaw),0.])
    dp=rr[:3]-yy[:3];dv=rr[5:8]-yy[5:8];epath=np.array([normal@dp/8.,dp[2]/5.,tangent@dv/2.])
   else: epath=np.array([(rr[1]-yy[1])/8.,(rr[2]-yy[2])/5.,(rr[5]-yy[5])/2.])
   if j==4:
    arr["time_control_s"].append(now);arr["reference_control"].append(rr);arr["output_control"].append(yy)
    arr["error_normalized"].append([(rr[1]-yy[1])/8.,(rr[2]-yy[2])/5.,(rr[5]-yy[5])/2.]);arr["error_path_normalized"].append(epath)
    if kind=="periodic":
     zin=np.zeros(3); probe_k=base.get("probe_k")
     if probe_k is not None and axis>=0:
      from reference10 import _probe
      zin[axis]=float(_probe(np.asarray(now),[probe_k],[0.],amp,True)[0])
    elif band is not None and axis>=0:
     from reference10 import multi_signal
     zin=np.zeros(3);zin[axis]=float(multi_signal(np.asarray(now),band,amp,phase)[0])
    else: zin=np.zeros(3)
    arr["input_normalized"].append(zin)
   for name,value in (("time_substep_s",now),("state_substep",plant.x.copy()),("actual_substep",plant.actual.copy()),
                      ("queue_substep",plant.queue.copy()),("command_substep",command.copy()),("reference_substep",rr),
                      ("output_substep",yy), ("diagnostic_substep",diag)):
    arr[name].append(value)
  arr["plant_integration_s"].append(time.perf_counter()-plant_start)
  if first_violation is not None or not np.isfinite(plant.x).all(): break
 arr={k:np.asarray(v) for k,v in arr.items()}
 nctrl=len(arr["command"]); nsub=len(arr["time_substep_s"]); observed=float(arr["time_substep_s"][-1]) if nsub else 0.
 sat=lambda a,lo:np.mean((a>=1-1e-6)|(a<=lo+1e-6),axis=0).tolist() if len(a) else [0.,0.,0.]
 command_norm=arr["command"]/SCALE; actual_norm=arr["actual_substep"]/SCALE
 if first_violation: status="early_violation"
 elif qp_fail: status="qp_failure"
 elif nctrl!=cycles or nsub!=cycles*5: status="incomplete"
 elif any(x>0 for x in sat(command_norm,np.array([0.,-1.,-1.]))+sat(actual_norm,np.array([0.,-1.,-1.]))): status="complete_constrained"
 else: status="complete_valid"
 # Record exact reference content through the full run and MPC preview.
 rt=np.arange(0.,duration+35*PERIOD+1e-9,DT); ref_sha=sha_bytes(np.asarray(ref(rt),dtype="<f8").tobytes())
 arr["omega_radps"]=2*np.pi*np.arange(1,33)/64
 tmp=tp.with_suffix(".npz.tmp");tmp.parent.mkdir(parents=True,exist_ok=True)
 with open(tmp,"wb") as f: np.savez_compressed(f,**arr)
 tmp.replace(tp)
 rec={"run_id":rid,"protocol_sha256":psha,"source_sha256":source_hash(),"reference_sha256":ref_sha,
  "controller_state_dim":10,"controller":"condensed_rti_osqp","plant_config":{**cfg.__dict__},"fin_gain_prediction":CONDITIONS[condition]["fin_gain"],
  "operating_regime":kind,"regime_parameters":base,"speed_mps":speed,"condition":condition,"axis":axis,"band":band,
  "probe_k":base.get("probe_k"),"amplitude_per_line":amp,"phase_id":phase,"status":status,"observed_duration_s":observed,
  "control_periods":nctrl,"substeps":nsub,"first_violation":first_violation,"qp_failures":int(qp_fail),
  "mean_qp_iterations":iterations/max(nctrl,1),"mean_osqp_solve_time_s":float(np.mean(arr["qp_solve_time_s"])) if nctrl else None,
  "latency_percentiles_s":{k:{"p50":float(np.percentile(arr[k],50)),"p95":float(np.percentile(arr[k],95)),"p99":float(np.percentile(arr[k],99))} for k in ("model_propagation_s","qp_assembly_s","osqp_setup_s","osqp_solve_wall_s","controller_wall_s","plant_integration_s")},
  "command_saturation_fraction":sat(command_norm,np.array([0.,-1.,-1.])),"actual_saturation_fraction":sat(actual_norm,np.array([0.,-1.,-1.])),
  "max_actual_normalized_abs":np.max(np.abs(actual_norm),axis=0).tolist() if len(actual_norm) else None,
  "trace_sha256":sha(tp),"wall_s":time.perf_counter()-start}
 import resource
 rec["worker_peak_rss_bytes"]=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 _json(jp,rec);return rec

def _stage_jobs(stage):
 jobs=[]
 def add(speed,c,kind="straight",base=None,band=None,axis=-1,amp=0.,phase="primary",duration=256.):
  jobs.append({"speed":speed,"condition":c,"kind":kind,"base":base or {},"band":band,"axis":axis,"amp":amp,"phase":phase,"duration":duration})
 def full_straight(speeds,conditions):
  for s in speeds:
   for c in conditions:
    add(s,c,axis=-1)
    for b in BANDS:
     for a in range(3):add(s,c,band=b,axis=a,amp=.006)
 if stage=="pilot":
  add(15.,"C0",duration=64.);add(15.,"C4",duration=64.)
  add(15.,"C4","turn",{"curvature":1/120.,"label":"_left"},"B1",0,.006,duration=64.)
  add(15.,"C4","coupled",{"probe_k":6,"label":"_coupled"},axis=0,amp=.006,duration=64.)
 elif stage.startswith("throughput-w"):
  phase=stage.replace("throughput-w", "pilot_w")
  add(15.,"C0",duration=64.,phase=phase);add(15.,"C4",duration=64.,phase=phase)
  add(15.,"C4","turn",{"curvature":1/120.,"label":"_left"},"B1",0,.006,phase,duration=64.)
  add(15.,"C4","coupled",{"probe_k":6,"label":"_coupled"},axis=0,amp=.006,phase=phase,duration=64.)
 elif stage=="precheck":
  for c in ("C0","C4"):
   add(15.,c,axis=-1)
   for b in ("B1","B4"):
    for a in range(3):add(15.,c,band=b,axis=a,amp=.006)
 elif stage=="regime-precheck":
  for sign in (-1,1):
   for c in ("C0","C4"):
    turnbase={"curvature":sign/120.,"label":"_left" if sign>0 else "_right"};add(15.,c,"turn",turnbase,axis=-1)
    slopebase={"gamma":sign*np.deg2rad(3.),"label":"_climb" if sign<0 else "_dive"};add(15.,c,"slope",slopebase,axis=-1)
    for b in ("B1","B4"):
     for a in range(3):
      add(15.,c,"turn",turnbase,b,a,.006);add(15.,c,"slope",slopebase,b,a,.006)
  for kind in ("s_bend","coupled"):
   for c in ("C0","C4"):
    add(15.,c,kind,{"label":"_periodic"},axis=-1)
    for k in (4,32):
     for a in range(3):add(15.,c,kind,{"probe_k":k,"label":"_periodic"},axis=a,amp=.006)
 elif stage=="straight-15": full_straight([15.],CONDITIONS)
 elif stage=="straight-half-phase":
  for c in CONDITIONS:
   for b in BANDS:
    for a in range(3): add(15.,c,band=b,axis=a,amp=.003)
  for c in ("C0","C4","C5g"):
   for b in ("B1","B3","B4"):
    for a in range(3):add(15.,c,band=b,axis=a,amp=.006,phase="secondary")
 elif stage=="straight-other-speeds":full_straight([10.,8.,12.,18.],("C0","C4","C5g"))
 elif stage=="turn-slope":
  for s in (10.,15.):
   for sign in (-1,1):
    for c in ("C0","C4"):
     base={"curvature":sign/120.,"label":"_left" if sign>0 else "_right"};add(s,c,"turn",base,axis=-1)
     for b in BANDS:
      for a in range(3):add(s,c,"turn",base,b,a,.006)
     base={"gamma":sign*np.deg2rad(3.),"label":"_climb" if sign<0 else "_dive"};add(s,c,"slope",base,axis=-1)
     for b in BANDS:
      for a in range(3):add(s,c,"slope",base,b,a,.006)
 elif stage=="periodic-probes":
  for s in (10.,15.):
   for kind in ("s_bend","coupled"):
    for c in ("C0","C4"):
     add(s,c,kind,{"label":"_periodic"},axis=-1)
     for k in (6,10,18,30):
      for a in range(3):add(s,c,kind,{"probe_k":k,"label":"_periodic"},axis=a,amp=.006)
 elif stage=="new-regime-half":
  for sign in (-1,1):
   for c in ("C0","C4"):
    for kind,base,bands in (("turn",{"curvature":sign/120.,"label":"_left" if sign>0 else "_right"},("B1","B4")),
                            ("slope",{"gamma":sign*np.deg2rad(3.),"label":"_climb" if sign<0 else "_dive"},("B1","B4"))):
     for b in bands:
      for a in range(3):add(15.,c,kind,base,b,a,.003)
  for kind in ("s_bend","coupled"):
   for c in ("C0","C4"):
    for k in (6,10,18,30):
     for a in range(3):add(15.,c,kind,{"probe_k":k,"label":"_periodic"},axis=a,amp=.003)
 else: raise ValueError(stage)
 return jobs

def _manifest(out,rec,stage,index,total):
 p=out/"manifest.json";m=json.loads(p.read_text()) if p.exists() else {"stages":[],"runs":[]}
 if not any(r["run_id"]==rec["run_id"] for r in m["runs"]):m["runs"].append({"run_id":rec["run_id"],"status":rec["status"],"stage":stage,"ordinal":index,"trace_sha256":rec["trace_sha256"]})
 if stage not in m["stages"]:m["stages"].append(stage)
 _json(p,m)

def run_stage(args,stage):
 out=args.out.resolve();stage_start=time.perf_counter();jobs=_stage_jobs(stage);start=args.start_index;stop=args.stop_index or len(jobs);jobs=jobs[start:stop]
 for j in jobs:j["repo"]=str(args.repo.resolve());j["out"]=str(out)
 results=[]
 if args.workers==1:
  for j in jobs:results.append(_case_run(j))
 else:
  with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
   futures={pool.submit(_case_run,j):i for i,j in enumerate(jobs,start+1)}
   for f in concurrent.futures.as_completed(futures):
    i=futures[f];r=f.result();results.append(r);_manifest(out,r,stage,i,len(_stage_jobs(stage)))
    print(json.dumps({"stage":stage,"i":i,"n":len(_stage_jobs(stage)),"run_id":r["run_id"],"status":r["status"],"duration_s":r["observed_duration_s"],"wall_s":r["wall_s"]}),flush=True)
 if args.workers==1:
  for i,r in enumerate(results,start+1):
   _manifest(out,r,stage,i,len(_stage_jobs(stage)));print(json.dumps({"stage":stage,"i":i,"n":len(_stage_jobs(stage)),"run_id":r["run_id"],"status":r["status"],"duration_s":r["observed_duration_s"],"wall_s":r["wall_s"]}),flush=True)
 if stage.startswith("throughput-w"):
  save_resource_reports(out,args.workers,time.perf_counter()-stage_start)
 return results

def save_resource_reports(out,workers,stage_wall):
 psha=sha(out/"protocol.json")
 records=[json.loads(p.read_text()) for p in (out/"runs").glob("*.json") if json.loads(p.read_text()).get("protocol_sha256")==psha]
 byworker={}
 for r in records:
  if str(r.get("phase_id","")).startswith("pilot_w"):
   try:w=int(r["phase_id"].replace("pilot_w",""))
   except ValueError:continue
   byworker.setdefault(w,[]).append(r)
 byworker.setdefault(workers,[])
 byworker[workers]=[r for r in records if str(r.get("phase_id","")).startswith(f"pilot_w{workers}")]
 p=out/"throughput_pilot.json";report=json.loads(p.read_text()) if p.exists() else {"trials":[]}
 trial={"workers":workers,"stage_wall_s":stage_wall,"completed_runs":len(byworker.get(workers,[])),
        "sum_episode_wall_s":sum(r.get("wall_s",0.) for r in byworker.get(workers,[])),
        "runs_per_wall_hour":3600*len(byworker.get(workers,[]))/max(stage_wall,1e-9),
        "parallel_efficiency_eta":sum(r.get("wall_s",0.) for r in byworker.get(workers,[]))/max(workers*stage_wall,1e-9),
        "peak_worker_rss_bytes":max((r.get("worker_peak_rss_bytes",0) for r in byworker.get(workers,[])),default=0)}
 report["trials"]=[t for t in report["trials"] if t["workers"]!=workers]+[trial]
 report["protocol_sha256"]=psha;_json(p,report)
 serial=[r for r in records if r.get("phase_id")=="pilot_w1" and r.get("operating_regime")=="straight" and r.get("status")=="complete_valid"]
 if serial:
  keys=("model_propagation_s","qp_assembly_s","osqp_setup_s","osqp_solve_wall_s","controller_wall_s","plant_integration_s")
  outstats={k:{q:float(np.mean([r["latency_percentiles_s"][k][q] for r in serial])) for q in ("p50","p95","p99")} for k in keys}
  _json(out/"serial_latency.json",{"protocol_sha256":sha(out/"protocol.json"),"source_run_ids":[r["run_id"] for r in serial],"runs":len(serial),"latency_percentiles_s":outstats,"method":"mean of per-run p50/p95/p99 across serial 64-s C0/C4 straight pilots"})

def _verify_controller(repo):
 sys.path.insert(0,str(repo));sys.path.insert(0,str(HERE));sys.path.insert(0,str(ANALYSIS/"frequency_extension_src"));os.environ["GS12_REPO"]=str(repo)
 from controller10 import Dynamics10
 from plant_harness import Plant,PlantConfig,SCALE,hull_functions
 rng=np.random.default_rng(951210);x=np.array([0.,0.,20.,.03,-.05,15.,.2,-.1,.01,-.02]);c=np.array([.65,.1,-.15])
 dyn=Dynamics10();f,a,b=dyn.evaluate(x[None],c[None]);eps=1e-6;fdx=np.column_stack([(dyn.evaluate((x+np.eye(10)[i]*eps)[None],c[None])[0][0]-dyn.evaluate((x-np.eye(10)[i]*eps)[None],c[None])[0][0])/(2*eps) for i in range(10)])
 fdc=np.column_stack([(dyn.evaluate(x[None],(c+np.eye(3)[i]*eps)[None])[0][0]-dyn.evaluate(x[None],(c-np.eye(3)[i]*eps)[None])[0][0])/(2*eps) for i in range(3)])
 fd_err=max(float(np.max(np.abs(a[0]-fdx))),float(np.max(np.abs(b[0]-fdc))))
 # Immediate actuator ideal: hull prediction must match the physical RK4 plant.
 step,_=hull_functions();pars=PlantConfig().parameters;xi=x.copy();physical=c*SCALE
 for _ in range(5):xi=np.asarray(step(xi,physical,pars)).reshape(10)
 ideal=float(np.max(np.abs(xi-f[0])))
 # Explicitly record structure error on true C0: first-step mismatch is expected.
 cfg0=PlantConfig();pl0=Plant(15.,cfg0);pl0.x=x.copy();command=c*SCALE
 for _ in range(5):pl0.step(command)
 c0err=float(np.max(np.abs(pl0.x-f[0])))
 # Reference position derivatives must reproduce the supplied inertial velocities.
 from reference10 import straight,turn,slope,periodic
 ts=np.array([1.3,5.2,22.7,100.3]); h=1e-5;checks=[]
 for name,rf in (("straight",straight(10.,"B1",0,.006)),("turn",turn(10.,1/120.,"B1",0,.006)),
                 ("slope",slope(10.,np.deg2rad(3.),"B1",1,.006)),
                 ("s_bend",periodic(10.,"s_bend",6,0,.006)),("coupled",periodic(10.,"coupled",6,2,.006))):
  derivative=(rf(ts+h)[:,:3]-rf(ts-h)[:,:3])/(2*h);velocity=rf(ts)[:,5:8]
  checks.append({"regime":name,"max_position_derivative_error":float(np.max(np.abs(derivative-velocity)))})
 ref_err=max(x["max_position_derivative_error"] for x in checks)
 return {"finite_difference_jacobian_max_abs":fd_err,"jacobian_tolerance":2e-5,"jacobian_passed":fd_err<2e-5,
         "ideal_immediate_actuator_one_step_max_abs":ideal,"true_C0_one_step_structure_error":c0err,
         "reference_derivative_max_abs":ref_err,"reference_derivative_tolerance":2e-6,"reference_derivative_passed":ref_err<2e-6,
         "reference_checks":checks,
         "QP":{"horizon":35,"decision_variables":105,"control_weights":[.15,.08,.08],"rate_weights":[1.8,.35,.35],"state_weights":[3,3,3,100,30,5,5,5],"terminal_factor":2.5,"osqp_eps_abs":1e-5,"osqp_eps_rel":1e-5,"max_iter":10000}}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("verify","pilot","throughput-w1","throughput-w2","throughput-w4","precheck","regime-precheck","straight-15","straight-half-phase","straight-other-speeds","turn-slope","periodic-probes","new-regime-half"))
 ap.add_argument("--repo",type=Path,default=REPO);ap.add_argument("--out",type=Path,default=DEFAULT_OUT);ap.add_argument("--workers",type=int,default=1)
 ap.add_argument("--start-index",type=int,default=0);ap.add_argument("--stop-index",type=int);args=ap.parse_args()
 if args.stage=="verify":
  verify_inputs(args);r=_verify_controller(args.repo.resolve());_json(args.out/"controller_verification.json",r);print(json.dumps(r,indent=2));return
 if not (args.out/"protocol.json").exists():verify_inputs(args)
 run_stage(args,args.stage)
if __name__=="__main__":main()
