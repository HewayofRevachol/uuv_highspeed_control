#!/usr/bin/env python3
"""Reproducible GS-12 local FRF extension with explicit repository path."""
import argparse, copy, hashlib, importlib.metadata, json, math, os, platform, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ANALYSIS=HERE.parent
DEFAULT_OUT=ANALYSIS/"frequency_extension"
CODE_ZIP=ANALYSIS/"GS-12_report_code.zip"
DATA_ZIP=ANALYSIS/"GS-12_frequency_data.zip"
SCALE=np.array([8.,5.,2.]); T0=64.; DC=.1; DS=.02; DURATION=256.
BANDS={f"B{i+1}":list(range(8*i+1,8*i+9)) for i in range(4)}
CONDITIONS={
 "C0":{"plant":{},"model_gain":1.,"model_tau":1.},
 "C1":{"plant":{"elevator":.9,"rudder":.9},"model_gain":1.,"model_tau":1.},
 "C2":{"plant":{"tau_scale":(2.,2.,2.)},"model_gain":1.,"model_tau":1.},
 "C3":{"plant":{"order":"second_all"},"model_gain":1.,"model_tau":1.},
 "C4":{"plant":{"elevator":.9,"rudder":.9,"order":"second_all"},"model_gain":1.,"model_tau":1.},
 "C5":{"plant":{"elevator":.9,"rudder":.9,"order":"second_all"},"model_gain":.85,"model_tau":2.}}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def phases():
 p={};q={}
 for b in range(1,5):
  x=-np.pi*np.arange(8)*(np.arange(8)-1)/8
  p[f"B{b}"]=x.tolist()
  q[f"B{b}"]=(x+np.random.default_rng(9512+b).uniform(-np.pi,np.pi,8)).tolist()
 return p,q

def protocol(repo):
 p,q=phases(); core=["aav_mpc/condensed_mpc.py","simulator/batch_mpc.py","simulator/rti_symbolic.py","simulator/model.py","aav_mpc/h15_target_domain.py"]
 vers={}
 for n in ("numpy","torch","scipy","osqp","casadi","matplotlib"):
  try: vers[n]=importlib.metadata.version(n)
  except importlib.metadata.PackageNotFoundError: vers[n]=None
 try: commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
 except Exception: commit=None
 return {"protocol_version":1,"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
  "repo_commit":commit,"repo":str(repo),"source_sha256":{x.name:sha(x) for x in HERE.glob("*.py")},
  "repo_source_sha256":{n:sha(repo/n) for n in core},"input_zip_sha256":{CODE_ZIP.name:sha(CODE_ZIP),DATA_ZIP.name:sha(DATA_ZIP)},
  "python":sys.version,"packages":vers,"platform":platform.platform(),
  "thread_limits":{k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
  "duration_s":DURATION,"control_dt_s":DC,"plant_substep_s":DS,
  "frequency_grid":{"T0_s":T0,"bins":list(range(1,33)),"omega_radps":(2*np.pi*np.arange(1,33)/T0).tolist(),"bands":BANDS},
  "amplitudes_per_line":[.006,.003],"primary_phase_rad":p,"secondary_phase_rad":q,
  "startup":{"duration_s":16.,"envelope":"0.5*(1-cos(pi*t/16))","analysis_windows_s":[[64,128],[128,192],[192,256]]},
  "normalization":{"E_m":8.,"D_m":5.,"VN_mps":2.},"conditions":CONDITIONS,
  "quality_rules":{"period_relative_change_max":.02,"half_amplitude_relative_change_max":.05,
   "diagnostic_limits":{"lateral_m":8,"depth_m":5,"tangent_speed_mps":2,"pitch_deg":40}},
  "data_integrity":"ZIP CRC and delivery_manifest member SHA256 checked by verify-inputs"}

def load_harness(repo):
 os.environ["GS12_REPO"]=str(repo);sys.path.insert(0,str(repo));sys.path.insert(0,str(HERE))
 import plant_harness
 return plant_harness

def int_sin(t,w,p): return (np.cos(p)-np.cos(w*t+p))/w
def int_cos_sin(t,w,p,a):
 q=w-a;num=1-np.cos(q*t+p);term=np.divide(num,q,out=np.zeros_like(np.broadcast_arrays(num,q)[0],dtype=float),where=np.abs(q)>1e-12)
 term=np.where(np.abs(q)<=1e-12,t*np.sin(p),term)
 return .5*((1-np.cos((w+a)*t+p))/(w+a)+term)

def signal(t,bins,phase,amp):
 t=np.asarray(t,dtype=float);w=2*np.pi*np.asarray(bins,dtype=float)/T0;p=np.asarray(phase)
 x=t[...,None]*w+p;raw=amp*np.sin(x).sum(-1);dr=amp*(np.cos(x)*w).sum(-1)
 g=np.where(t<=0,0.,np.where(t<16,.5*(1-np.cos(np.pi*t/16)),1.))
 gp=np.where((t>0)&(t<16),.5*np.pi/16*np.sin(np.pi*t/16),0.)
 y=g*raw;dy=gp*raw+g*dr;a=np.pi/16;tc=np.clip(t,0,16)
 pre=amp*(.5*(int_sin(16,w,p)-int_cos_sin(16,w,p,a))).sum()
 ip=amp*(.5*(int_sin(tc[...,None],w,p)-int_cos_sin(tc[...,None],w,p,a))).sum(-1)
 tail=np.maximum(t-16,0)
 it=amp*((np.cos(w*16+p)-np.cos(w*(16+tail[...,None])+p))/w).sum(-1)
 integ=np.where(t<=0,0.,np.where(t<16,ip,pre+it))
 return y,dy,integ

def make_reference(speed,band,axis,amp,phase_id,phase_map):
 bins=BANDS[band];p=phase_map[band]
 def ref(t):
  t=np.asarray(t,dtype=float);z,dz,iz=signal(t,bins,p,amp)
  n=speed*t+(2*iz if axis==2 else 0.);e=8*z if axis==0 else np.zeros_like(t)
  dep=20+5*z if axis==1 else np.full_like(t,20.);vn=speed+2*z if axis==2 else np.full_like(t,speed)
  ve=8*dz if axis==0 else np.zeros_like(t);vd=5*dz if axis==1 else np.zeros_like(t)
  pitch=-np.arctan2(vd,np.sqrt(vn**2+ve**2));yaw=np.arctan2(ve,vn)
  return np.stack((n,e,dep,pitch,yaw,vn,ve,vd),axis=-1)
 return ref

def constant_reference(speed):
 def ref(t):
  t=np.asarray(t,dtype=float);z=np.zeros_like(t)
  return np.stack((speed*t,z,z+20,z,z,z+speed,z,z),axis=-1)
 return ref

def rid(psha,s,c,b,a,amp,phase):
 return f"p{psha[:12]}_v{s:g}_{c}_{b}_axis-{a}_a{amp:g}_{phase}"
def source_hash(): return {p.name:sha(p) for p in HERE.glob("*.py")}

def execute(args,speed,condition,band,axis,amp,phase_id="primary",duration=DURATION):
 out=args.out;psha=sha(out/"protocol.json");name=rid(psha,speed,condition,band,axis,amp,phase_id)
 rp=out/"runs"/(name+".json");tp=out/"traces"/(name+".npz")
 if rp.exists() or tp.exists():
  if not(rp.exists() and tp.exists()): raise FileExistsError(f"partial existing run {name}")
  old=json.loads(rp.read_text())
  if old.get("protocol_sha256")!=psha or old.get("source_sha256")!=source_hash(): raise FileExistsError(f"stale run {name}")
  return old
 h=load_harness(args.repo);cfg=h.PlantConfig(**CONDITIONS[condition]["plant"])
 phase_map=phases()[0 if phase_id=="primary" else 1]
 ref=constant_reference(speed) if axis<0 else make_reference(speed,band,axis,amp,phase_id,phase_map)
 setup=None
 if condition=="C5":
  from simulator.rti_symbolic import ActuatorAwareCpuDynamics
  setup=lambda m:setattr(m,"dynamics",ActuatorAwareCpuDynamics(.1,.85,2.))
 result=h.run_closed_loop(cfg,speed,ref,duration,CONDITIONS[condition]["model_gain"],CONDITIONS[condition]["model_tau"],setup,True)
 meta,ar=result["meta"],result["arrays"];idx=np.arange(4,len(ar["time"]),5);tc=ar["time"][idx]
 if len(tc) and not np.allclose(tc,np.arange(1,len(tc)+1)*DC,rtol=0,atol=1e-10): raise AssertionError("period-end sampling misaligned")
 if meta["first_diagnostic_limit_violation"]: status="early_violation"
 elif meta["qp_failures"]: status="qp_failure"
 else:
  sat=any(max(x)>0 for x in (meta["command_saturation_fraction"],meta["actual_saturation_fraction_substeps"]))
  status="complete_constrained" if sat else "complete_valid"
 trace={"time_substep_s":ar["time"],"state_substep":ar["state"],"reference_substep":ar["reference"],
  "output_substep":ar["output"],"error_substep":ar["error"],"diagnostic_substep":ar["diagnostic"],
  "actuator_substep":ar["actual"],"delay_queue_substep":ar["queue"],"actuator_rate_substep":ar["rate"],
  "command_substep":ar["subcommand"],"time_control_s":tc,
  "reference_control":ar["reference"][idx],"output_control":ar["output"][idx],
  "error_normalized":ar["error"][idx][:,[1,2,5]]/SCALE,
  "input_normalized":np.column_stack((ar["reference"][idx,1],ar["reference"][idx,2]-20,ar["reference"][idx,5]-speed))/SCALE,
  "command":ar["command"],"qp_ok":ar["qp_ok"],"qp_primal":ar["qp_primal"],"qp_dual":ar["qp_dual"],
  "one_step_error":ar["one_step_error"],"omega_radps":2*np.pi*np.arange(1,33)/64}
 np.savez_compressed(tp,**trace)
 rec={"run_id":name,"protocol_sha256":psha,"source_sha256":source_hash(),"speed_mps":speed,"condition":condition,
  "plant_config":meta["config"],"model_gain":CONDITIONS[condition]["model_gain"],"model_tau":CONDITIONS[condition]["model_tau"],
  "band":band,"axis":axis,"amplitude_per_line":amp,"phase_id":phase_id,"status":status,
  "observed_duration_s":meta["duration_observed_s"],"first_violation":meta["first_diagnostic_limit_violation"],
  "qp_failures":meta["qp_failures"],"command_saturation_fraction":meta["command_saturation_fraction"],
  "actual_saturation_fraction":meta["actual_saturation_fraction_substeps"],"max_actual_normalized_abs":meta["max_actual_normalized_abs"],
  "substeps":meta["substeps"],"control_periods":len(ar["command"]),"trace_sha256":sha(tp),
  "one_step_max_abs_state_error":meta["one_step_max_abs_state_error"],"wall_s_not_realtime_benchmark":meta["wall_s_not_realtime_benchmark"]}
 rp.write_text(json.dumps(rec,indent=2)+"\n");return rec

def update_manifest(out,r,stage,i):
 p=out/"manifest.json";m=json.loads(p.read_text()) if p.exists() else {"runs":[],"stage_order":[],"notes":[]}
 if r["run_id"] not in {x["run_id"] for x in m["runs"]}:m["runs"].append({"run_id":r["run_id"],"status":r["status"],"stage":stage,"ordinal":i,"trace_sha256":r["trace_sha256"]})
 if stage not in m["stage_order"]:m["stage_order"].append(stage)
 p.write_text(json.dumps(m,indent=2)+"\n")

def run_stage(args,stage):
 jobs=[]
 if stage=="main-15":
  for c in CONDITIONS:
   jobs.append((15.,c,"B1",-1,0.,"primary"))
   jobs += [(15.,c,b,a,.006,"primary") for b in BANDS for a in range(3)]
 elif stage=="half-15": jobs=[(15.,c,b,a,.003,"primary") for c in CONDITIONS for b in BANDS for a in range(3)]
 elif stage=="phase-15": jobs=[(15.,c,b,a,.006,"secondary") for c in ("C0","C4","C5") for b in ("B1","B3","B4") for a in range(3)]
 elif stage=="other-speeds":
  for s in (10.,8.,12.,18.):
   jobs.append((s,"C0","B1",-1,0.,"primary"))
   jobs += [(s,"C0",b,a,.006,"primary") for b in ("B1","B4") for a in range(3)]
   jobs += [(s,"C0",b,a,.006,"primary") for b in ("B2","B3") for a in range(3)]
   for c in ("C4","C5"):
    jobs.append((s,c,"B1",-1,0.,"primary"))
    jobs += [(s,c,b,a,.006,"primary") for b in BANDS for a in range(3)]
 start=max(0,args.start_index);stop=min(len(jobs),args.stop_index if args.stop_index is not None else len(jobs))
 if start>stop:raise ValueError("--start-index must be <= --stop-index")
 for i,j in enumerate(jobs[start:stop],start+1):
  r=execute(args,*j);update_manifest(args.out,r,stage,i)
  print(json.dumps({"stage":stage,"i":i,"n":len(jobs),"run_id":r["run_id"],"status":r["status"],"duration_s":r["observed_duration_s"],"wall_s":r["wall_s_not_realtime_benchmark"]}),flush=True)

def dft(x,k): return 2/len(x)*np.exp(-2j*np.pi*np.arange(len(x))*k/len(x))@x
def cd(a):
 a=np.asarray(a);return {"real":a.real.tolist(),"imag":a.imag.tolist()}

def analyze(args):
 out=args.out;psha=sha(out/"protocol.json");proto=json.loads((out/"protocol.json").read_text())
 allowed={psha:proto["source_sha256"]}
 for item in proto.get("superseded_protocols",[]):allowed[item["protocol_sha256"]]=item["source_sha256"]
 records=[]
 for p in sorted((out/"runs").glob("*.json")):
  r=json.loads(p.read_text());expected=allowed.get(r.get("protocol_sha256"))
  if expected is not None and r.get("source_sha256")==expected:records.append(r)
 L={(r["speed_mps"],r["condition"],r["band"],r["axis"],r["amplitude_per_line"],r["phase_id"]):r for r in records}
 results=[];missing=[]
 groups=sorted({(r["speed_mps"],r["condition"],r["band"],r["amplitude_per_line"],r["phase_id"]) for r in records if r["axis"]>=0})
 for speed,c,band,amp,phase in groups:
  axes=[L.get((speed,c,band,a,amp,phase)) for a in range(3)]
  zero=L.get((speed,c,"B1",-1,0.,"primary"))
  if not all(axes) or zero is None:
   missing.append({"speed_mps":speed,"condition":c,"band":band,"amplitude":amp,"phase_id":phase,"reason":"missing axis or paired zero"});continue
  if any(r["status"] not in ("complete_valid","complete_constrained") or abs(r["observed_duration_s"]-256)>1e-8 for r in axes+[zero]):
   missing.append({"speed_mps":speed,"condition":c,"band":band,"amplitude":amp,"phase_id":phase,"reason":"failed or incomplete run"});continue
  dat=[np.load(out/"traces"/(r["run_id"]+".npz")) for r in axes]
  z=np.load(out/"traces"/(zero["run_id"]+".npz"))["error_normalized"];bins=np.array(BANDS[band]);windows=[]
  for lo,hi in ((64,128),(128,192),(192,256)):
   H=np.zeros((len(bins),3,3),complex)
   for ax,d in enumerate(dat):
    t=d["time_control_s"];mask=(t>lo+1e-9)&(t<=hi+1e-9)
    if mask.sum()!=640:raise AssertionError(f"expected 640 window samples, got {mask.sum()}")
    e=d["error_normalized"][mask]-z[mask];u=d["input_normalized"][mask,ax]
    for j,k in enumerate(bins):
     E=dft(e,int(k));Z=dft(u,int(k))
     if abs(abs(Z)-amp)>max(amp*1e-8,1e-12):raise AssertionError("excitation Fourier amplitude mismatch")
     H[j,:,ax]=E/Z
   windows.append(H)
  H=(windows[1]+windows[2])/2;sv=[];us=[];vs=[];quality=[]
  for h in H:
   u,s,vh=np.linalg.svd(h);sv.append(s);us.append(u);vs.append(vh.conj().T);quality.append("near_degenerate_top2" if (s[0]-s[1])/max(s[0],1e-15)<.05 else "resolved")
  period=np.linalg.norm(windows[2]-windows[1],axis=(1,2))/np.maximum(np.linalg.norm(windows[2],axis=(1,2)),1e-8)
  # Positive-frequency output energy outside the commanded band, using W3 and
  # the matched zero baseline. Coincident intermodulation at excited lines is
  # intentionally not identifiable from this ratio and remains a limitation.
  nonexcited=[]
  for ax,d in enumerate(dat):
   t=d["time_control_s"];mask=(t>192)&(t<=256);e=d["error_normalized"][mask]-z[mask]
   coef=2/len(e)*np.fft.rfft(e,axis=0)[1:]
   bins_pos=np.arange(1,len(coef)+1);excited=set(int(k) for k in bins)
   outside=np.array([int(k) not in excited for k in bins_pos]);energy=np.sum(np.abs(coef)**2,axis=1)
   nonexcited.append(float(np.sum(energy[outside])/max(np.sum(energy),1e-30)))
  halfdiff=None
  if amp==.006 and phase=="primary":
   ha=[L.get((speed,c,band,a,.003,"primary")) for a in range(3)]
   if all(ha):
    hd=[np.load(out/"traces"/(r["run_id"]+".npz")) for r in ha];halfw=[]
    for lo,hi in ((128,192),(192,256)):
     hm=np.zeros_like(H)
     for ax,d in enumerate(hd):
      t=d["time_control_s"];mask=(t>lo)&(t<=hi);e=d["error_normalized"][mask]-z[mask];u=d["input_normalized"][mask,ax]
      for j,k in enumerate(bins):hm[j,:,ax]=dft(e,int(k))/dft(u,int(k))
     halfw.append(hm)
    hm=(halfw[0]+halfw[1])/2
    halfdiff=(np.linalg.norm(hm-H,axis=(1,2))/np.maximum(np.linalg.norm(H,axis=(1,2)),1e-8)).tolist()
  statuses=[r["status"] for r in axes];zero_status=zero["status"]
  constrained=any(s=="complete_constrained" for s in statuses+[zero_status])
  results.append({"speed_mps":speed,"condition":c,"band":band,"amplitude_per_line":amp,"phase_id":phase,
   "bins":bins.tolist(),"omega_radps":(2*np.pi*bins/64).tolist(),"H_W1":cd(windows[0]),"H_W2":cd(windows[1]),"H_W3":cd(windows[2]),"H_final":cd(H),
   "singular_values":np.asarray(sv).tolist(),"left_singular_vectors":cd(us),"right_singular_vectors":cd(vs),
   "period_relative_change_W3_vs_W2":period.tolist(),"half_amplitude_relative_change":halfdiff,"quality":quality,
   "axis_run_statuses":statuses,"zero_baseline_status":zero_status,
   "matrix_status":"complete_constrained" if constrained else "complete_valid",
   "nonexcited_output_energy_ratio_W3_by_axis":nonexcited,
   "nonexcited_output_energy_ratio_W3_mean":float(np.mean(nonexcited)),
   "source_run_ids":[r["run_id"] for r in axes],"zero_baseline_run_id":zero["run_id"],
   "period_consistent_all_le_0p02":bool(np.all(period<=.02)),
   "half_amplitude_consistent_all_le_0p05":None if halfdiff is None else bool(np.all(np.asarray(halfdiff)<=.05))})
 (out/"matrices"/"all_matrices.json").write_text(json.dumps(results,indent=2)+"\n")
 for condition in CONDITIONS:
  shard=[x for x in results if x["condition"]==condition]
  (out/"matrices"/(condition+".json")).write_text(json.dumps({"condition":condition,"matrix_groups":len(shard),"matrices":shard},indent=2)+"\n")
 by={(x["speed_mps"],x["band"],x["amplitude_per_line"],x["phase_id"],x["condition"]):x for x in results};pairs=[]
 for k in sorted({(x["speed_mps"],x["band"],x["amplitude_per_line"],x["phase_id"]) for x in results}):
  a=by.get((*k,"C4"));b=by.get((*k,"C5"))
  if a and b:
   sa=np.asarray(a["singular_values"])[:,0];sb=np.asarray(b["singular_values"])[:,0]
   pairs.append({"speed_mps":k[0],"band":k[1],"amplitude":k[2],"phase_id":k[3],"sigma1_ratio_C5_over_C4":(sb/sa).tolist(),"sigma1_difference_C5_minus_C4":(sb-sa).tolist()})
 expected=[]
 for s,cs in ((15.,tuple(CONDITIONS)),(8.,("C0","C4","C5")),(10.,("C0","C4","C5")),(12.,("C0","C4","C5")),(18.,("C0","C4","C5"))):
  for c in cs:
   for b in BANDS:expected.append((s,c,b,.006,"primary"))
 present={(r["speed_mps"],r["condition"],r["band"],r["amplitude_per_line"],r["phase_id"]) for r in results}
 expected_missing=[{"speed_mps":s,"condition":c,"band":b,"amplitude":a,"phase_id":ph} for s,c,b,a,ph in expected if (s,c,b,a,ph) not in present]
 failed=[r for r in records if r["status"] not in ("complete_valid","complete_constrained")]
 constrained_runs=[{"run_id":r["run_id"],"speed_mps":r["speed_mps"],"condition":r["condition"],"band":r["band"],"axis":r["axis"],"amplitude_per_line":r["amplitude_per_line"],"phase_id":r["phase_id"],"status":r["status"]} for r in records if r["status"]=="complete_constrained"]
 summary={"protocol_sha256":sha(out/"protocol.json"),"matrix_groups":len(results),"missing":missing,
  "expected_full_matrix_groups":len(expected),"missing_expected_full_matrix_groups":expected_missing,
  "failed_runs":failed,"constrained_runs":constrained_runs,"matrices":results,"c4_c5_same_plant":pairs,
  "interpretation":"Finite-window deterministic closed-loop reference-to-error empirical response; not output-feedback sensitivity or robust stability."}
 (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary

def verify_inputs(args):
 with zipfile.ZipFile(CODE_ZIP) as z:bad=z.testzip();cn=z.namelist();manifest=json.loads(z.read("gs12_followup/delivery_manifest.json"))
 if bad:raise ValueError(f"code ZIP CRC error: {bad}")
 with zipfile.ZipFile(DATA_ZIP) as z:
  bad=z.testzip();data={n:z.read(n) for n in z.namelist() if not n.endswith("/")};dn=len(z.namelist())
 if bad:raise ValueError(f"data ZIP CRC error: {bad}")
 entries=manifest.get("file_sha256_by_archive",{});checks=0;bad_members=[]
 for archive in (DATA_ZIP.name,CODE_ZIP.name):
  for name,expect in entries.get(archive,{}).items():
   source=data if archive==DATA_ZIP.name else None
   if source is None:
    with zipfile.ZipFile(CODE_ZIP) as cz:
     try:payload=cz.read(name)
     except KeyError:bad_members.append({"member":name,"kind":"missing"});continue
   else:
    if name not in source:bad_members.append({"member":name,"kind":"missing"});continue
    payload=source[name]
   if sha_bytes(payload)!=expect:bad_members.append({"member":name,"kind":"sha256"})
   checks+=1
 # Numerical/reference checks frozen before any frequency runs.
 phase,phase2=phases();ref=make_reference(15.,"B1",2,.006,"primary",phase)
 t=np.array([70.13,81.27,119.91]);r=ref(t);hstep=1e-5
 deriv=(ref(t+hstep)[:,:3]-ref(t-hstep)[:,:3])/(2*hstep)
 velocity=r[:,[5,6,7]]
 fd_error=float(np.max(np.abs(deriv-velocity)))
 periodic=float(np.max(np.abs(ref(t+64.)[:,1:]-r[:,1:])))
 # Known k=2 Fourier amplitude and phase convention over one 64 s period.
 known=.006*np.sin(2*np.pi*2*np.arange(1,641)/640 + .37)
 coeff=dft(known,2);dft_error=float(abs(abs(coeff)-.006))
 harness=load_harness(args.repo);parity=harness.parity_check()
 selfcheck={"reference_position_derivative_max_abs":fd_error,"steady_64s_repeat_max_abs":periodic,
  "minimum_tested_vn_mps":float(np.min(r[:,5])),"known_dft_amplitude_abs_error":dft_error,
  "plant_torch_parity":parity,"passed":fd_error<2e-7 and periodic<1e-10 and np.min(r[:,5])>0 and dft_error<1e-12 and parity["passed"]}
 result={"code_zip_sha256":sha(CODE_ZIP),"data_zip_sha256":sha(DATA_ZIP),"code_members":len(cn),"data_members":dn,"manifest_members_checked":checks,"mismatches":bad_members,"selfcheck":selfcheck,"passed":not bad_members and selfcheck["passed"]}
 (args.out/"input_verification.json").write_text(json.dumps(result,indent=2)+"\n")
 if bad_members:raise ValueError(f"manifest mismatches: {bad_members[:3]}")
 if not selfcheck["passed"]:raise ValueError(f"numerical selfcheck failed: {selfcheck}")
 print(json.dumps(result),flush=True)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def bridge(args):
 oldbins=[2,4,6,8,12,16,24]
 if args.bridge_condition and args.bridge_axis is not None:
  c=args.bridge_condition;a=args.bridge_axis;h=load_harness(args.repo);cfg=h.PlantConfig(**CONDITIONS[c]["plant"])
  oldphase=-np.pi*np.arange(7)*(np.arange(7)-1)/7
  def oldref(speed,axis,amp):
   def f(t):
    t=np.asarray(t);w=2*np.pi*np.array(oldbins)/64;x=t[...,None]*w+oldphase
    s=amp*np.sin(x).sum(-1);ds=amp*(np.cos(x)*w).sum(-1);integ=amp*((np.cos(oldphase)-np.cos(x))/w).sum(-1)
    n=speed*t+(2*integ if axis==2 else 0);e=8*s if axis==0 else np.zeros_like(t);d=20+5*s if axis==1 else np.full_like(t,20)
    vn=speed+2*s if axis==2 else np.full_like(t,speed);ve=8*ds if axis==0 else np.zeros_like(t);vd=5*ds if axis==1 else np.zeros_like(t)
    return np.stack((n,e,d,-np.arctan2(vd,np.sqrt(vn**2+ve**2)),np.arctan2(ve,vn),vn,ve,vd),-1)
   return f
  ref=constant_reference(15.) if a==-1 else oldref(15.,a,.006)
  r=h.run_closed_loop(cfg,15.,ref,144.,1.,1.,None,True);ar=r["arrays"];idx=np.arange(4,len(ar["time"]),5)
  trace={"time_control_s":ar["time"][idx],"error_normalized":ar["error"][idx][:,[1,2,5]]/SCALE,
   "input_normalized":np.column_stack((ar["reference"][idx,1],ar["reference"][idx,2]-20,ar["reference"][idx,5]-15))/SCALE}
  tp=args.out/"traces"/f"bridge_{c}_axis{a}.npz";np.savez_compressed(tp,**trace)
  rp=args.out/"runs"/f"bridge_{c}_axis{a}.json"
  rp.write_text(json.dumps({"run_id":f"bridge_{c}_axis{a}","protocol_sha256":sha(args.out/"protocol.json"),"source_sha256":source_hash(),"speed_mps":15.,"condition":c,"axis":a,"status":"complete" if len(idx)==1440 else "incomplete","observed_duration_s":float(ar["time"][-1]),"qp_failures":r["meta"]["qp_failures"],"one_step_max_abs_state_error":r["meta"]["one_step_max_abs_state_error"],"trace_sha256":sha(tp)},indent=2)+"\n")
  print(json.dumps({"bridge_condition":c,"axis":a,"duration_s":r["meta"]["duration_observed_s"],"qp_failures":r["meta"]["qp_failures"]}),flush=True);return
 results=[]
 for c in ("C0","C4"):
  traces={};metas=[]
  for a in (-1,0,1,2):
   rp=args.out/"runs"/f"bridge_{c}_axis{a}.json";tp=args.out/"traces"/f"bridge_{c}_axis{a}.npz"
   if not(rp.exists() and tp.exists()):raise FileNotFoundError(f"bridge case missing: {c} axis {a}; run bridge-7 --condition {c} --axis {a}")
   meta=json.loads(rp.read_text())
   if meta["protocol_sha256"]!=sha(args.out/"protocol.json") or meta["source_sha256"]!=source_hash() or meta["status"]!="complete":raise ValueError(f"stale/incomplete bridge case {c} axis {a}")
   d=np.load(tp);traces[a]=(d["time_control_s"],d["error_normalized"],d["input_normalized"]);metas.append(meta)
  H=np.zeros((7,3,3),complex);baseline=traces[-1][1]
  for a in range(3):
   tt,e,u=traces[a];mask=(tt>80)&(tt<=144);e=e[mask]-baseline[mask];u=u[mask,a]
   for j,k in enumerate(oldbins):H[j,:,a]=dft(e,k)/dft(u,k)
  old_member=f"gs12_followup/frequency_data/frf_v15_{'matched' if c=='C0' else 'combined'}_a0.006.npz"
  with zipfile.ZipFile(DATA_ZIP) as zf:old=np.load(__import__("io").BytesIO(zf.read(old_member)))['H']
  rel=np.linalg.norm(H-old,axis=(1,2))/np.maximum(np.linalg.norm(old,axis=(1,2)),1e-8)
  results.append({"condition":c,"H":cd(H),"historical_H_member":old_member,"relative_frobenius_difference":rel.tolist(),"bridge_pass_all_below_1e-3":bool(np.all(rel<1e-3)),"matched_one_step_max_abs_state_error":metas[0]["one_step_max_abs_state_error"],"bridge_run_ids":[f"bridge_{c}_axis{a}" for a in (-1,0,1,2)]})
 (args.out/"matrices"/"bridge_7.json").write_text(json.dumps(results,indent=2)+"\n")
 if not all(x["bridge_pass_all_below_1e-3"] for x in results):raise ValueError(f"bridge regression failed: {[x['relative_frobenius_difference'] for x in results]}")
 print(json.dumps({"bridge":"passed","conditions":[x["condition"] for x in results],"max_relative_difference":max(max(x["relative_frobenius_difference"]) for x in results)}),flush=True)

def prepare(args):
 args.out.mkdir(parents=True,exist_ok=True)
 for n in ("runs","traces","matrices","figures"):(args.out/n).mkdir(exist_ok=True)
 pp=args.out/"protocol.json";candidate=json.loads(json.dumps(protocol(args.repo)))
 if pp.exists():
  old=json.loads(pp.read_text());candidate["superseded_protocols"]=old.get("superseded_protocols",[])
  old_cmp=copy.deepcopy(old);new_cmp=copy.deepcopy(candidate)
  for value in (old_cmp,new_cmp):
   value.pop("created_utc",None);value.pop("superseded_protocols",None)
   value.get("source_sha256",{}).pop("run_frequency_extension.py",None)
  old_full=copy.deepcopy(old);new_full=copy.deepcopy(candidate)
  for value in (old_full,new_full):value.pop("created_utc",None);value.pop("superseded_protocols",None)
  if old_full!=new_full:
   existing=list((args.out/"runs").glob("*.json"))
   old_sha=sha(pp)
   if old_cmp!=new_cmp:raise FileExistsError("protocol parameters or non-runner sources changed; choose a fresh --out")
   if existing:
    if old_cmp.get("repo_source_sha256")!=new_cmp.get("repo_source_sha256") or old_cmp.get("input_zip_sha256")!=new_cmp.get("input_zip_sha256"):
     raise FileExistsError("repository or input archive changed; choose a fresh --out")
    candidate["superseded_protocols"].append({"protocol_sha256":old_sha,"source_sha256":old["source_sha256"],"reason":"postprocessing-only revision; trajectory dynamics and protocol parameters unchanged"})
   candidate["created_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());pp.write_text(json.dumps(candidate,indent=2)+"\n")
 else:pp.write_text(json.dumps(candidate,indent=2)+"\n")

def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("verify-inputs","bridge-7","main-15","half-15","phase-15","other-speeds","analyze"))
 p.add_argument("--repo",type=Path,required=True);p.add_argument("--out",type=Path,default=DEFAULT_OUT)
 p.add_argument("--bridge-condition",choices=("C0","C4"));p.add_argument("--bridge-axis",type=int,choices=(-1,0,1,2))
 p.add_argument("--start-index",type=int,default=0);p.add_argument("--stop-index",type=int)
 a=p.parse_args()
 a.repo=a.repo.resolve();a.out=a.out.resolve()
 if not(a.repo/"simulator/model.py").exists():p.error("--repo must be the repository root")
 os.environ.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","GS12_REPO":str(a.repo)})
 prepare(a)
 if a.stage=="verify-inputs":verify_inputs(a)
 elif a.stage=="analyze":
  r=analyze(a);print(json.dumps({k:r[k] for k in ("protocol_sha256","matrix_groups","expected_full_matrix_groups","missing","missing_expected_full_matrix_groups","failed_runs","constrained_runs")}),flush=True)
 elif a.stage=="bridge-7":bridge(a)
 else:run_stage(a,a.stage)
if __name__=="__main__":main()
