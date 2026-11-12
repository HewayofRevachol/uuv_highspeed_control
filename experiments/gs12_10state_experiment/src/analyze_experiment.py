#!/usr/bin/env python3
"""Recompute GS-12 frequency matrices and quality labels from raw traces."""
import argparse,json
from pathlib import Path
import numpy as np

def _read(out):
 records=[]
 for p in sorted((out/"runs").glob("*.json")):
  r=json.loads(p.read_text())
  if (out/"protocol.json").exists() and r.get("protocol_sha256")==_sha(out/"protocol.json") and r.get("phase_id") in ("primary","secondary"):records.append(r)
 return records
def _sha(p):
 import hashlib
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def _dft(x,k):
 x=np.asarray(x);return 2./len(x)*np.exp(-2j*np.pi*np.arange(len(x))*k/len(x))@x
def _cd(x):
 x=np.asarray(x);return {"real":x.real.tolist(),"imag":x.imag.tolist()}
def _key_base(r):
 b=r.get("regime_parameters",{}); return (r["speed_mps"],r["condition"],r["operating_regime"],b.get("label",""))
def _load(out,r):return np.load(out/"traces"/(r["run_id"]+".npz"))
def _window(d,lo,hi):
 t=d["time_control_s"]; m=(t>lo+1e-9)&(t<=hi+1e-9)
 if int(m.sum())!=640: raise ValueError(f"window {lo}:{hi} has {m.sum()} control samples")
 return m
def _saturated(d,lo,hi):
 scale=np.array([1500000.,.7853981633974483,.7853981633974483]);low=np.array([0.,-1.,-1.])
 mc=(d["time_control_s"]>lo+1e-9)&(d["time_control_s"]<=hi+1e-9)
 ms=(d["time_substep_s"]>lo+1e-9)&(d["time_substep_s"]<=hi+1e-9)
 c=d["command"][mc]/scale;a=d["actual_substep"][ms]/scale
 return bool(np.any((c>=1-1e-6)|(c<=low+1e-6)) or np.any((a>=1-1e-6)|(a<=low+1e-6)))
def _runs(out,records):
 direct={};probes={};
 for r in records:
  base=_key_base(r)
  if r["operating_regime"] in ("straight","turn","slope"):
   if r["axis"]<0:direct[(base,"zero")]=r
   else:direct[(base,r["band"],r["axis"],r["amplitude_per_line"],r["phase_id"])]=r
  else:
   if r["axis"]<0:probes[(base,None,-1,0.,r["phase_id"])]=r
   else:probes[(base,r.get("probe_k"),r["axis"],r["amplitude_per_line"],r["phase_id"])]=r
 return direct,probes
def analyze(out,compare22=None):
 out=Path(out);records=_read(out);direct,probes=_runs(out,records);matrices=[];missing=[]
 # Standard multisine response matrices: straight, turn and slope.
 groups=sorted({(k[0],k[1],k[2],k[3],k[4]) for k in direct if isinstance(k,tuple) and len(k)==5 and k[1] not in ("zero",)})
 # simpler discover each actual excited group
 for r in records:
  if r["operating_regime"] not in ("straight","turn","slope") or r["axis"]<0:continue
  base=_key_base(r);group=(base,r["band"],r["amplitude_per_line"],r["phase_id"])
  if any(x.get("group_key")==repr(group) for x in matrices):continue
  axes=[direct.get((base,r["band"],a,r["amplitude_per_line"],r["phase_id"])) for a in range(3)]
  zero=direct.get((base,"zero"))
  if not zero or not all(axes):
   missing.append({"group":repr(group),"reason":"missing zero baseline or one or more excited axes"});continue
  allruns=axes+[zero]
  if any(x["status"] not in ("complete_valid","complete_constrained") or abs(x["observed_duration_s"]-256)>1e-8 for x in allruns):
   missing.append({"group":repr(group),"reason":"run incomplete, constrained, or failed"});continue
  ds=[_load(out,x) for x in axes];zd=_load(out,zero);wins=[];bins=np.array([int(k) for k in range(1,33) if (k in range(1,9) if r["band"]=="B1" else k in range(9,17) if r["band"]=="B2" else k in range(17,25) if r["band"]=="B3" else k in range(25,33))])
  for lo,hi in ((0,64),(64,128),(128,192),(192,256)):
   H=np.zeros((len(bins),3,3),complex)
   for a,d in enumerate(ds):
    m=_window(d,lo,hi);e=d["error_path_normalized"][m]-zd["error_path_normalized"][m]
    u=d["input_normalized"][m,a]
    for i,k in enumerate(bins):
     den=_dft(u,int(k));H[i,:,a]=_dft(e,int(k))/den if abs(den)>1e-12 else np.nan
   wins.append(H)
  if not all(np.isfinite(w).all() for w in wins):
   missing.append({"group":repr(group),"reason":"nonfinite/zero input Fourier coefficient"});continue
  H=(wins[2]+wins[3])/2;sv=[];left=[];right=[]
  for h in H:
   u,s,vh=np.linalg.svd(h);sv.append(s);left.append(u);right.append(vh.conj().T)
  period=np.linalg.norm(wins[3]-wins[2],axis=(1,2))/np.maximum(np.linalg.norm(wins[3],axis=(1,2)),1e-8)
  saturation_windows={"W2":[_saturated(d,128,192) for d in ds+[zd]],"W3":[_saturated(d,192,256) for d in ds+[zd]]}
  saturated_analysis=any(any(v) for v in saturation_windows.values())
  quality=[]
  for i,k in enumerate(bins):
   reasons=[]
   if period[i]>.02:reasons.append("window_inconsistent")
   if saturated_analysis:reasons.append("analysis_window_saturation")
   for run in allruns:
    if run["qp_failures"]:reasons.append("qp_failure")
   quality.append("pass" if not reasons else ";".join(sorted(set(reasons))))
  half=None
  if abs(r["amplitude_per_line"]-.006)<1e-12 and r["phase_id"]=="primary":
   hr=[direct.get((base,r["band"],a,.003,"primary")) for a in range(3)]
   if all(hr):
    hd=[_load(out,x) for x in hr];hm=np.zeros_like(H)
    for a,d in enumerate(hd):
     m=_window(d,192,256);e=d["error_path_normalized"][m]-zd["error_path_normalized"][m];u=d["input_normalized"][m,a]
     for i,k in enumerate(bins):hm[i,:,a]=_dft(e,int(k))/_dft(u,int(k))
    half=(np.linalg.norm(hm-wins[3],axis=(1,2))/np.maximum(np.linalg.norm(wins[3],axis=(1,2)),1e-8)).tolist()
  matrices.append({"group_key":repr(group),"operating_regime":r["operating_regime"],"speed_mps":r["speed_mps"],"condition":r["condition"],
   "regime_parameters":r["regime_parameters"],"band":r["band"],"amplitude_per_line":r["amplitude_per_line"],"phase_id":r["phase_id"],"bins":bins.tolist(),
   "omega_radps":(2*np.pi*bins/64).tolist(),"H_W1":_cd(wins[1]),"H_W2":_cd(wins[2]),"H_W3":_cd(wins[3]),"H_mean_W2_W3":_cd(H),
   "singular_values":np.asarray(sv).tolist(),"left_singular_vectors":_cd(left),"right_singular_vectors":_cd(right),
   "period_relative_change_W3_vs_W2":period.tolist(),"half_amplitude_relative_change":half,"analysis_window_saturation_W2_W3":saturation_windows,"quality":quality,
   "column_frobenius_norms":np.linalg.norm(H,axis=1).tolist(),
   "runs":[x["run_id"] for x in axes],"zero_run":zero["run_id"]})
 # Periodic orbit probe: each frequency and axis is an independent run.
 seen=set()
 for r in records:
  if r["operating_regime"] not in ("s_bend","coupled") or r["axis"]<0:continue
  base=_key_base(r);k=r.get("probe_k");group=(base,k,r["amplitude_per_line"],r["phase_id"])
  if group in seen:continue
  seen.add(group);zero=probes.get((base,None,-1,0.,"primary"))
  axes=[probes.get((base,k,a,r["amplitude_per_line"],r["phase_id"])) for a in range(3)]
  if not zero or not all(axes):missing.append({"group":repr(group),"reason":"missing periodic zero baseline or excited axis"});continue
  if any(x["status"]!="complete_valid" or abs(x["observed_duration_s"]-256)>1e-8 for x in axes+[zero]):
   missing.append({"group":repr(group),"reason":"periodic run incomplete, constrained, or failed"});continue
  zd=_load(out,zero);ds=[_load(out,x) for x in axes];h0=[];side=[];base_consistency=[]
  for lo,hi in ((64,128),(128,192),(192,256)):
   H=np.zeros((3,3),complex);sideH={};
   for a,d in enumerate(ds):
    m=_window(d,lo,hi);e=d["error_path_normalized"][m]-zd["error_path_normalized"][m]
    # The original runner stored zeros for periodic probes because its input logger
    # matched kind=="periodic" while the configured regimes are "s_bend"/"coupled".
    # Reconstruct this exogenous input exactly from the frozen run metadata.
    tc=d["time_control_s"][m];g=np.where(tc<16.,.5*(1-np.cos(np.pi*tc/16.)),1.)
    u=r["amplitude_per_line"]*g*np.sin(2*np.pi*k*tc/64.)
    den=_dft(u,k)
    H[:,a]=_dft(e,k)/den
    for sk in sorted(set([k-5,k-4,k-3,k-1,k+1,k+3,k+4,k+5])):
     if 1<=sk<=32:sideH.setdefault(sk,np.zeros((3,3),complex))[:,a]=_dft(e,sk)/den
   h0.append(H);side.append(sideH)
  stationarity=float(np.linalg.norm(h0[2]-h0[1])/max(np.linalg.norm(h0[2]),1e-8))
  mz2=_window(zd,128,192);mz3=_window(zd,192,256)
  baseline_repeat=float(np.linalg.norm(zd["error_path_normalized"][mz3]-zd["error_path_normalized"][mz2])/
                        max(np.linalg.norm(zd["error_path_normalized"][mz3]),1e-8))
  side_norm={str(sk):float(np.linalg.norm(v)) for sk,v in side[2].items()}
  hnorm=max(float(np.linalg.norm(h0[2])),1e-12);side_ratio={sk:v/hnorm for sk,v in side_norm.items()}
  status="pass" if stationarity<=.02 and baseline_repeat<=.02 else "baseline_nonstationary"
  sv=[np.linalg.svd(v,compute_uv=False) for v in h0]
  matrices.append({"group_key":repr(group),"operating_regime":r["operating_regime"],"speed_mps":r["speed_mps"],"condition":r["condition"],
   "regime_parameters":r["regime_parameters"],"probe_k":k,"omega_radps":2*np.pi*k/64,"amplitude_per_line":r["amplitude_per_line"],
   "H0_W1":_cd(h0[0]),"H0_W2":_cd(h0[1]),"H0_W3":_cd(h0[2]),"sideband_W3":{sk:_cd(v) for sk,v in side[2].items()},
   "input_reconstruction":"amplitude * startup_envelope * sin(2*pi*k*t/64); periodic runner input_normalized field was zero",
   "sideband_norm_ratio":side_ratio,"H0_period_relative_change_W3_vs_W2":stationarity,"baseline_repeatability_W3_vs_W2":baseline_repeat,
   "H0_singular_values_W3":sv[2].tolist(),"quality":status,
   "runs":[x["run_id"] for x in axes],"zero_run":zero["run_id"]})
 # Paired controller comparison remains explicitly absent until equivalent 22-state rows exist.
 paired=[];comparison_note="No 10/22 numerical differences are emitted unless matching new-protocol 22-state run IDs and references are present."
 if compare22:
  p=Path(compare22); rows=list(p.glob("runs/*.json")) if p.exists() else []
  comparison_note=f"Found {len(rows)} candidate 22-state run summaries in {p}; exact 10/22 reference and protocol pairing has not been certified by this analyzer."
 missing=list({json.dumps(x,sort_keys=True):x for x in missing}.values())
 result={"protocol_sha256":_sha(out/"protocol.json"),"analysis_source_sha256":_sha(Path(__file__)),"records":len(records),"matrix_groups":len(matrices),"missing":missing,
  "matrices":matrices,"paired_10_vs_22":paired,"comparison_note":comparison_note,
  "interpretation":"Finite-window empirical closed-loop reference-to-error response; no stability claim. Periodic-orbit H0 is a same-frequency projection and is not a full LTI transfer matrix."}
 (out/"matrices").mkdir(exist_ok=True);(out/"matrices"/"all_matrices.json").write_text(json.dumps(matrices,indent=2,ensure_ascii=False)+"\n")
 (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--compare22",type=Path)
 a=ap.parse_args();r=analyze(a.out,a.compare22);print(json.dumps({k:r[k] for k in ("records","matrix_groups","missing","comparison_note")},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
