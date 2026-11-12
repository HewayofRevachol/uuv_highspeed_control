#!/usr/bin/env python3
"""Pair all complete exact-reference straight FRFs available in both campaigns."""
import hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
OUT=HERE/"output"
TEN=OUT
TWENTY=OUT/"paired22_campaign_v3"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_runs(root,protocol=None):
    result=[]
    for p in (root/"runs").glob("*.json"):
        r=json.loads(p.read_text())
        if protocol is None or r.get("protocol_sha256")==protocol:result.append(r)
    return result
def dft(x,k):return 2/len(x)*np.exp(-2j*np.pi*np.arange(len(x))*k/len(x))@x
def cd(x):return {"real":np.asarray(x).real.tolist(),"imag":np.asarray(x).imag.tolist()}
def key(r):return (r["speed_mps"],r["condition"],r.get("operating_regime"),r.get("band"),r.get("axis"),r.get("amplitude_per_line"),r.get("phase_id"))
def readtrace(root,r):return np.load(root/"traces"/(r["run_id"]+".npz"))

h10=sha(TEN/"protocol.json"); h22=sha(TWENTY/"protocol.json")
r10=load_runs(TEN,h10);r22=load_runs(TWENTY,h22)
i10={key(r):r for r in r10 if r.get("operating_regime")=="straight"}
i22={key(r):r for r in r22 if r.get("operating_regime")=="straight"}
groups=sorted({(r["speed_mps"],r["condition"],r["band"],r["amplitude_per_line"],r["phase_id"]) for r in r10 if r.get("operating_regime")=="straight" and r.get("axis",-1)>=0},key=str)
paired=[];missing=[]
for speed,cond,band,amp,phase in groups:
    axes10=[i10.get((speed,cond,"straight",band,a,amp,phase)) for a in range(3)]
    axes22=[i22.get((speed,cond,"straight",band,a,amp,phase)) for a in range(3)]
    zero10=i10.get((speed,cond,"straight",None,-1,0.,"primary"));zero22=i22.get((speed,cond,"straight",None,-1,0.,"primary"))
    if not all(axes10+axes22) or zero10 is None or zero22 is None:
        missing.append({"speed_mps":speed,"condition":cond,"band":band,"amplitude":amp,"phase":phase,"reason":"missing one or more axes or zero"});continue
    rs=axes10+axes22+[zero10,zero22]
    if any(r.get("status") not in ("complete_valid","complete_constrained") or abs(r.get("observed_duration_s",0)-256)>1e-8 for r in rs):
        missing.append({"speed_mps":speed,"condition":cond,"band":band,"amplitude":amp,"phase":phase,"reason":"incomplete or early-stop runs"});continue
    d10=[readtrace(TEN,r) for r in axes10];d22=[readtrace(TWENTY,r) for r in axes22]
    z10=readtrace(TEN,zero10);z22=readtrace(TWENTY,zero22)
    rdiff=max(float(np.max(np.abs(x["reference_substep"]-y["reference_substep"]))) for x,y in zip(d10,d22))
    if rdiff>1e-10 or any(a.get("plant_config")!=b.get("plant_config") for a,b in zip(axes10,axes22)):
        missing.append({"speed_mps":speed,"condition":cond,"band":band,"amplitude":amp,"phase":phase,"reason":"reference or plant config mismatch","reference_max_abs":rdiff});continue
    bins=np.arange(8*(int(band[1:])-1)+1,8*int(band[1:])+1)
    windows={}
    for wn,lo,hi in (("W1",64,128),("W2",128,192),("W3",192,256)):
        H10=np.zeros((8,3,3),complex);H22=np.zeros_like(H10)
        for ax,(x,y) in enumerate(zip(d10,d22)):
            for ds,bg,base,h in ((x,z10,H10,0),(y,z22,H22,1)):
                t=ds["time_control_s"];mask=(t>lo+1e-9)&(t<=hi+1e-9)
                err=ds["error_path_normalized"][mask]-bg["error_path_normalized"][mask]
                u=ds["input_normalized"][mask,ax]
                for j,k in enumerate(bins):base[j,:,ax]=dft(err,int(k))/dft(u,int(k))
        delta=H10-H22;sv10=np.linalg.svd(H10,compute_uv=False);sv22=np.linalg.svd(H22,compute_uv=False)
        windows[wn]={"H10":cd(H10),"H22":cd(H22),"H10_minus_H22":cd(delta),
            "relative_frobenius_by_bin":(np.linalg.norm(delta,axis=(1,2))/np.maximum(np.linalg.norm(H22,axis=(1,2)),1e-8)).tolist(),
            "absolute_frobenius_by_bin":np.linalg.norm(delta,axis=(1,2)).tolist(),
            "singular_values_10":sv10.tolist(),"singular_values_22":sv22.tolist(),
            "sigma1_ratio":(sv10[:,0]/np.maximum(sv22[:,0],1e-12)).tolist()}
    paired.append({"speed_mps":speed,"condition":cond,"band":band,"amplitude_per_line":amp,"phase_id":phase,
        "bins":bins.tolist(),"omega_radps":(2*np.pi*bins/64).tolist(),"reference_max_abs_10_vs_22":rdiff,
        "plant_config_exact_match":True,"runs_10":[r["run_id"] for r in axes10],"runs_22":[r["run_id"] for r in axes22],
        "zero_10":zero10["run_id"],"zero_22":zero22["run_id"],"windows":windows})
result={"protocol_10_sha256":h10,"protocol_22_sha256":h22,"analysis_source_sha256":sha(Path(__file__)),
        "pairing_rule":"identical reference_substep arrays, identical PlantConfig, common zero baseline, same amplitude and phase",
        "paired_groups":len(paired),"missing":missing,"pairs":paired}
target=OUT/"matrices"/"paired_10_22_straight_campaign.json";target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"paired_groups":len(paired),"missing_groups":len(missing),"target":str(target)},ensure_ascii=False))
