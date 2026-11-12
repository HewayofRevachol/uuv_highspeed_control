#!/usr/bin/env python3
"""Analyze constrained full-window responses and early-stop records without pass filtering."""
import collections, csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"
SCALE=np.array([1500000.,np.pi/4,np.pi/4])
WINDOWS=(("W1",64.,128.),("W2",128.,192.),("W3",192.,256.))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def dft(x,k):
    x=np.asarray(x); return 2./len(x)*np.exp(-2j*np.pi*np.arange(len(x))*k/len(x))@x
def cd(x):
    x=np.asarray(x)
    def clean(v):
        if isinstance(v,list): return [clean(y) for y in v]
        if isinstance(v,(float,np.floating)): return float(v) if np.isfinite(v) else None
        if isinstance(v,(int,np.integer)): return int(v)
        return v
    return {"real":clean(x.real.tolist()),"imag":clean(x.imag.tolist())}
def load(r): return np.load(OUT/"traces"/(r["run_id"]+".npz"))
def wm(d,lo,hi):
    t=d["time_control_s"]; m=(t>lo+1e-9)&(t<=hi+1e-9)
    return m if int(m.sum())==640 else None
def sat(d,lo,hi):
    t=d["time_control_s"]; mc=(t>lo+1e-9)&(t<=hi+1e-9)
    ts=d["time_substep_s"]; ms=(ts>lo+1e-9)&(ts<=hi+1e-9)
    low=np.array([0.,-1.,-1.]); c=d["command"][mc]/SCALE; a=d["actual_substep"][ms]/SCALE
    return bool(np.any((c>=1-1e-6)|(c<=low+1e-6)) or np.any((a>=1-1e-6)|(a<=low+1e-6)))
def regime_base(r):
    b=r.get("regime_parameters",{})
    return (r["speed_mps"],r["condition"],r["operating_regime"],b.get("label",""))
def reason_tag(flags):
    if flags.get("window_saturation") and flags.get("window_inconsistent"): return "饱和且窗口不一致"
    if flags.get("window_saturation"): return "有饱和"
    if flags.get("window_inconsistent"): return "窗口不一致"
    if flags.get("qp_failure"): return "QP失败"
    return "其他/缺项"

protocol=sha(OUT/"protocol.json")
records=[]
for p in sorted((OUT/"runs").glob("*.json")):
    r=json.loads(p.read_text())
    if r.get("protocol_sha256")==protocol and r.get("phase_id") in ("primary","secondary"):
        records.append(r)

# Collect window estimates for all available directions, including partial-column matrices.
direct=collections.defaultdict(dict); periodic=collections.defaultdict(dict)
for r in records:
    b=regime_base(r)
    if r["operating_regime"] in ("straight","turn","slope"):
        if r.get("axis",-1)<0: direct[(b,"zero")]=r
        else: direct[(b,r.get("band"),r.get("amplitude_per_line"),r.get("phase_id"),r["axis"])]=r
    else:
        if r.get("axis",-1)<0: periodic[(b,None,-1,0.,r.get("phase_id"))]=r
        else: periodic[(b,r.get("probe_k"),r["axis"],r.get("amplitude_per_line"),r.get("phase_id"))]=r

window_results=[]
direct_groups=set()
for r in records:
    if r["operating_regime"] not in ("straight","turn","slope") or r.get("axis",-1)<0: continue
    base=regime_base(r); band=r["band"]; amp=r["amplitude_per_line"]; phase=r["phase_id"]
    g=(base,band,amp,phase)
    if g in direct_groups: continue
    direct_groups.add(g)
    zero=direct.get((base,"zero")); axes=[direct.get((base,band,amp,phase,a)) for a in range(3)]
    ds=[load(x) if x else None for x in axes]; zd=load(zero) if zero else None
    bins=np.arange(8*(int(band[1:])-1)+1,8*int(band[1:])+1)
    for wname,lo,hi in WINDOWS:
        H=np.full((len(bins),3,3),np.nan+1j*np.nan); used=[]; absent=[]
        for ax,(run,d) in enumerate(zip(axes,ds)):
            m=wm(d,lo,hi) if d is not None else None
            zm=wm(zd,lo,hi) if zd is not None else None
            if run is None or zero is None or m is None or zm is None:
                absent.append(ax); continue
            e=d["error_path_normalized"][m]-zd["error_path_normalized"][zm]
            u=d["input_normalized"][m,ax]
            for i,k in enumerate(bins):
                den=dft(u,int(k))
                if abs(den)>1e-12: H[i,:,ax]=dft(e,int(k))/den
            used.append(ax)
        if not used: continue
        full=bool(len(used)==3 and np.isfinite(H).all())
        sv=np.linalg.svd(H,compute_uv=False)[:,0].tolist() if full else None
        satflags=[sat(d,lo,hi) for d in ds+[zd] if d is not None]
        qpbad=any(x and x.get("qp_failures",0)>0 for x in axes+[zero])
        violations=[x.get("first_violation") for x in axes+[zero] if x and x.get("first_violation")]
        violation_in_window=any(lo < v.get("time_s",-1) <= hi for v in violations)
        window_results.append({"group_key":repr(g),"regime":r["operating_regime"],"speed_mps":r["speed_mps"],
          "condition":r["condition"],"label":base[-1],"band":band,"amplitude_per_line":amp,"phase_id":phase,
          "window":wname,"window_s":[lo,hi],"available_axes":used,"missing_axes":absent,"full_3x3":full,
          "H":cd(H),"sigma1_by_bin":sv,"saturation_any_run":any(satflags),"saturation_by_run":satflags,
          "qp_failure":qpbad,"violation_in_window":violation_in_window,
          "statuses":[x["status"] if x else "missing" for x in axes+[zero]],
          "durations_s":[x["observed_duration_s"] if x else None for x in axes+[zero]],
          "interpretation":"windowed constrained/partial response; not automatically a locally linear FRF"})

# Periodic-orbit runs need a separate input reconstruction and retain H0 plus sidebands.
periodic_groups=set()
for r in records:
    if r["operating_regime"] not in ("s_bend","coupled") or r.get("axis",-1)<0: continue
    base=regime_base(r); k=int(r["probe_k"]); amp=float(r["amplitude_per_line"]); phase=r["phase_id"]
    g=(base,k,amp,phase)
    if g in periodic_groups: continue
    periodic_groups.add(g)
    zero=periodic.get((base,None,-1,0.,"primary"))
    axes=[periodic.get((base,k,a,amp,phase)) for a in range(3)]
    ds=[load(x) if x else None for x in axes]; zd=load(zero) if zero else None
    side_bins=[sk for sk in sorted({k-5,k-4,k-3,k-1,k+1,k+3,k+4,k+5}) if 1<=sk<=32]
    for wname,lo,hi in WINDOWS:
        H0=np.full((3,3),np.nan+1j*np.nan); SH={sk:np.full((3,3),np.nan+1j*np.nan) for sk in side_bins}
        used=[]; absent=[]
        for ax,(run,d) in enumerate(zip(axes,ds)):
            m=wm(d,lo,hi) if d is not None else None; zm=wm(zd,lo,hi) if zd is not None else None
            if run is None or zero is None or m is None or zm is None:
                absent.append(ax); continue
            e=d["error_path_normalized"][m]-zd["error_path_normalized"][zm]
            t=d["time_control_s"][m]; env=np.where(t<16.,.5*(1-np.cos(np.pi*t/16.)),1.)
            u=float(amp)*env*np.sin(2*np.pi*k*t/64.)
            den=dft(u,k)
            if abs(den)<=1e-12: absent.append(ax); continue
            H0[:,ax]=dft(e,k)/den
            for sk in side_bins: SH[sk][:,ax]=dft(e,sk)/den
            used.append(ax)
        full=bool(len(used)==3 and np.isfinite(H0).all())
        sv=np.linalg.svd(H0,compute_uv=False).tolist() if full else None
        satflags=[sat(d,lo,hi) for d in ds+[zd] if d is not None]
        qpbad=any(x and x.get("qp_failures",0)>0 for x in axes+[zero])
        violations=[x.get("first_violation") for x in axes+[zero] if x and x.get("first_violation")]
        violation_in_window=bool(any(lo<v.get("time_s",-1)<=hi for v in violations))
        ratios={str(sk):float(np.linalg.norm(v)/max(np.linalg.norm(H0),1e-12)) if np.isfinite(v).all() and np.isfinite(H0).all() else None for sk,v in SH.items()}
        window_results.append({"group_key":repr(g),"regime":r["operating_regime"],"speed_mps":r["speed_mps"],
          "condition":r["condition"],"label":base[-1],"probe_k":k,"amplitude_per_line":amp,"phase_id":phase,
          "window":wname,"window_s":[lo,hi],"available_axes":used,"missing_axes":absent,"full_3x3":full,
          "H0":cd(H0),"H0_singular_values":sv,"sideband_norm_ratio":ratios,
          "saturation_any_run":any(satflags),"saturation_by_run":satflags,"qp_failure":qpbad,
          "violation_in_window":violation_in_window,"statuses":[x["status"] if x else "missing" for x in axes+[zero]],
          "durations_s":[x["observed_duration_s"] if x else None for x in axes+[zero]],
          "interpretation":"periodic-orbit same-frequency projection and sidebands; constrained runs retained"})

# Early stop: analyze observed transient envelope and timing. The 64 s steady windows
# cannot be used when a run ends before them, so these rows are never mislabeled FRF.
early=[]
for r in records:
    if r["status"]!="early_violation": continue
    d=load(r); diag=np.asarray(d["diagnostic_substep"]); ep=np.asarray(d["error_path_normalized"])
    inp=np.asarray(d["input_normalized"]); tsub=np.asarray(d["time_substep_s"]); tc=np.asarray(d["time_control_s"])
    first=r.get("first_violation") or {}; kinds=first.get("kinds",[])
    early.append({"run_id":r["run_id"],"regime":r["operating_regime"],"speed_mps":r["speed_mps"],
        "condition":r["condition"],"label":regime_base(r)[-1],"band":r.get("band"),"probe_k":r.get("probe_k"),
        "axis":r["axis"],"amplitude_per_line":r["amplitude_per_line"],"status":r["status"],
        "observed_duration_s":r["observed_duration_s"],"first_violation_s":first.get("time_s"),"violation_kinds":kinds,
        "diagnostic_peak_abs":np.max(np.abs(diag),axis=0).tolist() if len(diag) else None,
        "normalized_path_error_peak_abs":np.max(np.abs(ep),axis=0).tolist() if len(ep) else None,
        "normalized_path_error_rms":np.sqrt(np.mean(ep**2,axis=0)).tolist() if len(ep) else None,
        "normalized_input_peak_abs":np.max(np.abs(inp),axis=0).tolist() if len(inp) else None,
        "command_saturation_fraction":r.get("command_saturation_fraction"),
        "actual_saturation_fraction":r.get("actual_saturation_fraction"),
        "qp_failures":r.get("qp_failures"),"control_samples":len(tc),"complete_64s_windows":[],
        "spectral_status":"No complete 64 s W1/W2/W3 window; transient metrics only"})

# Reclassify full direct matrices and summarize non-pass matrices with their numerical
# W3/W2 drift, saturation channels, and half-amplitude evidence.
summary=json.loads((OUT/"summary.json").read_text())
nonpass=[]; allfull=[]
for m in summary["matrices"]:
    if m["operating_regime"] not in ("straight","turn","slope"): continue
    q=m["quality"]
    passed=all(x=="pass" for x in q) if isinstance(q,list) else q=="pass"
    p=np.asarray(m.get("period_relative_change_W3_vs_W2",[]),float)
    h2=np.asarray(m["H_W2"]["real"])+1j*np.asarray(m["H_W2"]["imag"])
    h3=np.asarray(m["H_W3"]["real"])+1j*np.asarray(m["H_W3"]["imag"])
    abschange=np.linalg.norm(h3-h2,axis=(1,2))
    satw=m.get("analysis_window_saturation_W2_W3",{})
    entry={"group_key":m["group_key"],"regime":m["operating_regime"],"speed_mps":m["speed_mps"],
      "condition":m["condition"],"label":m.get("regime_parameters",{}).get("label",""),"band":m["band"],
      "amplitude_per_line":m["amplitude_per_line"],"phase_id":m["phase_id"],"passed_original_gate":passed,
      "quality_by_bin":q,"max_W3_vs_W2_relative_change":float(np.max(p)) if len(p) else None,
      "median_W3_vs_W2_relative_change":float(np.median(p)) if len(p) else None,
      "max_W3_vs_W2_absolute_frobenius_change":float(np.max(abschange)) if len(abschange) else None,
      "saturated_W2":satw.get("W2"),"saturated_W3":satw.get("W3"),
      "half_amplitude_relative_change":m.get("half_amplitude_relative_change"),
      "sigma1_W3_by_bin":np.asarray(m["singular_values"],float)[:,0].tolist(),
      "runs":m["runs"],"zero_run":m["zero_run"]}
    allfull.append(entry)
    if not passed: nonpass.append(entry)

# Window-level aggregate, including saturated and truncated data.
window_counts=collections.Counter()
for x in window_results:
    coverage=("full_3x3" if x["full_3x3"] else "partial_columns" if x["available_axes"] else "no_complete_window")
    label=(coverage, "saturated" if x["saturation_any_run"] else "unsaturated",
           "violation_in_window" if x["violation_in_window"] else "no_violation_in_window")
    window_counts[label]+=1
early_causes=collections.Counter(k for x in early for k in x["violation_kinds"])
early_regime=collections.Counter((x["regime"],x["condition"],x["label"]) for x in early)
nonpass_reason=collections.Counter()
for x in nonpass:
    qs=x["quality_by_bin"]
    if isinstance(qs,list):
        for q in qs:
            for term in q.split(";"):
                if term: nonpass_reason[term]+=1

doc={"protocol_sha256":protocol,"source_sha256":sha(Path(__file__)),
     "definition":"Unfiltered descriptive analyses from current 10-state records; saturated/early-stop responses are retained as constrained or transient diagnostics.",
     "counts":{"records":len(records),"early_violation_runs":len(early),"full_direct_matrix_groups":len(allfull),
       "nonpass_full_direct_matrix_groups":len(nonpass),"window_response_entries":len(window_results),
       "window_response_categories":{"|".join(k):v for k,v in window_counts.items()},
       "nonpass_bin_reason_counts":dict(nonpass_reason),"early_violation_causes":dict(early_causes)},
     "nonpass_full_direct_matrices":nonpass,"all_full_direct_matrices":allfull,
     "window_responses":window_results,"early_stop_diagnostics":early,
     "early_stop_by_regime_condition_label":[{"regime":k[0],"condition":k[1],"label":k[2],"runs":v} for k,v in sorted(early_regime.items())],
     "periodic_nonpass_note":"This pass reconstructs periodic probe input from frozen amplitude/frequency metadata and includes complete_constrained runs, plus any available complete-window columns from incomplete groups."}
analysis_dir=OUT/"analysis";analysis_dir.mkdir(exist_ok=True)
(analysis_dir/"unfiltered_response_analysis.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
def csv_write(path, rows, fields):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
csv_write(analysis_dir/"nonpass_full_direct_matrices.csv",nonpass,
  ["regime","speed_mps","condition","label","band","amplitude_per_line","phase_id","passed_original_gate",
   "max_W3_vs_W2_relative_change","median_W3_vs_W2_relative_change","max_W3_vs_W2_absolute_frobenius_change",
   "saturated_W2","saturated_W3","quality_by_bin"])
csv_write(analysis_dir/"early_stop_diagnostics.csv",early,
  ["run_id","regime","speed_mps","condition","label","band","probe_k","axis","amplitude_per_line",
   "observed_duration_s","first_violation_s","violation_kinds","diagnostic_peak_abs","normalized_path_error_peak_abs",
   "normalized_path_error_rms","normalized_input_peak_abs","qp_failures","control_samples","spectral_status"])
periodic_rows=[{"regime":x["regime"],"speed_mps":x["speed_mps"],"condition":x["condition"],"label":x["label"],
  "probe_k":x["probe_k"],"amplitude_per_line":x["amplitude_per_line"],"window":x["window"],"full_3x3":x["full_3x3"],
  "available_axes":x["available_axes"],"saturation_any_run":x["saturation_any_run"],
  "sigma1_H0":(x["H0_singular_values"][0] if x.get("H0_singular_values") else None),
  "max_sideband_to_H0":(max([v for v in x.get("sideband_norm_ratio",{}).values() if v is not None],default=None)),
  "sideband_norm_ratio":x.get("sideband_norm_ratio"),"statuses":x["statuses"]}
  for x in window_results if "probe_k" in x]
csv_write(analysis_dir/"periodic_window_responses.csv",periodic_rows,
  ["regime","speed_mps","condition","label","probe_k","amplitude_per_line","window","full_3x3","available_axes",
   "saturation_any_run","sigma1_H0","max_sideband_to_H0","sideband_norm_ratio","statuses"])
print(json.dumps({"counts":doc["counts"],"early_by_condition":doc["early_stop_by_regime_condition_label"],
  "nonpass_examples":[{k:x[k] for k in ("regime","speed_mps","condition","band","max_W3_vs_W2_relative_change","saturated_W2")} for x in nonpass[:8]]},ensure_ascii=False,indent=2))
