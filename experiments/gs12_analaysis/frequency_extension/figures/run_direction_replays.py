#!/usr/bin/env python3
"""Run preregistered single-frequency complex-direction replays after FRF analysis."""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO / "experiments/gs12_analaysis/frequency_extension_src/run_frequency_extension.py"
os.environ.update({"GS12_REPO": str(REPO), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(RUNNER_PATH.parent))
spec = importlib.util.spec_from_file_location("gs12_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def fexp(t, w):
    t = np.asarray(t, float)
    if abs(w) < 1e-12: return t.astype(complex)
    return np.expm1(1j * w * t) / (1j * w)


def envelope_integral(t, w):
    """Integral of startup-enveloped exp(i*w*t), with the protocol's 16 s ramp."""
    t = np.asarray(t, float); alpha = np.pi / 16.; tc = np.clip(t, 0., 16.)
    ramp = .5 * fexp(tc, w) - .25 * (fexp(tc, w + alpha) + fexp(tc, w - alpha))
    at16 = .5 * fexp(16., w) - .25 * (fexp(16., w + alpha) + fexp(16., w - alpha))
    tail = np.maximum(t - 16., 0.)
    return np.where(t <= 0, 0j, np.where(t < 16., ramp, at16 + np.exp(1j*w*16.)*fexp(tail, w)))


def direction_reference(speed, omega, v, amp):
    bin_k = int(round(omega * 64 / (2*np.pi))); w = 2*np.pi*bin_k/64
    q = amp * np.asarray(v, complex)
    def ref(t):
        t = np.asarray(t, float); env = np.where(t <= 0, 0., np.where(t < 16., .5*(1-np.cos(np.pi*t/16)), 1.))
        denv = np.where((t > 0) & (t < 16), .5*np.pi/16*np.sin(np.pi*t/16), 0.)
        carrier = np.exp(1j*w*t[..., None]); x = np.real(carrier*q)
        dx = np.real(carrier*(1j*w*q)); z = env[..., None]*x
        dz = denv[..., None]*x + env[..., None]*dx
        integ = np.real(envelope_integral(t[..., None], w)*q)
        north = speed*t + 2*integ[..., 2]
        east = 8*z[..., 0]; depth = 20+5*z[..., 1]; vn = speed+2*z[..., 2]
        ve = 8*dz[..., 0]; vd = 5*dz[..., 1]
        pitch = -np.arctan2(vd, np.sqrt(vn**2+ve**2)); yaw = np.arctan2(ve, vn)
        return np.stack((north,east,depth,pitch,yaw,vn,ve,vd),axis=-1)
    return ref, bin_k


def select_cases(rows):
    d = {(r["condition"], r["band"]): r for r in rows
         if r["speed_mps"] == 15 and r["amplitude_per_line"] == .006 and r["phase_id"] == "primary"}
    selected = {}
    def add(c, row, j, reason, gap_ratio=None):
        k=int(row["bins"][j]); key=(c,k)
        if key not in selected:
            selected[key]={"matrix":row,"j":j,"reasons":[]}
        selected[key]["reasons"].append(reason)
        if gap_ratio is not None: selected[key]["gap_ratio"]=gap_ratio
    for c in ("C0","C1","C2","C3","C4","C5"):
        rr = [r for (cc,_),r in d.items() if cc == c]
        flat = [(r,j,float(s[0]),float(s[1])) for r in rr for j,s in enumerate(r["singular_values"])]
        peak = max(flat,key=lambda x:x[2]); add(c,peak[0],peak[1],"maximum_sigma1")
        low = [(r,j,s1) for r,j,s1,_ in flat if r["omega_radps"][j] <= 1.5]
        if c == "C0":
            best = max(low,key=lambda x:x[2]); reason="largest_low_mid_sigma1_nominal"
        else:
            refc = "C4" if c == "C5" else "C0"
            ref = {int(k):(v,j) for (cc,_),v in d.items() if cc == refc
                   for j,k in enumerate(v["bins"])}
            def ratio(item):
                r,j,s1=item; k=int(r["bins"][j]); base=np.asarray(ref[k][0]["singular_values"])[ref[k][1],0]
                return float(s1/max(base,1e-12))
            best=max(low,key=ratio); reason=f"largest_low_mid_relative_sigma1_vs_{refc}"
        add(c,best[0],best[1],reason)
        gaps=[(r,j,float((s[0]-s[1])/max(s[0],1e-15))) for r in rr for j,s in enumerate(r["singular_values"])]
        near=min(gaps,key=lambda x:x[2])
        if near[2] < .05:
            add(c,near[0],near[1],"closest_near_degenerate_top2",near[2])
    return selected


def main():
    matrices = json.loads((ROOT/"matrices/all_matrices.json").read_text())
    chosen = select_cases(matrices)
    outdir=ROOT/"direction_replays"; outdir.mkdir(exist_ok=True)
    for p in (outdir/"runs",outdir/"traces"): p.mkdir(exist_ok=True)
    protocol={"protocol_version":1,"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
      "speed_mps":15.,"amplitude":.006,"duration_s":256.,"startup_s":16.,
      "input":"z(t)=a Re[v exp(j omega t)]","vector_norm":"||v||2=1",
      "selection_rule":"per condition maximum sigma1; low-mid relative sigma1 degradation (C0 uses largest low-mid sigma1); most near-degenerate frequency when gap <0.05",
      "runner_sha256":sha(RUNNER_PATH),"selection":[],"completed_runs":[]}
    tests=[]
    for (c,k),item in sorted(chosen.items(),key=lambda x:(x[0][0],x[0][1])):
        row=item["matrix"];j=item["j"]
        V=np.asarray(row["right_singular_vectors"]["real"])+1j*np.asarray(row["right_singular_vectors"]["imag"])
        v=V[j,:,0]/np.linalg.norm(V[j,:,0])
        tests.append({"run_id":f"v15_{c}_k{int(k)}","condition":c,"bin":int(k),
          "omega_radps":float(row["omega_radps"][j]),"selection_reasons":item["reasons"],
          "gap_ratio":item.get("gap_ratio"),"right_singular_vector_real":v.real.tolist(),
          "right_singular_vector_imag":v.imag.tolist()})
    protocol["selection"]=tests
    (outdir/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    harness=runner.load_harness(REPO); ns=0
    for (c,k),item in sorted(chosen.items(),key=lambda x:(x[0][0],x[0][1])):
        name=f"v15_{c}_k{int(k)}"
        rp=outdir/"runs"/(name+".json");tp=outdir/"traces"/(name+".npz")
        if rp.exists() or tp.exists():
            if not(rp.exists() and tp.exists()): raise FileExistsError(f"partial direction replay {name}")
            old=json.loads(rp.read_text())
            if old.get("run_id")!=name or old.get("trace_sha256")!=sha(tp):
                raise ValueError(f"stale or damaged direction replay {name}")
            if not any(x.get("run_id")==name for x in protocol["completed_runs"]):
                protocol["completed_runs"].append({"run_id":name,"status":old["status"],
                  "prediction_error_relative":old["prediction_error_relative"],"trace_sha256":old["trace_sha256"]})
            ns+=1
            continue
        row=item["matrix"]; j=item["j"]; omega=float(row["omega_radps"][j])
        H=np.asarray(row["H_final"]["real"])+1j*np.asarray(row["H_final"]["imag"])
        V=np.asarray(row["right_singular_vectors"]["real"])+1j*np.asarray(row["right_singular_vectors"]["imag"])
        v=V[j,:,0]; v=v/np.linalg.norm(v)
        ref,bin_k=direction_reference(15.,omega,v,.006)
        cfg=harness.PlantConfig(**runner.CONDITIONS[c]["plant"])
        setup=None
        if c=="C5":
            from simulator.rti_symbolic import ActuatorAwareCpuDynamics
            setup=lambda m:setattr(m,"dynamics",ActuatorAwareCpuDynamics(.1,.85,2.))
        result=harness.run_closed_loop(cfg,15.,ref,256.,runner.CONDITIONS[c]["model_gain"],runner.CONDITIONS[c]["model_tau"],setup,True)
        ar=result["arrays"]; idx=np.arange(4,len(ar["time"]),5); tc=ar["time"][idx]
        trace={"time_substep_s":ar["time"],"state_substep":ar["state"],"reference_substep":ar["reference"],
          "output_substep":ar["output"],"error_substep":ar["error"],"diagnostic_substep":ar["diagnostic"],
          "actuator_substep":ar["actual"],"delay_queue_substep":ar["queue"],"actuator_rate_substep":ar["rate"],
          "command_substep":ar["subcommand"],"time_control_s":tc,
          "reference_control":ar["reference"][idx],"output_control":ar["output"][idx],
          "error_normalized":ar["error"][idx][:,[1,2,5]]/runner.SCALE,
          "input_normalized":np.column_stack((ar["reference"][idx,1],ar["reference"][idx,2]-20,ar["reference"][idx,5]-15))/runner.SCALE,
          "command":ar["command"],"qp_ok":ar["qp_ok"],"qp_primal":ar["qp_primal"],
          "qp_dual":ar["qp_dual"],"one_step_error":ar["one_step_error"],"omega_radps":np.array([omega])}
        np.savez_compressed(tp,**trace)
        zrun=next(x for x in json.loads((ROOT/"summary.json").read_text())["matrices"]
                  if x["speed_mps"]==15 and x["condition"]==c and x["band"]==row["band"] and x["amplitude_per_line"]==.006 and x["phase_id"]=="primary")
        zp=np.load(ROOT/"traces"/(zrun["zero_baseline_run_id"]+".npz"))["error_normalized"]
        mask=(tc>192)&(tc<=256); err=trace["error_normalized"][mask]-zp[mask]
        meas=np.array([2/len(err)*np.exp(-2j*np.pi*np.arange(len(err))*bin_k/len(err))@err[:,i] for i in range(3)])
        # Local-window DFT phases are referenced to t=192.1 s; rotate back to t=0.
        meas=meas/.006*np.exp(-1j*omega*tc[mask][0])
        pred=H[j]@v; rel=float(np.linalg.norm(meas-pred)/max(np.linalg.norm(pred),1e-12))
        meta=result["meta"]; status="complete_constrained" if max(meta["command_saturation_fraction"]+meta["actual_saturation_fraction_substeps"])>0 else ("qp_failure" if meta["qp_failures"] else "complete_valid")
        rec={"run_id":name,"speed_mps":15.,"condition":c,"bin":int(k),"omega_radps":omega,"selection_reasons":item["reasons"],
          "gap_ratio":item.get("gap_ratio"),"right_singular_vector_real":v.real.tolist(),"right_singular_vector_imag":v.imag.tolist(),
          "status":status,"qp_failures":meta["qp_failures"],"duration_s":meta["duration_observed_s"],
          "prediction_error_relative":rel,
          "measured_error_phasor_per_unit_input":{"real":meas.real.tolist(),"imag":meas.imag.tolist()},
          "predicted_error_phasor_per_unit_input":{"real":pred.real.tolist(),"imag":pred.imag.tolist()},
          "trace_sha256":sha(tp)}
        rp.write_text(json.dumps(rec,indent=2)+"\n")
        protocol["completed_runs"].append({"run_id":name,"status":status,
          "prediction_error_relative":rel,"trace_sha256":rec["trace_sha256"]})
        (outdir/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
        ns+=1;print(json.dumps({"i":ns,"n":len(chosen),"run_id":name,"status":status,"prediction_error_relative":rel}),flush=True)
    protocol["created_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    (outdir/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")


if __name__=="__main__": main()
