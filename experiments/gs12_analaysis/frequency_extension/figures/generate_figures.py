#!/usr/bin/env python3
"""Generate publication-quality descriptive figures from all_matrices.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
MATRIX_PATH = ROOT / "matrices" / "all_matrices.json"
RUNS = ROOT / "runs"

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 9,
                     "axes.grid": True, "grid.alpha": .25})


def h_array(row, key="H_final"):
    z = row[key]
    return np.asarray(z["real"]) + 1j * np.asarray(z["imag"])


def main():
    rows = json.loads(MATRIX_PATH.read_text())
    if not rows:
        raise SystemExit("No analyzed matrices found")
    FIG.mkdir(exist_ok=True)
    main15 = [r for r in rows if r["speed_mps"] == 15 and r["amplitude_per_line"] == .006
              and r["phase_id"] == "primary"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, c in zip(axs.flat, ("C0", "C1", "C2", "C3", "C4", "C5")):
        group = [r for r in main15 if r["condition"] == c]
        for r in group:
            ax.semilogx(r["omega_radps"], np.asarray(r["singular_values"])[:, 0],
                        marker=".", ms=3, lw=.9, label=r["band"])
        ax.set_title(c); ax.set_ylabel(r"$\sigma_1$ (normalized error / normalized reference)")
        ax.set_xlabel(r"$\omega$ (rad/s)")
        if group: ax.legend(ncol=2, fontsize=7)
    fig.suptitle("15 m/s, primary phase, 0.006 per line: local closed-loop FRF singular values")
    fig.tight_layout(); fig.savefig(FIG / "singular_values_15ms.png"); plt.close(fig)

    by = {(r["speed_mps"], r["condition"], r["band"], r["amplitude_per_line"], r["phase_id"]): r for r in rows}
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for band in ("B1", "B2", "B3", "B4"):
        a = by.get((15., "C4", band, .006, "primary")); b = by.get((15., "C5", band, .006, "primary"))
        if a and b:
            w = np.asarray(a["omega_radps"]); sa = np.asarray(a["singular_values"])[:, 0]
            sb = np.asarray(b["singular_values"])[:, 0]
            ax.semilogx(w, sb / sa, marker=".", ms=4, label=band)
            axd.semilogx(w, sb-sa, marker=".", ms=4, label=band)
    ax.axhline(1, color="black", lw=.8); ax.set_ylabel(r"$\sigma_1(C5)/\sigma_1(C4)$")
    axd.axhline(0, color="black", lw=.8); axd.set_xlabel(r"$\omega$ (rad/s)")
    axd.set_ylabel(r"$\sigma_1(C5)-\sigma_1(C4)$")
    ax.set_title("Same-plant paired comparison at 15 m/s; points are measured frequencies")
    ax.legend(ncol=4,fontsize=8); fig.tight_layout(); fig.savefig(FIG / "c5_over_c4_ratio_15ms.png"); plt.close(fig)

    # Each fixed physical input is a column of T; report the output 2-norm.
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, c in zip(axs, ("C0", "C4", "C5")):
        for axis, label in enumerate(("East", "Depth", "VN")):
            xx, yy = [], []
            for r in sorted([x for x in rows if x["speed_mps"] == 15 and x["condition"] == c
                             and x["amplitude_per_line"] == .006 and x["phase_id"] == "primary"],
                            key=lambda x: x["omega_radps"][0]):
                H = h_array(r)
                xx.extend(r["omega_radps"]); yy.extend(np.linalg.norm(H[:, :, axis], axis=1))
            if xx: ax.semilogx(xx, yy, ".-", ms=3, lw=.8, label=label)
        ax.set_title(c); ax.set_xlabel(r"$\omega$ (rad/s)"); ax.legend()
    axs[0].set_ylabel(r"$\|T e_j\|_2$ (normalized)")
    fig.suptitle("Fixed physical input directions, 15 m/s")
    fig.tight_layout(); fig.savefig(FIG / "physical_direction_gains_15ms.png"); plt.close(fig)

    # Element-wise complex responses: each subplot maps one physical reference
    # direction to one normalized tracking-error channel.
    conds=("C0","C4","C5"); axis_names=("East","Depth","VN"); err_names=("East error","Depth error","VN error")
    fig,axs=plt.subplots(3,3,figsize=(14,10),sharex=True)
    for i in range(3):
        for j in range(3):
            ax=axs[i,j]
            for c in conds:
                group=sorted([r for r in main15 if r["condition"]==c],key=lambda x:x["omega_radps"][0])
                xx=[]; yy=[]
                for r in group:
                    H=h_array(r);xx.extend(r["omega_radps"]);yy.extend(np.abs(H[:,i,j]))
                if xx: ax.semilogx(xx,yy,".-",ms=2.5,lw=.7,label=c)
            ax.set_title(f"{axis_names[j]} reference → {err_names[i]}")
            ax.set_ylabel("Magnitude")
            if i==2:ax.set_xlabel(r"$\omega$ (rad/s)")
            if i==0 and j==0:ax.legend(fontsize=7)
    fig.suptitle("Element-wise FRF magnitude, 15 m/s; normalized physical directions")
    fig.tight_layout();fig.savefig(FIG/"physical_response_magnitude_15ms.png");plt.close(fig)

    fig,axs=plt.subplots(3,3,figsize=(14,10),sharex=True)
    for i in range(3):
        for j in range(3):
            ax=axs[i,j]
            for c in conds:
                group=sorted([r for r in main15 if r["condition"]==c],key=lambda x:x["omega_radps"][0])
                xx=[]; values=[]
                for r in group:
                    H=h_array(r);xx.extend(r["omega_radps"]);values.extend(H[:,i,j])
                if xx:
                    values=np.asarray(values,complex);phase=np.angle(values);valid=np.abs(values)>=1e-6;phase[~valid]=np.nan
                    # Unwrap each contiguous measurable segment; leave phases
                    # undefined when the response magnitude is near zero.
                    ids=np.flatnonzero(valid)
                    for seg in np.split(ids,np.where(np.diff(ids)>1)[0]+1):
                        if len(seg)>1:phase[seg]=np.unwrap(phase[seg])
                    ax.semilogx(xx,np.rad2deg(phase),".-",ms=2.5,lw=.7,label=c)
            ax.set_title(f"{axis_names[j]} reference → {err_names[i]}")
            ax.set_ylabel("Phase (degrees)");ax.set_ylim(-190,190)
            if i==2:ax.set_xlabel(r"$\omega$ (rad/s)")
            if i==0 and j==0:ax.legend(fontsize=7)
    fig.suptitle("Element-wise FRF phase, 15 m/s; normalized physical directions")
    fig.tight_layout();fig.savefig(FIG/"physical_response_phase_15ms.png");plt.close(fig)

    speed_rows = [r for r in rows if r["condition"] in ("C0", "C4", "C5")
                  and r["amplitude_per_line"] == .006 and r["phase_id"] == "primary"]
    speeds = (8., 10., 12., 15., 18.)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, c in zip(axs, ("C0", "C4", "C5")):
        for sp in speeds:
            rr = [r for r in speed_rows if r["condition"] == c and r["speed_mps"] == sp]
            x = np.concatenate([np.asarray(r["omega_radps"]) for r in rr]) if rr else np.array([])
            y = np.concatenate([np.asarray(r["singular_values"])[:, 0] for r in rr]) if rr else np.array([])
            if x.size: ax.semilogx(x, y, ".-", ms=2.5, lw=.75, label=f"{sp:g} m/s")
        ax.set_title(c); ax.set_xlabel(r"$\omega$ (rad/s)"); ax.legend(fontsize=7)
    axs[0].set_ylabel(r"$\sigma_1$")
    fig.suptitle("Five discrete operating speeds; constrained cases retained")
    fig.tight_layout(); fig.savefig(FIG / "speed_comparison.png"); plt.close(fig)

    # Explicit quality/status map per run, including incomplete/constrained records.
    recs = [json.loads(p.read_text()) for p in RUNS.glob("p*_v*.json")]
    recs = [r for r in recs if r["amplitude_per_line"] == .006 and r["phase_id"] == "primary" and r["axis"] >= 0]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"complete_valid": "#2b8cbe", "complete_constrained": "#e34a33",
              "qp_failure": "#756bb1", "early_violation": "#d95f0e", "nonfinite": "black"}
    ylabels = [f"{r:g} m/s {c}" for r, cs in ((s, ("C0", "C1", "C2", "C3", "C4", "C5"))
                                               for s in (8., 10., 12., 15., 18.)) for c in cs]
    ypos = {label: i for i, label in enumerate(ylabels)}
    for r in recs:
        label = f"{r['speed_mps']:g} m/s {r['condition']}"
        if label not in ypos: continue
        color = colors.get(r["status"], "gray")
        band_mid = {"B1": 4.5, "B2": 12.5, "B3": 20.5, "B4": 28.5}.get(r.get("band"), 1)
        ax.scatter(band_mid, ypos[label], c=color, s=12, marker="s")
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xticks([4.5, 12.5, 20.5, 28.5], ["B1", "B2", "B3", "B4"])
    ax.set_title("Run quality at each speed / condition / band (red = constrained)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=v, label=k)
                       for k, v in colors.items()], loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "run_quality_map.png"); plt.close(fig)

    # Largest-direction phasor composition and near-degeneracy status.
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    labels=[]; mags=[]; phases=[]; near=[]
    for c in ("C0","C1","C2","C3","C4","C5"):
        rr=[r for r in main15 if r["condition"]==c]
        if not rr: continue
        best=max(((r,j) for r in rr for j,s in enumerate(r["singular_values"])),
                 key=lambda q:q[0]["singular_values"][q[1]][0])
        r,j=best; V=np.asarray(r["right_singular_vectors"]["real"])+1j*np.asarray(r["right_singular_vectors"]["imag"])
        v=V[j,:,0]; s=np.asarray(r["singular_values"])[j]; is_near=(s[0]-s[1])/max(s[0],1e-15)<.05
        labels.append(f"{c} k={r['bins'][j]}{'*' if is_near else ''}"); mags.append(np.abs(v)); phases.append(np.angle(v,deg=True));near.append(is_near)
    x=np.arange(len(labels)); width=.24
    for q,name in enumerate(("East","Depth","VN")):
        axs[0].bar(x+(q-1)*width,np.asarray(mags)[:,q],width,label=name)
        axs[1].bar(x+(q-1)*width,np.asarray(phases)[:,q],width,label=name)
    axs[0].set_ylabel("Magnitude of right singular vector"); axs[0].legend(ncol=3)
    axs[1].set_ylabel("Phase (degrees)"); axs[1].set_ylim(-190,190); axs[1].legend(ncol=3)
    axs[1].set_xticks(x,labels,rotation=25,ha="right")
    fig.suptitle("Right singular vectors at each 15 m/s maximum-$\\sigma_1$ frequency; * marks top-two gap <5%")
    fig.tight_layout(); fig.savefig(FIG / "dominant_right_singular_vectors_15ms.png"); plt.close(fig)

    # Distortion/out-of-band energy is descriptive and excludes products landing
    # exactly on an excited line; retain all constrained rows visibly.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dist=[r for r in rows if r["amplitude_per_line"]==.006 and r["phase_id"]=="primary"]
    marks={8.:"o",10.:"s",12.:"^",15.:"D",18.:"v"}
    for sp in (8.,10.,12.,15.,18.):
        group=[r for r in dist if r["speed_mps"]==sp]
        if group:
            xx=np.concatenate([r["omega_radps"] for r in group])
            yy=np.concatenate([np.full(len(r["omega_radps"]),r["nonexcited_output_energy_ratio_W3_mean"]) for r in group])
            cc=["#e34a33" if r["matrix_status"]=="complete_constrained" else "#2b8cbe" for r in group for _ in r["omega_radps"]]
            ax.scatter(xx,yy,s=13,marker=marks[sp],c=cc,label=f"{sp:g} m/s",alpha=.8)
    ax.set_xscale("log"); ax.set_xlabel(r"Excited-band frequency $\omega$ (rad/s)")
    ax.set_ylabel("W3 output energy outside the excited lines / total positive-frequency energy")
    ax.set_title("Non-excited spectral energy; constrained matrices shown in red")
    ax.legend(ncol=5,fontsize=8); fig.tight_layout(); fig.savefig(FIG / "nonexcited_spectral_energy.png"); plt.close(fig)

    # Command and actual actuator saturation fractions on the same full-amplitude
    # primary runs; red outlines identify constrained matrices.
    fullruns=[r for r in recs if r["speed_mps"] in (8.,10.,12.,15.,18.)]
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    speed_colors={8.:"#1b9e77",10.:"#d95f02",12.:"#7570b3",15.:"#1f78b4",18.:"#e7298a"}
    speed_markers={8.:"o",10.:"s",12.:"^",15.:"D",18.:"v"}
    for sp in (8.,10.,12.,15.,18.):
        grouped={}
        for r in fullruns:
            if r["speed_mps"]!=sp:continue
            key=(r["condition"],r.get("band")); mid={"B1":4.5,"B2":12.5,"B3":20.5,"B4":28.5}.get(r.get("band"),0.)
            x=2*np.pi*mid/64
            q=grouped.setdefault(key,{"x":x,"command":0.,"actual":0.,"constrained":False})
            q["command"]=max(q["command"],max(r["command_saturation_fraction"]))
            q["actual"]=max(q["actual"],max(r["actual_saturation_fraction"]))
            q["constrained"]|=(r["status"]=="complete_constrained")
        for key,q in grouped.items():
            edge="#d73027" if q["constrained"] else "white"
            axs[0].scatter(q["x"],100*q["command"],marker=speed_markers[sp],color=speed_colors[sp],edgecolor=edge,s=42)
            axs[1].scatter(q["x"],100*q["actual"],marker=speed_markers[sp],color=speed_colors[sp],edgecolor=edge,s=42,label=f"{sp:g} m/s" if key==next(iter(grouped)) else None)
    axs[0].set_title("MPC commands");axs[1].set_title("Applied actuators")
    for ax in axs:ax.set_xscale("log");ax.set_xlabel(r"Band-center frequency $\omega$ (rad/s)")
    axs[0].set_ylabel("Maximum saturated sample fraction (%)")
    axs[1].legend(ncol=2,fontsize=7)
    fig.suptitle("Saturation on 0.006 primary runs; red edge = constrained band (some markers overlap)")
    fig.tight_layout();fig.savefig(FIG/"actuator_saturation_fractions.png");plt.close(fig)

    # Five-speed heatmaps use exact measured k values; constrained entries are
    # deliberately masked so they cannot read as unconstrained FRFs.
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),sharey=True,constrained_layout=True)
    cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#bdbdbd")
    all_bins=np.arange(1,33); omega=2*np.pi*all_bins/64
    edges=2*np.pi*(np.arange(33)+.5)/64
    vmax_global=max((float(np.max(np.asarray(r["singular_values"])[:,0])) for r in speed_rows
                     if r["matrix_status"]=="complete_valid"),default=1.)
    for ax,c in zip(axs,("C0","C4","C5")):
        Z=np.full((5,32),np.nan)
        for si,sp in enumerate((8.,10.,12.,15.,18.)):
            for r in speed_rows:
                if r["speed_mps"]!=sp or r["condition"]!=c:continue
                if r["matrix_status"]=="complete_constrained":continue
                bins=np.asarray(r["bins"],int);vals=np.asarray(r["singular_values"])[:,0]
                Z[si,bins-1]=vals
        mesh=ax.pcolormesh(edges,np.arange(6),np.ma.masked_invalid(Z),shading="flat",cmap=cmap,vmin=0,vmax=vmax_global)
        ax.set_title(c);ax.set_xscale("log");ax.set_xlabel(r"$\omega$ (rad/s)")
        ax.set_yticks(np.arange(5)+.5,["8","10","12","15","18"])
    axs[0].set_ylabel("Discrete operating speed (m/s)")
    fig.colorbar(mesh,ax=axs.ravel().tolist(),label=r"$\sigma_1$ (constrained rows masked)",shrink=.86,pad=.03)
    fig.suptitle("Measured frequency–speed heatmaps; gray cells are constrained or missing")
    fig.savefig(FIG/"speed_frequency_heatmaps.png");plt.close(fig)

    replay_dir=ROOT/"direction_replays"/"runs"
    replay_files=sorted(replay_dir.glob("*.json")) if replay_dir.exists() else []
    if replay_files:
        rps=[json.loads(p.read_text()) for p in replay_files]
        fig,ax=plt.subplots(figsize=(7,6.5),constrained_layout=True)
        condition_colors={"C0":"#1f77b4","C1":"#ff7f0e","C2":"#bcbd22","C3":"#2ca02c","C4":"#9467bd","C5":"#e377c2"}
        for rp in rps:
            mr=rp["measured_error_phasor_per_unit_input"]; pr=rp["predicted_error_phasor_per_unit_input"]
            m=np.asarray(mr["real"])+1j*np.asarray(mr["imag"]); p=np.asarray(pr["real"])+1j*np.asarray(pr["imag"])
            col=condition_colors[rp["condition"]]
            ax.scatter(p.real,m.real,s=18,color=col)
            ax.scatter(p.imag,m.imag,s=18,marker="x",color=col)
        lim=max(1e-6,ax.get_xlim()[1],ax.get_ylim()[1],-ax.get_xlim()[0],-ax.get_ylim()[0])
        ax.plot([-lim,lim],[-lim,lim],"k--",lw=1,label="identity")
        ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_aspect("equal",adjustable="box")
        ax.set_xlabel("Matrix-predicted error phasor (per unit input)")
        ax.set_ylabel("Measured replay error phasor (per unit input)")
        ax.set_title("Complex-direction single-frequency replay")
        from matplotlib.lines import Line2D
        handles=[Line2D([],[],marker="o",linestyle="",color=condition_colors[c],label=c) for c in ("C0","C1","C2","C3","C4","C5")]
        handles += [Line2D([],[],marker="o",linestyle="",color="black",label="Real part"),
                    Line2D([],[],marker="x",linestyle="",color="black",label="Imaginary part"),
                    Line2D([],[],linestyle="--",color="black",label="Identity")]
        ax.legend(handles=handles,fontsize=7,ncol=3,loc="upper left");fig.savefig(FIG/"direction_replay_prediction.png");plt.close(fig)

    print(json.dumps({"figures": sorted(p.name for p in FIG.glob("*.png")), "matrix_groups": len(rows),
                      "direction_replays": len(list((ROOT/"direction_replays"/"runs").glob("*.json"))) if (ROOT/"direction_replays"/"runs").exists() else 0}))


if __name__ == "__main__":
    main()
