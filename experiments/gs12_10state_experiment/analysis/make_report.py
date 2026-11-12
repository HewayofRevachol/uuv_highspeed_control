#!/usr/bin/env python3
"""Render review figures and a Chinese experiment report from frozen outputs."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

summary = json.loads((OUT / "summary.json").read_text())
records = summary["records"]
matrices = summary["matrices"]
protocol = json.loads((OUT / "protocol.json").read_text())
manifest = json.loads((OUT / "manifest.json").read_text())
audit = json.loads((OUT / "quality_gate_audit.json").read_text())
latency = json.loads((OUT / "serial_latency.json").read_text())
throughput = json.loads((OUT / "throughput_pilot.json").read_text())
ideal_summary = json.loads((OUT / "ideal_diagnostic/summary.json").read_text())
unfiltered = json.loads((OUT / "analysis/unfiltered_response_analysis.json").read_text())

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})

def is_pass(m):
    q = m.get("quality")
    return q == "pass" if isinstance(q, str) else bool(q) and all(v == "pass" for v in q)

def freq_bin_sigma(m):
    sv = np.asarray(m.get("singular_values", []), dtype=float)
    return sv[:, 0] if sv.ndim == 2 else np.array([])

def status_counts():
    counts = {}
    for rec in manifest["runs"]:
        if not rec["run_id"].startswith("p" + summary["protocol_sha256"][:12]):
            continue
        status = rec.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts

# Figure 1: passing straight-through matrices, discrete measured bins only.
straight = [m for m in matrices if m["operating_regime"] == "straight" and is_pass(m)
            and m.get("amplitude_per_line") == 0.006 and m.get("phase_id") == "primary"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
for ax, condition in zip(axes, ("C0", "C4")):
    subset = [m for m in straight if m["condition"] == condition]
    for speed in sorted({m["speed_mps"] for m in subset}):
        by_band = {}
        for m in subset:
            if m["speed_mps"] == speed:
                by_band[m["band"]] = m
        for band, m in sorted(by_band.items()):
            sig = freq_bin_sigma(m)
            ax.plot(m["omega_radps"], sig, marker="o", ms=2.5, lw=1.3,
                    label=f"{speed:g} m/s {band}")
    ax.set_title(f"直航 {condition}：通过质量门槛的 s1")
    ax.set_xlabel("角频率 ω (rad/s)")
    ax.set_ylabel("最大奇异值 s1（归一化误差 / 归一化输入）")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("10 状态直航局部频响（离散实测频点；未插值）")
fig.tight_layout()
fig.savefig(FIG / "straight_sigma1.png", dpi=180)
plt.close(fig)

# Figure 2: B1 spectra on successful 10 m/s turn and slope local matrices.
local = [m for m in matrices if m["operating_regime"] in ("turn", "slope")
         and m.get("speed_mps") == 10.0 and m.get("band") == "B1" and is_pass(m)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, regime, title in zip(axes, ("turn", "slope"), ("稳定转弯", "稳定爬升 / 下潜")):
    for m in local:
        if m["operating_regime"] != regime:
            continue
        label = f'{m["condition"]} {m.get("regime_parameters", {}).get("label", "")}'
        ax.plot(m["omega_radps"], freq_bin_sigma(m), marker="o", ms=2.5, label=label)
    ax.set_title(title + "：B1 σ₁")
    ax.set_xlabel("角频率 ω (rad/s)")
    ax.set_ylabel("σ₁")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8)
fig.suptitle("10 m/s 典型工况局部频响（仅通过质量门槛的矩阵）")
fig.tight_layout()
fig.savefig(FIG / "turn_slope_sigma1.png", dpi=180)
plt.close(fig)

# Figure 3: measured periodic H0 and dominant sideband ratios.
periodic = [m for m in matrices if m["operating_regime"] in ("s_bend", "coupled")
            and m["speed_mps"] == 10.0 and m["condition"] == "C0" and is_pass(m)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, regime in zip(axes, ("s_bend", "coupled")):
    sub = sorted((m for m in periodic if m["operating_regime"] == regime), key=lambda x: x["probe_k"])
    ks = [m["probe_k"] for m in sub]
    h0 = [max(m["H0_singular_values_W3"]) for m in sub]
    side = [max((float(v) for v in m["sideband_norm_ratio"].values()), default=float("nan")) for m in sub]
    ax.plot(ks, h0, "o-", label="H0 最大奇异值")
    ax.set_xlabel("探针谐波 k")
    ax.set_ylabel("H0 最大奇异值")
    ax.grid(True, alpha=.25)
    ax2 = ax.twinx()
    ax2.plot(ks, side, "s--", color="tab:orange", label="最大旁频 / 同频")
    ax2.set_ylabel("最大旁频范数 / 同频范数")
    ax.set_title("周期耦合" if regime == "coupled" else "周期 S 弯")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [x.get_label() for x in lines], fontsize=8, loc="best")
fig.suptitle("10 m/s C0 周期轨迹探针（仅已测且通过的离散 k）")
fig.tight_layout()
fig.savefig(FIG / "periodic_probe.png", dpi=180)
plt.close(fig)

# Figure 4: serial latency and measured throughput pilot.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
lp = latency["latency_percentiles_s"]
keys = ["model_propagation_s", "qp_assembly_s", "osqp_setup_s", "osqp_solve_wall_s", "controller_wall_s", "plant_integration_s"]
labels = ["模型传播", "QP组装", "OSQP设置", "OSQP求解", "控制器总计", "植物积分"]
x = np.arange(len(keys)); width = .24
for i, pct in enumerate(("p50", "p95", "p99")):
    axes[0].bar(x + (i-1)*width, [1000*lp[k][pct] for k in keys], width, label=pct)
axes[0].set_xticks(x, labels, rotation=25, ha="right")
axes[0].set_ylabel("毫秒 / 控制周期")
axes[0].set_title("串行单艇延迟分位数")
axes[0].grid(axis="y", alpha=.25); axes[0].legend()
tr = throughput["trials"]
axes[1].plot([t["workers"] for t in tr], [t["runs_per_wall_hour"] for t in tr], "o-", lw=2)
for t in tr:
    axes[1].annotate(f'{t["runs_per_wall_hour"]:.0f}/h', (t["workers"], t["runs_per_wall_hour"]), xytext=(0,7), textcoords="offset points", ha="center")
axes[1].set_xticks([1,2,4]); axes[1].set_xlabel("并行 worker 数")
axes[1].set_ylabel("试运行吞吐（条 / 墙钟小时）")
axes[1].set_title("4 条 64 s 任务吞吐试跑")
axes[1].grid(True, alpha=.25)
fig.tight_layout()
fig.savefig(FIG / "runtime.png", dpi=180)
plt.close(fig)

# Figure 5: speed-by-plant audit heatmap from the main protocol manifest.
cells = {}
for item in manifest["runs"]:
    rid = item["run_id"]
    if not rid.startswith("p" + summary["protocol_sha256"][:12]) or "model10" not in rid or "pilot_w" in rid:
        continue
    match = re.search(r"_v([0-9.]+)_(C[0-5]g?)_", rid)
    if not match:
        continue
    row = f"{match.group(1)} m/s {match.group(2)}"
    status = item.get("status", "unknown")
    if status == "complete_valid": category = "通过"
    elif status in ("early_violation", "nonfinite", "qp_failed"): category = "早停越界 / 失败"
    else: category = "受限 / 不完整"
    cells[(row, category)] = cells.get((row, category), 0) + 1
rows = sorted({r for r, _ in cells}, key=lambda x: (float(x.split()[0]), x.split()[2]))
categories = ["通过", "受限 / 不完整", "早停越界 / 失败"]
heat = np.array([[cells.get((r, c), 0) for c in categories] for r in rows])
fig, ax = plt.subplots(figsize=(8, max(4, .33*len(rows))))
im = ax.imshow(heat, aspect="auto", cmap="YlOrRd")
ax.set_xticks(np.arange(len(categories)), categories, rotation=15, ha="right")
ax.set_yticks(np.arange(len(rows)), rows)
for i in range(len(rows)):
    for j in range(len(categories)):
        ax.text(j, i, str(heat[i,j]), ha="center", va="center", color="black", fontsize=8)
ax.set_title("运行质量审计（按航速 × 植物；主协议清单）")
fig.colorbar(im, ax=ax, label="记录数")
fig.tight_layout()
fig.savefig(FIG / "quality_audit.png", dpi=180)
plt.close(fig)

# Unfiltered supplemental analysis: repeated but constrained matrices and early-stop traces.
nonpass = unfiltered["nonpass_full_direct_matrices"]
nonpass = sorted(nonpass, key=lambda x: x.get("max_W3_vs_W2_relative_change") or 0.)
labels = [f'{x["speed_mps"]:g} {x["condition"]} {x["regime"]}{x["label"]} {x["band"]} a{x["amplitude_per_line"]:g}' for x in nonpass]
values = [100*(x.get("max_W3_vs_W2_relative_change") or 0.) for x in nonpass]
colors = ["#c44e52" if v > 2. else "#dd8452" for v in values]
fig, ax = plt.subplots(figsize=(11, max(5, .30*len(labels))))
y=np.arange(len(labels));ax.scatter(values,y,c=colors,s=38,zorder=3)
ax.axvline(2.,color="black",ls="--",lw=1,label="2% 窗口重复性门槛")
ax.set_xscale("symlog",linthresh=2,linscale=1)
ax.set_yticks(y,labels,fontsize=7);ax.invert_yaxis()
ax.set_xlabel("W3 相对 W2 的最大频点 Frobenius 差 (%)；对数轴突出大偏差")
ax.set_title("已形成完整 3×3 矩阵但原门槛未通过的 30 组")
ax.grid(axis="x",alpha=.25);ax.legend()
fig.tight_layout();fig.savefig(FIG/"nonpass_repeatability.png",dpi=180);plt.close(fig)

early=unfiltered["early_stop_diagnostics"]
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
dur=[x["observed_duration_s"] for x in early]
axes[0].hist(dur,bins=np.arange(10,66,5),color="tab:blue",edgecolor="white")
axes[0].axvline(64,color="black",ls="--",label="首个稳态窗起点 64 s")
axes[0].set_xlabel("早停时已观测时长 (s)");axes[0].set_ylabel("运行数")
axes[0].set_title("早停记录时长（n=37）");axes[0].legend();axes[0].grid(axis="y",alpha=.2)
cause_counts=unfiltered["counts"]["early_violation_causes"]
cause_labels={"depth_5m":"深度误差超过 5 m","pitch_40deg":"俯仰超过 40°","tangent_speed_2mps":"切向速度误差超过 2 m/s"}
causes=list(cause_counts);vals=[cause_counts[k] for k in causes]
axes[1].barh([cause_labels.get(k,k) for k in causes],vals,color=["tab:red","tab:purple","tab:orange"])
axes[1].invert_yaxis();axes[1].set_xlabel("首次越界运行数");axes[1].set_title("触发的诊断边界")
axes[1].grid(axis="x",alpha=.2)
for i,v in enumerate(vals):axes[1].text(v+.15,i,str(v),va="center")
fig.tight_layout();fig.savefig(FIG/"early_stop_diagnostics.png",dpi=180);plt.close(fig)

periodic_w3=[x for x in unfiltered["window_responses"] if "probe_k" in x and x["window"]=="W3" and x["full_3x3"] and x["saturation_any_run"]]
periodic_w3=sorted(periodic_w3,key=lambda x:(x["regime"],x["speed_mps"],x["condition"],x["probe_k"]))
if periodic_w3:
    plabels=[f'{"S弯" if x["regime"]=="s_bend" else "耦合"} {x["speed_mps"]:g}m/s {x["condition"]} k{x["probe_k"]}' for x in periodic_w3]
    side=[max([v for v in x.get("sideband_norm_ratio",{}).values() if v is not None],default=0.) for x in periodic_w3]
    sig=[x["H0_singular_values"][0] if x.get("H0_singular_values") else 0. for x in periodic_w3]
    fig,ax=plt.subplots(figsize=(11,max(4,.38*len(plabels))));yy=np.arange(len(plabels))
    ax.barh(yy,side,color="tab:orange",alpha=.8,label="W3 最大旁频范数 / H0 范数")
    ax.set_yticks(yy,plabels,fontsize=8);ax.invert_yaxis();ax.set_xlabel("最大旁频 / 同频范数")
    ax.axvline(1.,color="black",ls="--",lw=1)
    ax2=ax.twiny();ax2.plot(sig,yy,"o",color="tab:blue",label="W3 H0 最大奇异值")
    ax2.set_xlabel("H0 最大奇异值")
    lines=ax.get_legend_handles_labels();lines2=ax2.get_legend_handles_labels();ax.legend(lines[0]+lines2[0],lines[1]+lines2[1],fontsize=8,loc="lower right")
    ax.set_title("包含饱和组的周期轨迹投影响应；H0 与旁频并列展示")
    fig.tight_layout();fig.savefig(FIG/"constrained_periodic_responses.png",dpi=180);plt.close(fig)

counts = status_counts()
passing_groups = sum(1 for m in matrices if is_pass(m))
primary_records = summary["records"]
record_totals = 347  # current main protocol records: includes 12 resource-pilot runs
integrity = json.loads((OUT / "scope_10state_integrity.json").read_text())
integrity_errors = len(integrity["trace_hash_errors"])
ideal_records = ideal_summary.get("records", 0)
status_text = ", ".join(f"`{k}` {v} 条" for k, v in sorted(counts.items())) or "详见运行摘要"
u_counts = unfiltered["counts"]
u_nonpass = unfiltered["nonpass_full_direct_matrices"]
u_repeatable_sat = sum((x.get("max_W3_vs_W2_relative_change") or 0.) <= .02 for x in u_nonpass)
u_drifting_sat = len(u_nonpass) - u_repeatable_sat
u_early = unfiltered["early_stop_diagnostics"]
u_early_duration = [x["observed_duration_s"] for x in u_early]
u_win = [x for x in unfiltered["window_responses"] if x["window"] == "W3"]
u_full_windows = sum(x["full_3x3"] for x in u_win)
u_partial_windows = sum(bool(x["available_axes"]) and not x["full_3x3"] for x in u_win)
u_no_windows = sum(not x["available_axes"] for x in u_win)
u_saturated_windows = sum(x["saturation_any_run"] for x in u_win)
u_periodic_sat = [x for x in u_win if "probe_k" in x and x["full_3x3"] and x["saturation_any_run"]]
u_periodic_max = max((max(v for v in x.get("sideband_norm_ratio", {}).values() if v is not None) for x in u_periodic_sat), default=0.)
u_c0_b4 = next(x for x in u_nonpass if x["regime"] == "straight" and x["speed_mps"] == 10.0 and x["condition"] == "C0" and x["band"] == "B4" and x["amplitude_per_line"] == .006 and x["phase_id"] == "primary")
u_c4_b4 = next(x for x in u_nonpass if x["regime"] == "straight" and x["speed_mps"] == 10.0 and x["condition"] == "C4" and x["band"] == "B4" and x["amplitude_per_line"] == .006 and x["phase_id"] == "primary")
u_12_c4 = next(x for x in u_nonpass if x["regime"] == "straight" and x["speed_mps"] == 12.0 and x["condition"] == "C4" and x["band"] == "B1" and x["amplitude_per_line"] == .006 and x["phase_id"] == "primary")
u_periodic_high = next(x for x in u_periodic_sat if x["speed_mps"] == 15.0 and x["condition"] == "C0" and x["regime"] == "coupled" and x["probe_k"] == 4)
regime_zh={"straight":"直航","turn":"转弯","slope":"坡度","s_bend":"S弯","coupled":"耦合"}
label_zh={"_left":"左转","_right":"右转","_climb":"爬升","_dive":"下潜","_periodic":"周期基线","":"—"}
early_table="\n".join(
    f'| {regime_zh.get(x["regime"],x["regime"])} | {x["condition"]} | {label_zh.get(x["label"],x["label"])} | {x["runs"]} |'
    for x in unfiltered["early_stop_by_regime_condition_label"])

report = f'''# GS-12 10 状态局部频响与典型工况实验报告

## 范围与结论

实验按方案中的预检与运行规则执行。10 状态主实验当前协议 `{summary["protocol_sha256"][:16]}…` 下汇总 **{primary_records} 条主测量记录**，包括早停和受限记录；另有 **12 条资源试跑**，当前协议共 **{record_totals} 条记录**。另有 **7 条理想即时执行器诊断记录**。这两部分共 **{integrity["runs_checked"]} 条 10 状态轨迹**完成 SHA-256 核查，错误 **{integrity_errors} 条**。产生 **{summary["matrix_groups"]} 个矩阵组**，其中 **{passing_groups} 个满足原方案严格判据**（周期工况使用其专用质量字段）；这只是对重复性/线性范围的标签，不是纳入本报告分析的条件。其他受限、漂移或缺项结果也纳入补充分析，原始记录均保留。

实验执行沿用了方案中预检和触限后停止后续频带的运行规则，因此部分预定频点没有采集；这影响覆盖范围，不代表已采集的受限数据没有分析价值。报告同时给出严格判据子集、受限完整窗响应和早停瞬态诊断。每个窗的数据充分性、频谱估计条件、饱和、触限、窗口漂移分别标注；不把受限响应解释成未受限的小信号线性模型，也不因其受限而从结果中删除。

## 模型核验与实验设置

- 10 状态预测器雅可比有限差分最大绝对误差为 `4.56e-9`（门槛 `2e-5`）；解析参考导数最大误差为 `3.33e-9`（门槛 `2e-6`）。理想即时执行器下一步传播误差为 0。带真实延迟的 C0 下一步差为 `0.0321`，这是预期的执行器结构失配。
- 10 状态与真实植物共用艇体模型与 22 状态凝聚型 QP 目标／约束；10 状态控制器只读艇体状态。所有运行保存协议、源码、参考哈希和原始时序。主记录汇总的状态为：{status_text}。
- 已完成 C1、15 m/s、直航、`a=0.006` 的半幅复测；该条件 B1–B4 在 `a=0.003` 和 `0.006` 下均通过。按频带计算的最大 W3/W2 相对变化不超过约 `0.075%`；最大半幅差不超过 `1.63%`，均低于方案的 2% / 5% 门槛。

## 频响结果

![通过门槛的直航离散频响](figures/straight_sigma1.png)

按原方案严格判据，C1、15 m/s 是唯一完整满足全部 B1–B4 的直航条件；图中该子集用于呈现较稳定、近似线性的结果。其余已采集的完整矩阵也有 W1/W2/W3 估计，并在补充分析中展示，包括饱和和触限工况；触限后停止后续频带造成的缺项另行标记。

![转弯和坡度局部频响](figures/turn_slope_sigma1.png)

稳定转弯与坡度中，10 m/s、C0/C4、B1 的左右转弯和爬升／下潜组满足原方案严格判据；B4 和 15 m/s 的已采集响应按受限/完整状态分别标注。B1 之外未作连续曲线推断。

![周期轨迹探针响应](figures/periodic_probe.png)

周期 S 弯与耦合轨迹的有效探针矩阵来自 10 m/s C0，已测 `k=4,6,10,18,30,32` 中通过门槛的组按原始分析记录绘图；其中 `k=4/32` 是预注册预检点，其余为完整频率探针。C4 只在 `k=4` 预检通过，`k=32` 受限后停止；不得把预检点当成完整频带。旁频是局部周期投影指标，不能解释为一般轨迹追踪成功率。

## 本次范围

本次交付聚焦 10 状态局部频响与典型工况。报告不纳入 22 状态比较。执行过程中曾启动 22 状态对照；收到范围澄清后已停止。140 条部分结果及其协议清单保留在 `paired22_campaign_v3/`，另有早期试跑和 13 条 C1/15 m/s 配对记录位于 `paired22_campaign/`、`paired22_campaign_v2/`、`paired22_exact/`；这些记录均不参与本报告结论。依赖 10/22 同工况配对的复方向重放也未纳入本次 10 状态范围。

## 数据标签、运行停止规则与计算代价

![质量门槛审计](figures/quality_audit.png)

方案中的 2% 窗口差和 5% 半幅差用于判断重复性与近似线性范围；饱和用于识别执行器受限响应。它们是解释标签，不是“能否计算频响”的门槛。横向、深度、速度、俯仰边界是本轮仿真运行的停止规则：触限后没有继续跑后续频带，因而形成缺项；但触限前有效样本仍可用于瞬态或短窗频谱分析。所有早停都保留第一次触限时间和子步。完整 256 s 及 W1/W2/W3 是为方案内窗口对比而设，不是频响分析的普遍要求。

![运行时间测量](figures/runtime.png)

两次 64 s 串行试跑均值的控制器耗时为 p50 `11.03 ms`、p95 `13.61 ms`、p99 `15.12 ms`；植物积分 p99 `1.92 ms`。4 条任务试跑吞吐为 1/2/4 worker 分别 `523/566/819` 条每墙钟小时，4 worker 比 1 worker 提高约 `56.7%`，无 OOM；单艇延迟与并行吞吐分开测量。

## 未完成项与复现材料

- 15 m/s 新工况未完成预定半幅复测；这些频点仍可报告实测响应，但没有足够证据判断该幅值范围内的线性程度。
- 周期轨迹的 `input_normalized` 原始日志字段曾写成零；分析器根据运行元数据、幅值、启动包络和频率确定性重建探针输入，矩阵记录已显式标注 `input_reconstruction`。艇体实际参考输入未更改。
- 理想即时执行器诊断 7 条全部有效，B1/B4 矩阵通过。

复现入口和配置见本目录 `README.md`、`src/`、`analysis/`。关键文件：`protocol.json`、`controller_verification.json`、`manifest.json`、`scope_10state_integrity.json`、`summary.json`、`matrices/`、`quality_gate_audit.json`、`ideal_diagnostic/`。方案 SHA-256：`{protocol["plan_sha256"]}`；主分析源码 SHA-256：`{summary["analysis_source_sha256"]}`。所有新实验产物均在 `experiments/gs12_10state_experiment/` 下。
'''
supplement = f'''

## 未过滤数据的补充分析

![受限完整矩阵的窗口重复性](figures/nonpass_repeatability.png)

这部分把受限矩阵也作为响应结果展示。30 个完整 3×3 矩阵虽未满足原方案严格判据，**全部在 W2 或 W3 有至少一个命令/执行器饱和通道**。其中 {u_repeatable_sat} 组的 W3/W2 最大相对差不超过 2%，另 {u_drifting_sat} 组超过 2%。前者是重复的**受限闭环频响估计**，后者同时显示窗口漂移；饱和改变了系统的局部线性条件，但不妨碍报告实测频谱比值。是否称为“小信号线性 FRF”与是否分析响应是两件事。

例如，10 m/s C0 直航 B4 的 W3/W2 相对变化约 `{u_c0_b4["max_W3_vs_W2_relative_change"]:.2e}`，绝对 Frobenius 差约 `{u_c0_b4["max_W3_vs_W2_absolute_frobenius_change"]:.2e}`；同时 W2/W3 的横向激励轴和深度激励轴出现饱和。这说明重复性好并不自动代表线性响应。10 m/s C4 直航 B4 的相对变化为 `{u_c4_b4["max_W3_vs_W2_relative_change"]:.1%}`，绝对差 `{u_c4_b4["max_W3_vs_W2_absolute_frobenius_change"]:.3f}`；12 m/s C4 直航 B1 则分别为 `{u_12_c4["max_W3_vs_W2_relative_change"]:.1%}` 和 `{u_12_c4["max_W3_vs_W2_absolute_frobenius_change"]:.3f}`。相对差很大时同时给绝对差，避免在响应范数很小时误读百分比。

![包含饱和条件的周期轨迹响应](figures/constrained_periodic_responses.png)

周期工况补算了完整饱和运行的 H0 同频投影和旁频。W3 有 20 个完整 3×3 周期探针组，其中 6 组包含饱和；15 m/s C0 耦合轨迹 `k=4` 的最大旁频范数达到 H0 范数的 `{max(v for v in u_periodic_high["sideband_norm_ratio"].values() if v is not None):.2f}` 倍。旁频与同频同量级或更大时，H0 只能看作周期轨迹上的同频投影，不能单独概括响应。

![早停记录的时长与触限类型](figures/early_stop_diagnostics.png)

另对 **{len(u_early)} 条早停运行**计算了首次触限、触限前归一化路径误差的峰值/RMS、输入峰值、QP 失败和饱和比例。早停时长为 `{min(u_early_duration):.1f}–{max(u_early_duration):.1f} s`，中位数 `{float(np.median(u_early_duration)):.1f} s`；没有一条达到方案固定 W1 窗的 64 秒起点。它们不能与 64 秒稳态窗作同口径比较，但仍可对较长记录做短窗谱/时频响应，对极短记录做瞬态响应和触限前输入输出分析。首次触限包括深度边界 {unfiltered["counts"]["early_violation_causes"].get("depth_5m",0)} 次、俯仰边界 {unfiltered["counts"]["early_violation_causes"].get("pitch_40deg",0)} 次、切向速度边界 {unfiltered["counts"]["early_violation_causes"].get("tangent_speed_2mps",0)} 次。当前报告给出触限前瞬态诊断；没有把不同窗长的截断谱误称为方案定义的稳态 FRF。

| 工况 | 植物 | 方向/轨迹 | 早停数 |
|---|---|---|---:|
{early_table}

在 W3 可用数据中，有 {u_full_windows} 个完整 3×3 窗口、{u_partial_windows} 个可计算部分方向列的窗口，另有 {u_no_windows} 个组没有任何完整 64 秒窗；其中 {u_saturated_windows} 个已有响应窗口至少一条参与运行有饱和。部分方向组只保存实际测得的矩阵列，没有补齐缺失列。完整逐频点复矩阵、旁频、状态和受限标记见 [`unfiltered_response_analysis.json`](analysis/unfiltered_response_analysis.json)；分组数据见 [`nonpass_full_direct_matrices.csv`](analysis/nonpass_full_direct_matrices.csv)、[`periodic_window_responses.csv`](analysis/periodic_window_responses.csv) 和 [`early_stop_diagnostics.csv`](analysis/early_stop_diagnostics.csv)。
'''
report += supplement
(OUT / "REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps({"report": str(OUT / "REPORT.md"), "figures": sorted(p.name for p in FIG.glob("*.png")), "passing_groups": passing_groups}, ensure_ascii=False, indent=2))
