#!/usr/bin/env python3
"""Compile the final human-readable experiment report from machine outputs."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    summary=json.loads((ROOT/"summary.json").read_text())
    verification=json.loads((ROOT/"input_verification.json").read_text())
    records=[json.loads(p.read_text()) for p in (ROOT/"runs").glob("p*_v*.json")]
    matrices=summary["matrices"]
    design_key=lambda r:(r["speed_mps"],r["condition"],r["band"],r["axis"],r["amplitude_per_line"],r["phase_id"])
    unique=len({design_key(r) for r in records})
    statuses=Counter(r["status"] for r in records)
    half=[r for r in matrices if r["half_amplitude_relative_change"] is not None]
    half_pass=sum(r["half_amplitude_consistent_all_le_0p05"] for r in half)
    constrained_matrices=[r for r in matrices if r["matrix_status"]=="complete_constrained"]
    period_pass=sum(r["period_consistent_all_le_0p02"] for r in matrices)
    nonexcited=max((r["nonexcited_output_energy_ratio_W3_mean"] for r in matrices),default=0.)
    peaks=[]
    for speed in (8.,10.,12.,15.,18.):
        cs=("C0","C1","C2","C3","C4","C5") if speed==15 else ("C0","C4","C5")
        for c in cs:
            group=[r for r in matrices if r["speed_mps"]==speed and r["condition"]==c and
                   r["amplitude_per_line"]==.006 and r["phase_id"]=="primary"]
            if not group: continue
            best=max(((float(np.asarray(r["singular_values"])[:,0][j]),r,j) for r in group
                      for j in range(len(r["bins"]))),key=lambda x:x[0])
            val,r,j=best
            peaks.append((speed,c,r["omega_radps"][j],val,r["matrix_status"]))
    replay_dir=ROOT/"direction_replays"/"runs"
    replays=[json.loads(p.read_text()) for p in sorted(replay_dir.glob("*.json"))] if replay_dir.exists() else []
    constraint_rows=summary["constrained_runs"]
    lines=[]
    lines += ["# GS-12 局部频响扩展实验结果", "",
      "## Material Passport", "",
      "- ID：`GS12-FRF-EXT-2026-11-10`", "- Type：确定性闭环仿真与经验频响辨识",
      "- Status：已完成；约束运行和幅值差异作为结果保留", "- Run command：`python3 experiments/gs12_analaysis/frequency_extension_src/run_frequency_extension.py <stage> --repo . --out experiments/gs12_analaysis/frequency_extension`",
      "- Primary output：`summary.json`、`matrices/all_matrices.json`、`runs/`、`traces/`、`figures/`、`manifest.json`", "",
      "## 实验范围与执行", "",
      "按方案测试整数频线 `k=1…32`（`0.09817…3.14159 rad/s`），单频周期 64 s，每条闭环运行 256 s；使用 `[64,128]`、`[128,192]`、`[192,256] s` 三个稳态窗口，主矩阵取 W2/W3 复数平均。误差与输入按 `[8 m, 5 m, 2 m/s]` 归一化。15 m/s 测 C0–C5；8/10/12/18 m/s 测 C0、C4、C5。15 m/s 执行 0.006 全幅、0.003 半幅和第二相位复核；低速受约束频带补测 0.003 半幅。", "",
      f"- 输入归档核验：{verification['manifest_members_checked']} 个成员哈希通过；源码 ZIP `{verification['code_zip_sha256']}`，数据 ZIP `{verification['data_zip_sha256']}`。", 
      f"- 运行记录：{len(records)} 个运行 JSON，覆盖 {unique} 个唯一设计键；其中 3 个协议修订前的重复键被保留，汇总按有效协议/源码哈希选择记录。",
      f"- 状态：`complete_valid` {statuses.get('complete_valid',0)} 条，`complete_constrained` {statuses.get('complete_constrained',0)} 条；失败/提前终止 {sum(v for k,v in statuses.items() if k not in ('complete_valid','complete_constrained'))} 条。",
      f"- 计划全幅矩阵：{summary['expected_full_matrix_groups']} 组，缺失 {len(summary['missing_expected_full_matrix_groups'])} 组；全部矩阵组 {summary['matrix_groups']}，分析缺失 {len(summary['missing'])} 组。",
      f"- W2/W3 周期一致性：{period_pass}/{len(matrices)} 组所有频点均 ≤2%；最大逐频点相对差 {max(max(r['period_relative_change_W3_vs_W2']) for r in matrices):.3g}。",
      f"- 半幅比较：{len(half)} 组中 {half_pass} 组所有频点 ≤5%；{len(half)-half_pass} 组超过工程阈值，仍作为幅值相关经验响应报告。最大逐频点差 {max(max(r['half_amplitude_relative_change']) for r in half):.1%}。",
      f"- 矩阵中有 {len(constrained_matrices)} 组因激励轴或零基线受命令/实际执行器约束；原始轨迹和矩阵均保留。受约束全幅触发了 {sum(r['amplitude_per_line']==.006 and r['phase_id']=='primary' and r['speed_mps']!=15 for r in constraint_rows)} 条低速轴记录，并完成 24 条半幅补测。",
      f"- W3 非激励正频线输出能量占比（排除恰落在激励线上的互调项）最大均值：{nonexcited:.3g}。这是谱能量描述量，不是自动剔除阈值。", "",
      "## 各离散航速的峰值", "",
      "下表列出实测频点上最大的 `σ1`。低速峰值若标为受约束，不作无约束局部线性解释。不同速度是五个离散工作点，峰值不代表连续扫速。", "",
      "| 航速 | 条件 | 峰值频率 (rad/s) | 最大 `σ1` | 矩阵状态 |", "|---:|:---:|---:|---:|:---|"]
    lines += [f"| {s:g} | {c} | {w:.5f} | {v:.6f} | {st} |" for s,c,w,v,st in peaks]
    lines += ["", "15 m/s 主全幅最大峰值均位于 B4 上边界 `π rad/s`，C0 的最大值位于 B3 的 `2.258 rad/s`。这是当前测量网格上的离散最大值；不能据此推断边界外响应。", "",
      "## 幅值、约束与控制器配对结果", "",
      "- 0.003/0.006 矩阵在 32 个有配对半幅组中有 10 组超过 5%差异，主要集中于 8–12 m/s 的 B4（另含 8 m/s C4-B3）；显示低速高频段具有明显幅值依赖。", 
      "- 15 m/s 的 C4/C5 共用真实植物，比较按相同速度、频点和相位配对；使用 `summary.json` 的 `c4_c5_same_plant` 中逐点比值/差值，不把跨植物比较解释成控制器收益。",
      "- C4/C5 在 15 m/s 的 B4 峰值分别为 0.27237 与 0.18499，约 0.679 比值；此处 B4 上边界 `σ1` 接近全带最大，且部分其他配置存在近重奇异值，完整奇异向量和质量标签见矩阵 JSON。", "",
      "## 独立单频方向重放", "",
      f"完成 {len(replays)}/15 条按固定规则选择的复右奇异向量重放。选择包含每条件最大 `σ1`、低中频相对退化点，以及首两奇异值近重点（C0 无近重点）。同一 `a=0.006`、`||v||₂=1`，保留复相位。",
      (f"实测复误差向量相对矩阵预测误差的相对范数：中位数 {np.median([r['prediction_error_relative'] for r in replays]):.3g}，最大值 {max(r['prediction_error_relative'] for r in replays):.3g}。" if replays else "重放尚未完成，暂不报告预测误差。"),
      "各方向的选择原因、奇异向量、预测/实测复数分量、运行状态及原始时序见 `direction_replays/`。", "",
      "## 图表与复现", "",
      "`figures/` 中展示奇异值曲线、C5/C4 同植物比值与绝对差、三个物理输入方向增益、逐通道复响应幅值与相位、命令和实际执行器饱和比例、非激励谱能量、主右奇异向量及近重标记、单频方向重放预测对照，以及五个离散航速的频率—航速热图。热图对受约束矩阵留空；曲线连接仅作视觉引导，响应只在标出的离散测量频点有证据。",
      "`protocol.json` 记录频率、相位、启动包络、窗口、质量阈值及代码/仓库/输入哈希；`manifest.json` 汇总文件 SHA-256、运行顺序及状态。分析入口可重建矩阵：",
      "```bash", "python3 experiments/gs12_analaysis/frequency_extension_src/run_frequency_extension.py analyze --repo . --out experiments/gs12_analaysis/frequency_extension", "```", "",
      "## 解释边界", "",
      "这些数据是确定性仿真中参考到跟踪误差的经验频响，不是输出反馈灵敏度或鲁棒稳定性证明。重复周期、频线和数值子步不是独立样本；不报告随机置信区间或失效概率。半幅差超过 5%、执行器受约束的频点均需与无约束线性结论区分。8/10/12/15/18 m/s 是离散直航工作点，不能外推到连续变速、转弯、海流、噪声或未测试的模型参数。", ""]
    (ROOT/"results.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"report":str(ROOT/"results.md"),"direction_replays":len(replays),"matrix_groups":len(matrices)}))


if __name__=="__main__":main()
