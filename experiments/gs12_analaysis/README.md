# GS-12 分析附件与复现

先读 `GS-12_UUV模型失配影响分析.md`。这是当前代码的分析交付，不是生产修复或新的训练结果。

解压后保留 `gs12_analysis/` 与 `uuv_highspeed_control/` 同级。控制代码使用提交 `d001f99`（完整 hash 在 `paired_results/protocol.json`）。Python 3.11；依赖版本记录反映本次实际环境，平台为 macOS arm64。其他平台选择可用 PyTorch 构建并记录差异。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r gs12_analysis/requirements-analysis.txt
cd uuv_highspeed_control
../.venv/bin/python ../gs12_analysis/paired_mismatch.py \
  --repo . --out ../gs12_analysis/reproduced_results --duration 24
../.venv/bin/python ../gs12_analysis/control_audit_probe.py
../.venv/bin/python -m pytest -q \
  tests/unit/test_h15_residual_predictor.py tests/unit/test_h15_target_domain.py
cd ..
.venv/bin/python gs12_analysis/make_figures.py
.venv/bin/python gs12_analysis/reproduce_step_response.py --repo uuv_highspeed_control
```

`paired_mismatch.py` 拒绝覆盖已有输出目录。绘图脚本默认读取本附件的 `paired_results`，不重新运行 MPC；如需绘制新运行，修改其 RESULTS 路径。`control_audit_probe.py` 的 E0 对照若已有 episode.json 会复用；独立 tau/安装顺序探针每次重新计算。要重跑 E0，可将旧目录另行改名。

`paired_results/protocol.json` 记录实际执行脚本 hash 和交付脚本 hash：执行后仅更正文档字符串里的论文公式编号，控制与统计代码未变。`script_sha256` 是执行时版本；`packaged_script_sha256` 是交付版本。

文件说明：

- `paired_results/*_*.json`：12 条闭环轨迹；每周期有完整艇体状态、参考、误差、命令与实际执行器。
- `paired_results/summary.json`：各失配案例相对自己设计闭环的共同前缀 VR、RMSE、均值、边界与饱和统计。
- `common_window_metrics.json`：不同控制器/植物在全体共同时间窗上的误差统计；coupled 为 0.1–5.1 s。
- `control_audit_probe.json`：tau 重复缩放、旧入口安装顺序以及 E0 激励覆盖的复核。
- `matched_nominal_e0_existing/`：现有 E0 对照的原始输出，内含运行时本地路径，仅作为溯源字段。
- `step_response.json`：仓库现有阶跃函数在 12/15/18 m/s 的复算。
- `paper_notes.md` / `code_audit_notes.md`：论文与代码证据细节。
- `mismatch_effects.png` / `.pdf`：可直接分享的曲线图。
- `artifact_manifest.json`：附件包内文件 SHA-256；不含论文原始 PDF、仓库源码或 Python 环境。

这里的 VR 是有限确定性窗口的中心方差比，包含启动瞬态，不是稳态随机方差或 RSI。两个控制器的 VR 分母不同；比较控制器时应使用同一真实植物及共同时间窗口的误差/边界结果。输出原始方向为 east/depth，不把 east 直接命名为最近路径的横向误差。
