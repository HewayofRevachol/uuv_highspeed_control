# GS-12 MPC 权重路线实验

本目录按 [`GS-12_短期研究路线_5天冲刺版.md`](../gs12_analaysis/GS-12_短期研究路线_5天冲刺版.md) 执行。完整结论见 [`REPORT.md`](REPORT.md)，冻结协议、任务、运行记录、原始轨迹、分析表和图都在 `output/`。

## 本次结论

在 15 m/s 的 C4 真实植物和 C4 预测器下，静态权重 `(a_D,a_Δδ)=(1.4,0.7)` 在三条预先冻结的留出任务上，相对原始 C4 的等任务平均 `J_track` 降低 13.46%；每条任务均未恶化，没有新增早停、QP 失败或执行器饱和，达到路线预设的 5% 工程门槛。这个结果只支持本协议中的固定权重调参。开发集未显示动态频谱调度的独立必要性，因此没有实现频谱规则或扩训 RL。

完成了 39 条可选 256 s 长协议运行，重建 B1/B4 三轴经验频响矩阵，所有运行和矩阵组完整。

## 代码与数据

- `reference_tasks.py`：单频、平滑混合及 B1/B4 历史协议参考生成。
- `runner_core.py`：复用研究植物并在每次 MPC 求解前设置两个权重倍率。
- `run_weight_route.py`：冻结主协议、吞吐 pilot、开发和锁定阶段、长协议批次。
- `analyze_weight_route.py`、`analyze_long_matrix.py`：外部 `J_track` 与配对经验频响分析。
- `verify_route.py`：路线要求的导数、QP 接线和旧 runner 轨迹一致性检查。
- `rekey_runs.py`：将输出按含协议哈希的完整主键命名；不会覆盖不完整或校验失败的同名运行。
- `output/protocol.json`、`output/tasks.jsonl`：冻结的主协议与参考任务。
- `output/selection.json`、`output/test_locked_decision.json`：开发候选冻结及留出门槛判定。
- `output/long_protocol.json`、`output/long_matrix/`：可选长协议及 39 条原始轨迹。
- `output/integrity_audit.json`：最终文件数、SHA-256 和运行状态审计。

## 从新输出目录复现

为避免混合不同代码版本的结果，复现实验请使用新的输出目录。以下命令在仓库根目录运行：

```bash
export GS12_WEIGHT_OUT="$PWD/experiments/gs12_weight_route/reproduction_output"
python3 experiments/gs12_weight_route/verify_route.py
python3 experiments/gs12_weight_route/run_weight_route.py prepare
python3 experiments/gs12_weight_route/run_weight_route.py pilot
python3 experiments/gs12_weight_route/run_weight_route.py run dev_scan --workers 4
python3 experiments/gs12_weight_route/analyze_weight_route.py
python3 experiments/gs12_weight_route/run_weight_route.py run dev_rule --workers 4
python3 experiments/gs12_weight_route/analyze_weight_route.py
python3 experiments/gs12_weight_route/run_weight_route.py run test_locked --workers 4
python3 experiments/gs12_weight_route/analyze_weight_route.py
python3 experiments/gs12_weight_route/run_weight_route.py run long_matrix --workers 4
python3 experiments/gs12_weight_route/analyze_long_matrix.py
python3 experiments/gs12_weight_route/plot_results.py
python3 experiments/gs12_weight_route/finalize_outputs.py
```

在最终候选冻结前，`test_locked` 阶段不会启动；在留出门槛通过前，长协议阶段不会启动。worker 数应以本机重新测得的 pilot 为准。
