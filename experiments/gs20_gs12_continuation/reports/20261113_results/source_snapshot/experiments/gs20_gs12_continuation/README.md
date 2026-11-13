# GS-12 延续实验：多路径 RMSE 与预测器调度

本目录延续 GS-12 原植物、原 MPC 权重、N35、0.1秒控制和0.02秒积分。新增三轴/耦合路径、物理 RMSE、同结构固定β/规则/RL对照，β范围为0–1。

**交付状态：代码已编写，只完成最小检查；未运行仿真、求解QP或训练。** 不能把本目录当作已有实验结果。完整顺序与命令见 [STEPS.md](STEPS.md)，上轮协议保留在 [PROTOCOL.md](PROTOCOL.md)。

| 文件 | 用途 |
|---|---|
| `prepare.py`、`references.py` | 固定任务清单及解析一致的参考路径 |
| `closed_loop.py`、`run.py` | 原植物、冻结QP、22状态/混合预测器；固定β、规则和RL评价 |
| `metrics.py`、`analyze.py` | 物理RMSE、归一化合成RMSE、共同前缀、冻结基线与配对对照 |
| `frequency.py` | 原七频3×3复频响、SVD、重复窗口和旧数据连接检查 |
| `plot.py` | 从已保存CSV生成β—频率—物理RMSE热图 |
| `policy.py`、`train.py` | 19维观测、0–1连续β、CPU PPO训练 |
| `check.py` | 导入、任务、解析导数、合成RMSE与源码哈希最小检查 |
| `config.json`、`legacy_sources.json` | 固定参数及原代码来源 |
| `legacy/` | GS-12原植物/结构辅助代码及原QP的私有副本 |

所有命令均从 `uuv_workspace` 根目录执行。`check`只做轻量检查，`prepare`只生成输入；`frequency`、`analyze`、`plot`读取已保存结果，不自动补跑。`run`和`train`才会推进植物。

```bash
python3 -m experiments.gs20_gs12_continuation.check
```

依赖为工作区已有的 NumPy、PyTorch、SciPy、OSQP、CasADi；绘图另用 Matplotlib，不安装新后端。原模型文件哈希由最小检查核对，运行时另保存实际依赖版本。不同依赖版本是否影响旧数值，需由步骤A的实际复现判断。此次检查记录见 [CHECKS.md](CHECKS.md)。

本次仅在此目录添加文件，未修改生产代码或其他实验。无需运行已有GPU实验，也不使用其旧β≤0.75策略充当本轮新RL。
