# AAV-MPC 复现实验

主报告：[`REPORT_20261102.md`](REPORT_20261102.md)。任务冻结：[`configs/tasks.json`](configs/tasks.json)。所有新脚本、日志、模型和结果均在此目录下；旧项目数据未搬动。

| 阶段 | 入口 | 输出 |
| --- | --- | --- |
| 0 配对归因 | `run_stage0_v2.sh`、`run_stage0_ablations.sh`、`run_cost_ablation.sh`、`run_oracle_ablation.sh` | `data/stage0/` |
| 1 离线可行性 | `run_stage1.sh` | `data/stage1/` |
| 2 增强 MPC | `run_stage2.sh`、`run_condensed.sh`、`run_endpoint_probe.sh` | `data/stage2/` |
| 3 价值与裕度离线检验 | `run_stage3_collection.sh`、`run_stage3_train.sh`、`run_stage3_tune.sh`、`run_stage3_aligned.sh` | `data/stage3/raw/`、`model*/` |
| 4 配对闭环 | `run_stage4_paired.sh`、`/bin/python3 -m aav_mpc.stage4_summary` | `data/stage4/` |

运行环境：`/bin/python3`，当前 Torch 2.14.0+cu130，CPU 运行；训练使用 `tmux -S /tmp/uuv_aav_20261102.sock` 独立会话。查看运行中的实验：

```bash
tmux -S /tmp/uuv_aav_20261102.sock list-sessions
tail -f aav_mpc/data/stage4/paired/run.log
```

任务文件中的 T1/T3 走廊与航速容差是**本轮冻结的示例仿真任务定义**，不代表实艇设计限值。阶段 3 在未见几何上的裕度校准失败，因此模型 checkpoint 保留作负结果复查，当前不接入闭环 MPC。
