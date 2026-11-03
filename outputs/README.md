# 输出约定

| 目录 | 建议内容 |
| --- | --- |
| `DDAT/` | 原有 DDAT checkpoint，保留原路径以兼容现有验证命令 |
| `experiments/` | `experiments/run.py` 创建的隔离训练与验证运行 |
| `models/` | 手工导出的模型、权重或中间表示 |
| `benchmarks/` | 性能测试结果与耗时记录 |
| `figures/` | 图像、动画和可视化 |
| `reports/` | 汇总表、分析报告 |
| `trajectories/` | 新生成的轨迹或语料；原始轨迹留在数据目录 |

每个方案应有独立子目录。不要让多个进程写入同一个运行目录；启动脚本会自动生成唯一目录。需要长期保留的结果请连同其 `run.json`、配置、种子和 checkpoint 一起归档。

执行器状态感知 RL–MPC 路线的实跑审查结果集中在 [`reports/uuv_mechanism_audit_20261102/`](reports/uuv_mechanism_audit_20261102/)；分析报告见 [`academic_search_report/AUDIT_UUV_MECHANISM_REALISM_20261102.md`](../academic_search_report/AUDIT_UUV_MECHANISM_REALISM_20261102.md)。
