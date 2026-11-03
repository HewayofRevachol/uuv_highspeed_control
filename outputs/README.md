# 输出与归档约定

| 目录 | 内容 |
| --- | --- |
| `DDAT/` | 原有 DDAT checkpoint，保留路径以兼容已有验证命令 |
| `experiments/` | `experiments/run.py` 创建的独立训练与验证运行 |
| `models/` | 手工导出的模型、权重或中间表示 |
| `benchmarks/` | 性能测试结果与耗时记录 |
| `figures/` | 图像、动画和可视化 |
| `reports/` | 新汇总表、分析报告 |
| `trajectories/` | 新生成的临时轨迹或语料 |

新实验名称使用 `<algorithm>_<model>_<scenario>_s<seed>`。启动器创建 `experiments/<name>/<时间_进程号>/`，其中 `run.json` 保存输入配置、命令、源码哈希及运行状态，`console.log` 保存控制台输出。训练和验证产物写入同一运行目录；不要让多个进程共用手工指定的输出目录。

长期归档至少包括配置、种子、使用的数据路径/版本、checkpoint、指标、日志、源码版本和依赖环境。启动器不自动打包依赖或为 CSV 计算版本；报告需补齐这些信息。正式训练中断时保留已有运行记录，再用新的运行目录开展后续实验。

历史结果仍位于 `high_speed_experiments/outputs/` 和 `aav_mpc/data/`，统一入口见 [数据索引](../data/README.md)。历史研究报告不搬迁；新报告使用 `<topic>_YYYYMMDD.md`。审查报告见 [仿真与现实性审查](../academic_search_report/AUDIT_UUV_MECHANISM_REALISM_20261102.md)，其所述 `outputs/reports/uuv_mechanism_audit_20261102/` 不在当前仓库中。
