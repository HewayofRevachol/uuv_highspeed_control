# 数据与历史资产索引

[`catalog.json`](catalog.json) 集中登记训练数据、压力轨迹、参考 checkpoint、历史结果和报告。真实文件保留在原路径，以兼容已有命令和 checkpoint；目录清单不复制数据，也不改变训练脚本的路径解析。

## 轨迹集

| ID | 仓库路径 | 划分与用途 | 来源 |
| --- | --- | --- | --- |
| `ddat_trajectories` | `DDAT_trajectories/` | 20 条 train，5 条 val，五类机动 | [生成器](../DDAT_trajectories/generate.py)、[manifest](../DDAT_trajectories/manifest.json)、[设计说明](../DDAT_trajectories/README.md) |
| `stress_15mps` | `trajectories/stress_15mps/` | 4 条压力评估轨迹，已有多次历史研究使用记录 | [来源说明](../high_speed_experiments/REPORT_RL_MPC_20261102.md)；原生成命令与种子未完整记录 |
| `reference_curriculum_1102` | `high_speed_experiments/generated/reference_curriculum_1102/` | 24 条参考控制训练轨迹，本组不单列验证集 | [生成器](../high_speed_experiments/generate_reference_curriculum.py)、[实验报告](../high_speed_experiments/REPORT_RL_MPC_20261102.md) |

默认 DDAT 训练使用第一组；压力集不是默认训练集。参考课程由四种半径、三种入弯长度、两种坡度组合而成；V2/V3 留出压力任务的设计见原报告。不要把所有 CSV 合并后再宣称保留了原验证划分。

程序生成数据的原目录保留用于复现。新生成副本建议使用独立输出位置，例如：

```bash
python3 high_speed_experiments/generate_reference_curriculum.py --output outputs/trajectories/reference_curriculum_20261103
```

DDAT 的 `generate.py` 默认写入既有训练/验证目录；重建前检查脚本的输出位置，避免在已有实验中途覆盖输入。

## checkpoint、结果与报告

| ID | 位置及内容 |
| --- | --- |
| `ddat_reference_checkpoint` | `outputs/DDAT/run_20261029_114239/`：`final_actor.pt`、配置与训练路径 |
| `high_speed_reference_results` | `high_speed_experiments/outputs/`：参考策略模型、训练历史、评估 JSONL 和日志 |
| `aav_mpc_results` | `aav_mpc/data/`：stage0 至 stage4 的研究产物；stage 不是数据划分 |
| `academic_research_reports` | `academic_search_report/`：文献、研究方向与证据审查 |
| `ddat_stage_report` | `report1030.md`：基础阶段实验记录 |
| `high_speed_research_reports` | `high_speed_experiments/` 中 catalog 明列的三个报告文件 |
| `aav_mpc_report` | `aav_mpc/REPORT_20261102.md`：分阶段复现实验报告 |

报告的可点击入口见 [文档总索引](../docs/README.md)。不在仓库中的历史产物不登记为可用路径；例如现实性审查报告提及的 `outputs/reports/uuv_mechanism_audit_20261102/` 未收录。原 AAV stage0 中仅大小写不同的两个文件已按 [迁移映射](../docs/migration.md) 分别保留为 `T1_osqp25.json` 与 `t1_osqp25_legacy.json`。

## 清单格式与新增数据

顶层 `schema_version` 当前为 `1`，`datasets` 是资产条目数组。每个条目的 `id` 是稳定小写标识，`kind` 区分 `trajectories`、`checkpoint`、`experiment_results`、`reports`，`path` 相对于仓库根目录。`purpose` 描述用途；`splits` 将划分名映射为包含 `path` 的对象，无训练划分的资产用 `{}`；`provenance` 保存已知生成器、配置、说明和缺项。

`file_count` 是当前轨迹快照的 CSV 数量；新增或删除轨迹时应同步更新。报告集合可以使用 `files` 明列成员。清单中的 ID 仅用于查找，传给 CLI 的仍是实际路径。

新数据先记录来源、单位、生成参数和划分，再登记清单；长期输入可按需创建 `data/datasets/<dataset_id>/`，临时产物使用 `outputs/trajectories/`。完整约定见 [命名与数据生命周期](../docs/conventions.md)。
