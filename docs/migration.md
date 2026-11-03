# 迁移映射与兼容性

此次整理把训练和验证实现移入独立包，并补齐模型扩展与文档入口。默认算法和历史数据位置保持一致。

| 原位置或职责 | 现行位置 | 兼容方式 |
| --- | --- | --- |
| `DDAT_train.py` 中的训练编排 | `training/ddat.py` | 根文件保留 CLI 和导入兼容入口；推荐 `python3 -m training.ddat` |
| `DDAT_train.py` 中的优势估计与 PPO 更新 | `training/ppo.py` | 原入口继续导出 `advantages`、`ppo_update` |
| `DDAT_train.py` 中的路径选择 | `trajectories/selection.py` | 原入口继续导出 `select_paths` |
| `DDAT_train.py` 中的路径指标汇总 | `evaluation/metrics.py` | 原入口继续导出 `summarize_paths` |
| `validate_ddat.py` 中的验证实现 | `evaluation/ddat.py` | 根文件保留 CLI 和导入兼容入口；推荐 `python3 -m evaluation.ddat` |
| 训练/验证中重复构造环境和网络 | `research/variants.py` 的公共工厂 | 两端共享配置解析，保留默认控制器与网络 |
| 内置 `BatchedUUVModel` | 实现在 `simulator/batch_mpc.py`，选择入口为 `research/models/uuv.py` | 默认 `DynamicsModel` 指向原模型；物理参数未因目录整理改变 |
| 根 README 中的开发说明 | `docs/README.md` 及专题指南 | 根 README 保留快速使用与导航 |
| 分散的轨迹和历史资产入口 | `data/catalog.json`、`data/README.md` | 建立索引，实际文件继续位于原路径 |

## 大小写冲突修复

原仓库同时跟踪 `aav_mpc/data/stage0/T1_osqp25.json` 与 `aav_mpc/data/stage0/t1_osqp25.json`，在不区分大小写的文件系统上会互相覆盖。此次保留大写文件及其原内容，将原小写文件原样重命名为 `aav_mpc/data/stage0/t1_osqp25_legacy.json`。两份历史结果均保留；引用原小写文件的外部脚本应更新为带 `_legacy` 的路径。其余历史资产按下表保留原位置。

## 保持原位置的资产

| 路径 | 保留原因 |
| --- | --- |
| `DDAT_trajectories/train/`、`DDAT_trajectories/val/` | 默认训练/验证路径与原 manifest |
| `trajectories/stress_15mps/` | 预设压力验证及历史报告引用 |
| `high_speed_experiments/generated/` | 已归档参考控制训练输入 |
| `outputs/DDAT/run_20261029_114239/` | 随仓库提供的参考 checkpoint |
| `high_speed_experiments/outputs/`、`aav_mpc/data/` | 专题实验原始结果和模型 |
| `academic_search_report/`、`report1030.md`、专题目录内报告 | 保持历史引用和报告上下文 |

缺少 `controller_module`、`network_module`、`model_module` 等新字段的旧配置使用默认组件。历史 checkpoint 的路径解析仍由现有加载逻辑处理；数据目录中的 ID 并不是 CLI 路径别名。

新实验建议直接导入 `training.ddat` 或 `evaluation.ddat`，使用 `research.variants` 创建组件；旧脚本可继续从根入口导入原有符号。根兼容文件仅用于入口适配，后续训练或绘图改动应落在新实现模块中。

历史 README 的部分命令引用未随仓库提供的脚本，当前可运行入口以根 README 和 [测试说明](../tests/README.md) 为准。切换模型、奖励、观测或网络形状属于实验变更，应新建配置与运行目录，不能把该变更解释为纯路径迁移。
