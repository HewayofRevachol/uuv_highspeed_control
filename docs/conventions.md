# 命名与数据生命周期

本约定用于新增内容。文件或目录名不能仅靠大小写区分，以保证 Linux、macOS 和 Windows 的检出一致。历史文件名、CSV、checkpoint 与报告保留原路径和大小写；移动旧数据会影响 checkpoint 的路径记录和已有复现命令。

## 命名

| 对象 | 约定 | 示例 |
| --- | --- | --- |
| 新 Python 模块、目录和配置文件 | 小写 `snake_case`；包内保留 `__init__.py` | `research/models/high_drag.py` |
| 类 | `PascalCase`；插件导出名按协议固定 | `DynamicsModel`、`ActorCritic` |
| 函数、参数和变量 | `snake_case` | `create_environment`、`controller_kwargs` |
| 常量 | `UPPER_SNAKE_CASE` | `DEFAULT_MODEL` |
| 实验标识 | `<algorithm>_<model>_<scenario>_s<seed>` | `ddat_uuv_turn_s1027` |
| 实验配置 | 与实验标识同名的 `.json` | `ddat_uuv_turn_s1027.json` |
| 新报告 | `<topic>_YYYYMMDD.md`，日期用 ISO 基本形式 | `turn_comparison_20261103.md` |
| 数据集 ID | 稳定、小写 `snake_case`，不因展示名变化而改动 | `ddat_trajectories` |

实验标识记录方案和主随机种子；重复运行由启动器生成独立时间戳目录。多个评估种子、具体模型参数和训练划分完整保存在配置中，不依靠文件名承载全部信息。已有 `ddat_cpu_smoke` 等名称继续可用；命名约定用于新实验，启动器仍接受原有名称。

## 数据从生成到归档

1. **登记来源。** 在 `data/catalog.json` 为新增数据集登记 ID、仓库相对路径、用途、划分和来源。外部数据记录获取方式与版本；程序生成数据记录脚本、参数和种子。未知信息显式写明，不根据文件名推断。
2. **固定输入。** 新增长期使用的数据可放在 `data/datasets/<dataset_id>/`；临时生成物放在 `outputs/trajectories/<experiment_id>/`。这两个路径是新数据的约定，按需创建。保留已有三组 CSV 的原位置，不覆盖用于已发表或已归档结果的文件。
3. **明确划分。** 在数据目录中记录训练、验证、测试或压力评估用途。划分应按轨迹/场景隔离；不能仅随机拆分同一条轨迹的相邻采样点并声称泛化。压力集反复用于调参后，应记录其开发用途并增加新的留出集。
4. **运行与验证。** 配置记录数据实际路径、模型/控制器/网络模块、后端、种子和海流/艇体设置。配对比较使用相同路径与扰动条件。烟雾测试验证程序链路；正式结论需要足够回合和独立评估。
5. **归档证据。** 保留 `run.json`、输入配置、checkpoint、训练/评估指标、日志和报告。共享结果时同时记录源码版本、依赖/设备信息、实际评估路径及种子；启动器的源码哈希只覆盖其列出的 Python 源文件，不替代数据版本记录。

目录注册并不会复制、下载或加载数据，也不提供数据版本管理功能。若必须搬迁已有数据，先保留旧位置兼容映射、更新调用方并验证历史 checkpoint；不要仅修改目录清单后删除原文件。

## CSV 与元数据

轨迹使用 NED 坐标，位置列为 `north_m,east_m,depth_m`，目标切线速度列为 `target_tangent_speed_mps`。显式姿态列为 `target_pitch_rad,target_yaw_rad`，单位为弧度；未提供时由既有路径加载器推导。每条路径应记录生成几何、速度范围、坐标与单位，尽可能保留 manifest 和生成配置。

数据清单的 `schema_version` 标记清单结构版本。每个条目必须含 `id`、`kind`、`path`、`purpose`、`splits`、`provenance`；全部路径相对于仓库根目录。数据条目中的 `splits` 将划分名映射为路径对象；无训练划分的报告与结果使用空对象 `{}`。历史结果即使没有完整生成配置，也应登记现有来源说明并明确缺项。
