# UUV 高速控制研究工作区

本项目用于 UUV 高速路径跟踪、MPC 与强化学习控制研究。训练、评估、仿真和实验变体各有独立入口；历史 CSV、checkpoint 和报告保留原路径，便于复查已有实验。以下命令均从仓库根目录运行。

## 从这里开始

| 需求 | 入口 |
| --- | --- |
| 了解目录职责和调用关系 | [架构说明](docs/architecture.md) |
| 新增控制器、网络或动力学模型 | [扩展指南](docs/extending.md) |
| 找训练数据、压力轨迹及历史结果 | [数据索引](data/README.md)、[机器可读目录](data/catalog.json) |
| 命名新实验、保存数据和结果 | [命名与数据生命周期](docs/conventions.md) |
| 找迁移前后的代码位置 | [迁移映射](docs/migration.md) |
| 查算法、研究报告和安装说明 | [文档总索引](docs/README.md) |

## 安装与快速运行

使用 Python 3.10+。先安装适配本机 CPU/CUDA 的 PyTorch，再安装通用依赖：

```bash
python3 -m pip install -r requirements-ddat.txt
python3 -m training.ddat --list-training-paths --device cpu
python3 experiments/run.py experiments/configs/ddat_cpu_smoke.json --dry-run
python3 experiments/run.py experiments/configs/ddat_cpu_smoke.json
```

`admm` 是无需额外求解器的默认后端；`osqp` 需要另装 `osqp`，`rti_cpu` 需要 `casadi` 和 `osqp`。CUDA RTI 依赖见 [RTI 说明](README_RTI_CUDA.md)。烟雾实验的 4096 步仅用于检查流程，不能作为控制性能结论。

验证随仓库保留的 checkpoint，并显式选择验证集：

```bash
python3 -m evaluation.ddat --checkpoint outputs/DDAT/run_20261029_114239/final_actor.pt --all-paths --split val --device cpu --no-show --output-dir outputs/experiments/ddat_uuv_val_s101026/manual_eval
```

根目录兼容入口仍可运行：`python3 DDAT_train.py ...`、`python3 validate_ddat.py ...`。直接验证入口未传 `--all-paths` 时仍使用预设压力轨迹；请显式指定 `--path` 或 `--all-paths` 以固定评估范围。

## 代码与实验布局

| 目录 | 职责 |
| --- | --- |
| `training/` | DDAT 训练编排与通用 PPO 更新 |
| `evaluation/` | checkpoint 加载、配对验证、指标与绘图 |
| `simulator/` | UUV 物理模型、批量环境、MPC 求解器、执行器与恢复机制 |
| `research/controllers/` | 控制环境、观测和奖励变体 |
| `research/networks/` | 策略与价值网络变体 |
| `research/models/` | 可选择的动力学模型，默认 UUV 模型保持原行为 |
| `research/variants.py` | 组件加载与环境、网络构造的公共接口 |
| `experiments/` | JSON 实验配置及独立运行目录启动器 |
| `data/` | 数据和历史研究资产的统一目录，不重复存放旧文件 |
| `docs/` | 当前架构、扩展、命名与迁移指南 |
| `tests/` | 单元、集成与性能测试，见 [测试说明](tests/README.md) |
| `outputs/` | 运行日志、模型和分析产物，见 [输出约定](outputs/README.md) |

已有专题实验分别位于 [高速 RL–MPC](high_speed_experiments/README_RL_MPC.md) 和 [AAV-MPC](aav_mpc/README.md)。它们保留独立的复现脚本和研究记录；文献与证据审查从 [文档总索引](docs/README.md) 进入。

复制 `experiments/configs/` 中的 JSON，以 `<algorithm>_<model>_<scenario>_s<seed>` 命名新实验。每次启动会建立 `outputs/experiments/<name>/<时间_进程号>/`，并记录 `run.json`、`console.log` 和对应产物。多个方案可以使用各自配置并行运行。比较时固定路径划分、种子、海流和艇体参数，用配对验证评估差异。
