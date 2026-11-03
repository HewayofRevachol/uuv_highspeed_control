# UUV 控制与学习实验工作区

本目录保留原有训练、验证命令及轨迹数据，同时将后续研究用的控制环境、策略网络和实验配置分开。所有命令建议从项目根目录运行。

## 目录

| 位置 | 用途 |
| --- | --- |
| `DDAT_train.py`、`validate_ddat.py` | 训练与配对验证入口，仍可直接运行 |
| `simulator/` | 艇体动力学、批量环境、MPC 求解器、执行器预测与恢复机制 |
| `research/controllers/` | 实验级控制环境及后续控制器变体 |
| `research/networks/` | DDAT 策略网络及后续网络变体 |
| `research/variants.py` | 根据配置或 checkpoint 载入上述变体 |
| `DDAT_trajectories/`、`trajectories/` | 现有轨迹集、轨迹生成及路径编码工具 |
| `high_speed_experiments/` | 高速轨迹筛查与 [RL 增强 MPC 实验](high_speed_experiments/README_RL_MPC.md) |
| [`academic_search_report/`](academic_search_report/README.md) | 文献综述、推荐路线与证据核验报告 |
| `experiments/configs/` | 一次实验的参数文件，复制后改名即可建立新方案 |
| `experiments/run.py` | 创建独立运行目录、记录命令与日志 |
| `tests/unit/`、`tests/integration/`、`tests/benchmarks/` | 新增单元、集成与性能测试脚本 |
| `outputs/` | 模型、训练、验证、图表、报告、基准和生成轨迹输出，见 `outputs/README.md` |

## 快速使用

先按 `README_打包说明.md` 安装依赖。直接运行原有入口：

```bash
python3 DDAT_train.py --list-training-paths --device cpu
python3 validate_ddat.py --checkpoint outputs/DDAT/run_20261029_114239/final_actor.pt --device cpu --no-show
```

按配置启动独立实验：

```bash
python3 experiments/run.py experiments/configs/ddat_cpu_smoke.json --dry-run
python3 experiments/run.py experiments/configs/ddat_cpu_smoke.json
```

复制 JSON 配置并改动 `name`、`arguments`，即可在不同终端同时运行不同方案。每次运行进入 `outputs/experiments/<name>/<时间_进程号>/`；其中 `run.json` 记录启动命令、状态和源码哈希，`console.log` 保存控制台输出，其余训练或验证文件也写入该目录。验证配置必须指定 checkpoint；`all_paths: true` 会使用所选 `split` 的路径集。

已有 `ddat_cpu_smoke.json` 与 `ddat_dense_heading_smoke.json` 是相同种子和训练规模下的两种预览、奖励设置示例；可以分别启动它们。配置里的 4096 步只用于流程检查，正式比较需扩大步数，并用相同路径、种子及验证条件进行配对评估。

## 修改位置与比较方法

- 改艇体参数、动力学：`simulator/model.py`、`simulator/batch_mpc.py`。
- 改 MPC 求解：`simulator/batch_mpc.py`、`simulator/rti_mpc.py`、`simulator/rti_symbolic.py`、`simulator/rti_cuda.py`。
- 改低层控制和在线估计：`simulator/ddat_env.py`；改预览或奖励：`research/controllers/ddat.py`。
- 改策略网络：`research/networks/ddat.py`；改 PPO 训练：`DDAT_train.py`。
- 改轨迹和数据处理：`DDAT_trajectories/`、`trajectories/`。

比较方案时，建议在 `research/controllers/` 或 `research/networks/` 中新增模块，分别提供 `ExperimentDDATPathEnv` 或 `ActorCritic` 类，再在实验配置中设置 `controller_module`、`network_module`。训练会把模块名保存到 checkpoint，验证脚本据此加载。保持环境构造参数、`reset/observe/step` 接口及网络的 `deterministic/value/sample/log_prob` 接口；若改变观测或网络结构，请用新方案名训练自己的 checkpoint。现有 checkpoint 默认使用 `research.controllers.ddat` 与 `research.networks.ddat`，无需转换。

现有 `outputs/DDAT/run_20261029_114239/` 是随包提供的参考 checkpoint；新实验放在 `outputs/experiments/`，便于并行比较。算法说明见 `README_DDAT.md`；RTI CUDA 说明见 `README_RTI_CUDA.md`。这份打包快照不含那些文档中提到的部分早期脚本。
