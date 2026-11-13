# GS-12 连续 β：GPU 批量 ADMM 训练

此目录提供独立训练入口。一个 CUDA batch 对应一次 PPO 更新的 16 个冻结 task；每个 lane 有独立的 Plant、执行器队列、MPC plan、β 与终止状态。控制周期在时间上顺序推进，lane 之间并行。

## 文件

- `train.py`：冻结任务和 F75 缓存加载、批量 rollout、逐 lane GAE、PPO 更新及 checkpoint。
- `continuous_ppo.py`：19 维观测、一维高斯 latent、`0.75*sigmoid(z)` 动作和 clipped PPO。
- `environment.py`：8 秒决策段、β 限速、参考、奖励及终止掩码。
- `dynamics.py`：22 维一阶执行器预测模型及二阶执行器真实 Plant 的 CUDA 张量实现。
- `cusadi22.py`：参数化 22 维预测器的 CuSADi CUDA 内核生成与加载。
- `mpc.py`：35 步 condensed QP、批量 box ADMM、CUDA Graph 和未收敛重试。

## 依赖和运行

需要 CUDA PyTorch、CasADi、NumPy、tqdm、`nvcc` 和 CuSADi 源码。CuSADi 默认位于仓库根目录 `.third_party/cusadi`，也可用 `CUSADI_ROOT` 指定。源码固定到仓库现有 `setup_rti_cuda.sh` 使用的提交 `bffea25849a901277a8f8dc4db16748e629f9053`。首次运行会在 `.rti_build` 生成并编译内核。此实现使用 dense box QP，无需 cuDSS。

若工作区尚无 CuSADi，可从仓库根目录准备：

```bash
git clone https://github.com/se-hwan/cusadi.git .third_party/cusadi
git -C .third_party/cusadi checkout bffea25849a901277a8f8dc4db16748e629f9053
```

从仓库根目录运行：

```bash
python3 -m experiments.gs12_rl_beta_gpu_admm.train \
  --source experiments/gs12_rl_beta_pilot/output \
  --out experiments/gs12_rl_beta_gpu_admm/output_run1 \
  --episodes 64 --update-episodes 16 --predictor cusadi
```

默认 ADMM 先进行 100 次迭代；若某个活跃 lane 未达到残差阈值，则整批用至少 400 次迭代重试。GPU QP 组装和 ADMM 使用 float32，预测器与 Plant 使用 float64。`--predictor torch` 可在没有 CuSADi 时运行，但其雅可比计算很慢，不适合作为性能训练入口。`--no-cuda-graph` 可关闭图捕获。

输出包含 `checkpoint_*.pt`、`final_actor.pt`、`history.json`、`episodes.jsonl` 和 `run.json`。输出目录必须为空。训练默认读取 pilot 冻结的 task 和 F75 奖励基线；此目录的结果应作为独立 GPU ADMM 实验记录。

## 同任务评估

`compare_mpc_rl.py` 默认在冻结的 8 条 dev 任务上比较从 β=0 开始的纯 MPC、固定 β=0.75 的 MPC，以及训练完成的 RL+MPC。三种方法使用相同的 GPU ADMM 后端，RL 使用确定性动作。运行：

```bash
python3 -m experiments.gs12_rl_beta_gpu_admm.compare_mpc_rl \
  --checkpoint outputs/gs12_gpu_admm_20261112T051540Z/formal_1h/final_actor.pt \
  --out outputs/gs12_gpu_admm_20261112T051540Z/comparison_dev
```

结果包含逐任务 `episodes.csv`、`episodes.jsonl`、汇总 `summary.json` 和 `REPORT.md`。要评估其他任务清单，可传 `--tasks`；输出目录仍须为空。脚本不会自动运行锁定测试集。
