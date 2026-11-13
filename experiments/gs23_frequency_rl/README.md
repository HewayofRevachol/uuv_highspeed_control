# GS-23：频率与模型偏差感知的 β 调度实验

按 `docs/gs22_frequency_rl/README.md` 实现。**本次仅交付代码和最小功能检查，没有启动 S0、PPO 或留出实验，没有生成训练权重或真实训练奖励曲线。** 下列命令由使用者明确执行后才运行实验。

## 运行

从仓库根目录运行；需要 CUDA PyTorch、NumPy、CasADi、tqdm、Matplotlib。测试另需 pytest、SciPy、OSQP。见 `requirements.txt`。默认使用 CuSADi 的解析雅可比和 CUDA Graph，要求 CUDA Toolkit 的 `nvcc`，以及项目原有固定版本的 CuSADi：

```bash
git clone https://github.com/se-hwan/cusadi.git .third_party/cusadi
git -C .third_party/cusadi checkout bffea25849a901277a8f8dc4db16748e629f9053
python3 -m experiments.gs23_frequency_rl.train --out outputs/gs23_frequency_rl_run1
```

已有 CuSADi 可设置 `CUSADI_ROOT`，编译目录可设置 `UUV_RTI_BUILD`，编译器可设置 `NVCC`；默认构建缓存是仓库下 `.rti_build/`。输出目录必须尚不存在。仅做机会检查时加 `--s0-only`，仍会运行 20 条完整轨迹。

没有 CuSADi 时可用 `--predictor torch`，解析雅可比改用 `torch.func.jacfwd/vmap`；功能一致但该模式更慢。CPU 功能排查可用 `--device cpu --predictor torch --no-cuda-graph`。默认不作 CPU 自动回退。

`tqdm` 显示每组并行轨迹的仿真秒进度；训练批次为 8 个 lane，开发检查为 4 个，留出为 8 个。各 lane 的对象、β、指令历史、QP 计划和终止掩码独立。GPU 并行发生在 lane、矩阵运算和 ADMM 的线性求解内，时间仍按因果顺序推进。

## 实验顺序与停止条件

启动时先写 `protocol.json`，冻结开发 4 条、训练 16 条、留出 8 条任务及代码哈希，所有任务 144 秒。每回合从 β=目标 β=0.5 开始，前 8 秒固定；后续目标每秒更新，每 0.1 秒最多变化 0.02。

1. S0：开发任务比较固定 0/0.5/1 和两条诊断调度 `0.5*h(t+L)`（L=0/2.5 秒），共 20 条。最佳调度对最佳固定参数须平均 Jtrack 至少低 5%、至少 3/4 配对改善，且两者都完整、无 QP 失败或饱和，否则停止。
2. S1：固定种子 41001，两个独立乱序遍历训练清单，32 回合、每 8 回合更新一次。78→64→64 tanh 网络，初始均值动作 β=0.5、标准差 0.15；γ=1，GAE λ=0.95，学习率 3e-4，4 epochs，clip=0.2，minibatch=64。
3. S2：四个 PPO 检查点各评四条开发任务，共 16 条。按失败数、发生饱和的回合数、平均 Jtrack 依次排序。初始常数策略复用 S0，学习策略没有超过固定开发基线时停止，不打开留出集。
4. S3：冻结留出 8 条，比较最佳固定参数、选定诊断调度、β=0.5、选定 PPO，共 32 条。只评一次，不再据此调参。整个流程最多 100 条。

诊断规则只用于有已知参考分解的机会探针，频段标签和对象类型不进入 Actor/Critic。训练阶段不重复开启旧 GS-21 E/F，也不保证这次会通过机会门槛。

## GPU MPC 和模型一致性

`mpc.py` 沿用 first 22 状态的单次 RTI 线性化，35 步预测、每步 0.1 秒、105 个凝聚变量。权重为状态 `[3,3,3,100,30,5,5,5]`、输入 `[.15,.08,.08]`、输入变化 `[1.8,.35,.35]`，末步状态权重乘 2.5；这些与 GS-21 保存的 first 控制器一致。

QP 是 `min 0.5*xᵀHx+gᵀx, l≤x≤u`，约束为实际指令推进 `[0,1]`、双舵 `[-1,1]`。ADMM 每轮用批量 Cholesky 解 `(H+ρI)x=-g+ρ(z-y)`，再做 box 投影和对偶更新。默认 ρ=10，400 次固定迭代，原始和对偶无穷范数均须 ≤1e-5，否则该 lane 立即记 QP 失败。可通过命令行指定迭代数和容差，参数写入协议；不重试、不切换其他求解器。

模型、QP 与积分使用 float64，策略使用 float32。CUDA Graph 缓存 QP 组装、ADMM 迭代及每周期的五个对象积分子步，避免每轮 ADMM 都由 Python 提交 CUDA 操作。CuSADi 编译缓存按符号函数哈希命名。该实现降低启动开销并并行求解，**未通过训练或吞吐基准验证，不承诺固定 GPU 利用率或加速倍数**；特别是 4/8 lane 和消费级 GPU 的 FP64 性能会限制吞吐。

真实对象在 lane 内分为 matched（一阶执行器）和 combined（双舵效能×0.9、全执行器二阶）。二者都相对 `MISMATCHED_PLANT` 基准定义，不能将 matched 解释为理想无滞后对象。初态、三槽延迟和 0.02 秒 RK4 沿用已有对象。

**GS-22 的一处接口澄清：** 文档奖励引用了“原 MPC 增量约束”，但 GS-21 first 控制器实际未调用 `set_command_slew_limits`，默认 `max_normalized_slew=None`，只有变化惩罚。本实现保持这一约束集合，不新增硬限幅；奖励中的 Δc 采用指令全量程 `[1,2,2]` 归一化，作为明确的新实验参数登记。新结果须在本后端内部配对，不能与旧 OSQP 结果直接宣称完全等同。

## 观测、奖励和结果

`features.py` 按 GS-22 的 `[0:10]` 至 `[73:78]` 顺序拼接全部 78 维；参考/误差频率系数使用预计算的 80×6 最小二乘矩阵，历史相位移到当前时刻，历史不足 80 点时频率误差块置零。执行器速率用反馈后向差分经 0.2 秒滤波；创新使用共同 β=0.5 模型对上周期实际命令的预测。MPC 与观测使用自己的已发命令历史，不读取对象内部的真实速率或参数。未使用未来测量。

每 0.02 秒积分裁剪后的阶段代价，含 1e6 主跟踪误差、1e-3 β 变化和 1e-3 指令变化。奖励除以预定 144 秒；失败补齐剩余时长的最坏阶段代价再罚 10。首 8 秒奖励计入总回报，但不将被强制覆盖的策略动作放入 PPO 更新。提前失败后的 lane 不参与 GAE；完整回合末尾价值为零。PPO 保存 Gaussian latent，以其概率计算 ratio，绝不把 β 当作高斯变量。

输出文件：

| 文件 | 内容 |
| --- | --- |
| `protocol.json` | 数据划分、种子、求解器、奖励、权重、观测版本、源码哈希 |
| `baseline_05.pt` | 初始常数 β=0.5 检查点；只有明确执行实验后才写出 |
| `checkpoint_01.pt` … `checkpoint_04.pt` | 每次 PPO 更新后的网络、优化器和协议 |
| `final_actor.pt` | 选中策略；停止时可能为常数，以 `policy_type` 和 `constant_beta` 为准 |
| `rewards.csv`, `reward_curve.png` | 逐训练回合总回报及每批均值；S0 停止时曲线明确标注 PPO 未运行 |
| `*_trace.npz` | 每 0.1 秒的 β、目标、指令、三轴误差、活跃掩码和 QP 残差，含列名 |
| `*.json`, `summary.json`, `updates.json` | 各方法逐任务指标、门槛/停止原因、PPO loss |
| `holdout.json` | 仅触发 S3 时生成，保留全部任务和失败分母 |

权重加载示例，常数分支必须尊重元数据：

```python
from experiments.gs23_frequency_rl.train import load_policy
import torch
agent, info = load_policy('outputs/gs23_frequency_rl_run1/final_actor.pt')
with torch.no_grad():
    target = (torch.full((len(observation),), info['constant_beta'], device=observation.device)
              if info['policy_type'] == 'constant' else agent.deterministic(observation))
```

评价中的 Jtrack、RMSE 和最大误差均不裁剪；提前失败任务的这些指标只描述已观测前缀，`failed` 必须先看。物理 RMSE 顺序为 N/E/D/pitch/yaw/VN/VE/VD，另有体轴 v/w/q/r RMS、饱和周期数、β/命令总变差。`windows` 给出低频、高频、参考混合过渡和调参恢复窗口；参考混合区间为 `[46,50)`、`[94,98)`，恢复窗为最近实际 β 变化后的 8 秒（排除参考混合区间），是一项冻结的诊断定义，不代表已证明系统恢复稳态。原始轨迹允许另行分析。

## 最小检查（不训练）

```bash
python3 -m pytest experiments/gs23_frequency_rl/test_contract.py -q
python3 -m experiments.gs23_frequency_rl.train --help
```

检查覆盖双频投影和历史相位、参考导数、78 维因果观测、动作中心和限速、失败代价、终止 lane 的 GAE、S0 门槛、PNG/CSV 导出，及 CPU/CUDA ADMM 对 OSQP 的受限 QP 精度。CUDA 集成检查仅推进两次 MPC 控制周期（0.2 秒），不运行完整轨迹、不更新网络。GPU 检查需要 CuSADi；无 CUDA 时跳过。检验记录见 `VALIDATION.md`。

`dynamics.py`、`cusadi22.py` 和 `mpc.py` 基于工作空间已有 `experiments/gs12_rl_beta_gpu_admm/` 的实现整理；新目录自包含，不要求该旧实验目录被安装或复制。
