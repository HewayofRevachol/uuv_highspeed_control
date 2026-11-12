# 连续 β PPO 流程与 ADMM 批量环境边界

本文按 `experiments/gs12_rl_beta_pilot/train_ppo_continuous.py`、`episode_beta.py` 和 `experiments/gs12_weight_route/runner_core.py` 的实际代码说明训练计算，并解释为什么不能直接将另一套 ADMM 批量环境当作等价替代。这里的“不能直接”是指没有闭环转移等价验证前不能替换；不是说 ADMM 算法永远不能用于 GS-12。

## 训练对象

- 输入是离散 pilot 已冻结的 tasks_train.jsonl 和 reward_baseline_cache.jsonl。每个 task 使用同任务完整 64 秒 F75 的 J 作为训练 reward 基线；脚本检查基线存在、有效且有限。
- 默认取冻结训练清单前 64 个 task，每 16 个 task 做一次 PPO 更新，共 4 次更新。每个 task 持续一条独立 Plant/MPC 会话。
- 一次高层决策保持 8 秒，即 80 次 0.1 秒 MPC solve。完整 task 是 8 段、640 次 MPC solve、最多 3200 个 0.02 秒 Plant 子步。
- actor 从 19 维观测产生一维高斯 latent z。连续动作是 beta_target = 0.75 * sigmoid(z)，在 (0, 0.75) 内。部署确定性动作由高斯均值经过同一映射得到。
- beta_target 在 8 秒段内固定；每个 MPC 周期实际 beta 最多变化 0.02。实际 beta 受限速影响，可以滞后于目标。

PPO 的 log probability 在未压缩 latent z 上计算，rollout 和更新使用相同 Normal 分布口径。由于同一动作的 sigmoid 变换项在新旧策略概率之比中相消，ratio 可在 z 空间计算；熵正则也是 latent 高斯的熵。初始实际 beta 与上一目标均为 0.75。

## 一批训练的计算顺序

```text
冻结 task 和同任务 F75 J
        |
复制当前 policy 参数，形成同一版本的 CPU worker 快照
        |
并行运行独立完整 episode：观测 -> 高斯采样 -> 连续 beta 目标 -> 闭环 8 秒段
        |
每个 episode 独立计算 GAE 和 return，避免跨任务递推
        |
将最多 16 x 8 条 transition 转为 CPU/CUDA tensor
        |
4 epochs clipped PPO，minibatch 64，更新 actor 和 critic
        |
保存 checkpoint；下一批使用更新后的同一 policy 版本
        |
所有批次结束保存 final_actor.pt
```

### Episode 采样

父进程为本批每个 task 提交一个 rollout job。进程池最多同时运行 `--workers` 个 job，每个 job 加载相同的冻结策略快照，分别构造独立 Plant、MPC、参考和 warm-start 状态；一个进程可先后处理多个 task，但每条 episode 的状态不共享。

每个决策边界读取 19 维 observation，采样 z，记录 old_logp 和 value，将连续 beta_target 交给 BetaEpisode.step_target。step_target 按 0.02/周期的 slew 限制执行目标，并推进原来的 Plant/MPC 闭环 80 个周期。episode 每段记录 observation、z、old_logp、value、reward、终止标志和 beta_target。QP 失败或约束越界时，该 episode 提前终止并按失败段规则扣分。

19 维观测依次由 16 秒参考预览的三项频谱能量、三项高频比例、总高频比例及低能量标志、三项当前跟踪误差、三项实际执行器状态，以及当前 beta、上次目标、归一化时间、上一段代价和上一段命令变化组成。参考预览来自任务的已知解析参考，当前误差来自 Plant 实际输出。策略未获得 task_id 或未来 Plant 响应。

### Reward

每个 0.02 秒 Plant 子步的跟踪代价为：

```text
e_t = ((E_ref-E)/8)^2 + ((D_ref-D)/5)^2 + ((VN_ref-VN)/2)^2
J_seg = mean(e_t over this 8-second segment)
reward_j = -min(J_seg / max(J_F75, 1e-9), 20)
           - 0.02 * (beta_target_j - beta_target_(j-1))^2
```

失败时额外减 20 * (9-j)，随后终止。J_F75 是相同训练 task 的完整基线缓存，1e-9 是冻结的奖励尺度下限。最终外部性能仍按完整轨迹的 32–64 秒时间窗计算原始 J_track，不按归一化 reward 排序。

### GAE 和 PPO

每条 episode 在 CPU 上单独反向算 GAE，gamma=0.98、lambda=0.95：

```text
delta_j = reward_j + gamma * V(next_state) * (1-done_j) - V(state_j)
A_j     = delta_j + gamma * lambda * (1-done_j) * A_(j+1)
return  = A_j + V(state_j)
```

正常完整 episode 的最后一步 done=true，故末步 next_value 为零。多个 task 的 transition 先分别算 GAE，再拼成 batch，并对整批 advantage 标准化。16 task 至多 128 条 transition。Batch tensor 放到 `--device` 指定设备；`auto` 在可用时选择 CUDA，否则 CPU。每次更新 4 epochs，minibatch 64，Adam 学习率 3e-4，PPO clip=0.2，value loss 权重 0.5，entropy 权重 0.01，梯度范数裁剪 1。

每个 minibatch 用当前策略重新计算保存的 latent z 的 logp，并以 `r=exp(new_logp-old_logp)` 形成 PPO ratio。实际最小化的损失为 `-mean(min(r*A, clip(r,0.8,1.2)*A)) + 0.5*mean((V-return)^2) - 0.01*mean(entropy)`。策略更新后才开始下一批 rollout，因此同一批数据始终由更新前同一个策略生成。

## CPU 与 GPU 分工

当前 GS-12 Plant/MPC 是逐 episode 的 CPU/NumPy 闭环。worker 进程并行的是不同 task 的完整 rollout；每个 task 内部的时间步仍按物理顺序串行。脚本将 OpenBLAS、OMP、MKL 线程设为 1，减少 worker 数乘数值库线程造成的 CPU 超额竞争。

策略网络本身只有共享的 19-64-64 主干和各自的 64-1 actor/critic 头，每次默认更新 batch 至多 128 条。CUDA 支持这部分批量张量更新，但网络计算量很小，传输及启动开销可能超过收益，环境中的 MPC/Plant 仍占主要时间。因此“CUDA 可用”不等于端到端训练已由 GPU 加速，worker 数和端到端速度需要在目标机器上实测。当前采样 worker 的策略推理在 CPU，避免每个 19 维观测都跨进程往返 GPU 并同步标量动作。

tqdm 每完成一条 episode 更新进度。每个 PPO 批次保存 checkpoint_NNNN.pt、history.json 和 episodes.jsonl；结束时保存 final_actor.pt 与 run.json。run.json 记录设备、任务和奖励缓存 SHA、seed、训练参数及动作定义。

建议从仓库根目录执行：

```bash
python3 -m experiments.gs12_rl_beta_pilot.train_ppo_continuous --source experiments/gs12_rl_beta_pilot/output --out experiments/gs12_rl_beta_pilot/output_continuous --episodes 64 --update-episodes 16 --workers 4 --device auto
```

## 为什么 ADMM 批量环境不能直接替换

ADMM 是求解 QP 的一类方法；OSQP 也采用 ADMM 类迭代，但它与 `simulator/batch_mpc.py` 中的 Torch 批量 box ADMM 不是同一个求解器接口。批量环境把多个 lane 的状态、动力学、参考、QP 和投影迭代沿 batch 维同时计算。即使用同一种优化算法，也不保证得到同一个闭环转移。

| 项目 | 当前 GS-12 连续 beta 训练 | 参考批量环境 |
| --- | --- | --- |
| 代码 | `runner_core.py` + `aav_mpc/condensed_mpc.py` | `high_speed_experiments/train_reference_policy.py` + `simulator/batch_mpc.py` |
| 预测状态和求解 | 22 维 actuator-aware 预测模型；CPU NumPy/SciPy 构造 condensed QP，每 lane 调用 OSQP | 10 维本地线性模型；Torch 批量构造 QP，box ADMM 迭代 |
| 预测时域 | 35 步，每步 0.1 秒 | `make_env()` 配置 25 步；实际参考训练入口设 `device="cpu"` |
| 策略动作入口 | 连续 beta 目标经限速后进入预测器参数 `alpha=[beta, 0]`，`theta_hat=[0.85, 2]` | actor 调参考速度或 residual，经 `ReferencePolicyEnv` 的参考生成路径进入控制 |
| 实际闭环 | 每次 MPC 命令后推进 5 个 0.02 秒 `plant_harness.Plant` 子步 | `ReferencePolicyEnv` 自己的 batch 环境状态、转移和 reset |

`CondensedActuatorMPC.solve()` 用 `base_theta + beta * (theta_hat - base_theta)` 设置预测参数；在 GS-12 默认基值下对应 `model_gain_t=1-0.15*beta`、`model_tau_t=1+beta`。这改变的是 MPC 内部预测模型，不是参考速度。参考脚本虽然展示了批量 ADMM 的实现方式，其 `make_env()` 实际选用 CPU；其组件具备 Torch/CUDA 路径也不能说明 GS-12 的现有闭环已能直接搬到 GPU。

当前 GS-12 真实闭环在 experiments/gs12_weight_route/runner_core.py：

1. 每个 ClosedLoopSession 构造一个 plant_harness.Plant、CPU 单 lane BatchedUUVModel 和 CondensedActuatorMPC，固定 horizon=35、dt=0.1；Plant、MPC plan 和 warm start 属于该 task。
2. 每次 solve 由当前 22 维 controller state、参考索引和后续 horizon 参考构造预测与 QP。动态 predictor 的连续 beta 通过 `CondensedActuatorMPC.solve(..., alpha=[beta, 0], theta_hat=[.85, 2.])` 注入预测参数路径。
3. MPC command 最多保持 5 个 0.02 秒子步。每个子步调用 harness Plant，再按冻结诊断检查侧向位置、深度、速度、俯仰和非有限状态；若提前越界则少于 5 步。
4. BetaEpisode 额外定义 8 秒决策边界、beta slew、19 维观测、段 reward 和失败终止。这些都是训练 MDP 的组成部分。

参考的 `high_speed_experiments/train_reference_policy.py` 使用 `ReferencePolicyEnv`、`PathBank` 和 `simulator.batch_mpc.BatchedLinearMPC`。它将多个 lane 打包，批量构建本地线性模型与 QP，再进行 ADMM 迭代。该训练脚本的 actor 动作调节参考速度或残差，动作进入 `ReferencePolicyEnv` 的参考生成路径；它不是 GS-12 的 alpha/beta predictor 注入接口。

所以，直接把 GS-12 task 送入 PathBank 或把 beta 填入参考速度动作，会改变动作物理含义。直接拿现有批量 MPC 取代 OSQP 版本，则会改变模型、horizon、cost、约束、warm-start 和 fallback 语义。两个系统即使产生相似轨迹，也不能据此把其 reward baseline、旧轨迹和 dev 结果混作同一实验。还需明确 batch lane 的失败、reset mask 和 MPC 状态隔离，否则一个 lane 的早停可能污染其它 lane。

当前 runner 虽然内部有 Torch 张量接口，但明确创建 CPU 模型，batch size=1；`CondensedActuatorMPC.solve()` 内部也逐 lane 调用 CPU OSQP，并在 Python 中逐步推进真实 Plant。把 actor 放到 CUDA 不会自动向量化这条 MPC/Plant 路径。当前训练脚本因此只并行独立 episode，CUDA 只负责 PPO 更新。

## 后续复用 ADMM 的验证门

批量 ADMM 可以作为下一步性能工程，但需要实现 GS-12 自己的 batch transition：

1. 每个 lane 都保留相同 GS-12 Plant、状态、参考索引、beta predictor 参数、MPC plan/warm start、QP 状态与失败状态。
2. 批量实现需要匹配原模型、horizon=35、0.1 秒 solve、5 个 0.02 秒 Plant 子步、tracking/input/rate cost、actuator bounds 和 solver failure 语义。
3. 在同一冻结任务和初态逐周期配对，核对参考、command、Plant state、actual actuator、beta/model parameters、QP 标志与终止时刻。先固定 beta，再测 beta 切换。
4. 检查 batch reset mask、失败 lane 隔离及 CUDA 精度；通过预注册数值容差后再用于训练吞吐优化。

若 batch 化需要改变动力学、MPC 或约束，就应冻结新 protocol 并重新训练、评估，不能混合旧 pilot 数据。若目标只是先提升训练效率，则保持现有真实闭环、多进程并行 episode，并把同批 PPO tensor 更新放到 GPU，是改动较小且可审计的路径。
