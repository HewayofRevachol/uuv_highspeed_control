# GS-12：离散 β 策略的 RL 小批量试跑执行指令

> **状态：关键闭环接线已实现并通过 smoke；PPO 训练、任务冻结、正式对照仍待实现与运行。** 本文件按当前仓库代码和 `experiments/gs12_predictor_route/output/REPORT.md` 的实测结果制定。它授权一次真实的 64 episode PPO pilot；性能不佳也必须交付训练记录和配对开发结果。`experiments/gs12_rl_beta_pilot/` 已有 `episode_beta.py`、`features.py`、`reward.py`、`controllers.py`、`categorical_ppo.py`、`verify_loop.py`；其余命令与 RL 指标仍是**待实施目标**，不是已取得的 RL 收益。
>
> **边界：**真实植物仍是 C4；W1=`(a_D,a_Δδ)=(1.4,0.7)`、MPC 约束、`dt=.1 s,N=35` 固定。RL 只选 MPC 内部预测器路径的 β，`gain=1-.15β,tau=1+β`；不改真实植物、不读取未来真实状态、不训练 Q/S 权重或残差模型。预先冻结新的 train/dev/rl_test_locked_v1 任务；旧 `ps_lock_*`、`pd_lock_*`、更早的 `locked_*` 只作背景，不能作为本轮独立测试。

## 0. 先知道哪些结论已成立

最新实测报告是 [`gs12_predictor_route/output/REPORT.md`](REPORT.md)，不是旧固定权重的 `REPORT.md` 或不存在的 `REPORT(2).html`：

- 静态 W1 五点扫描 20/20 完整。新开发任务的任务等权平均 `J_track`：β=0 为 `4.35895e-7`，β=.25 为 `1.46942e-7`，β=.5 为 `2.15187e-8`，β=.75 为 **`1.69775e-8`**，β=1 为 `1.08340e-7`。低频深度偏好 β=0，高频深度/东向偏好 .75，混合任务偏好 .5；这是任务级静态取舍，不证明动态切换有效。
- 冻结静态候选 β=.5 的新锁定测试 16/16 完整。它相对 β=0 的平均 J 降 95.81%，但比预先指定的强固定 β=.75 **高 19.41%**，低频深度相对 β=0 也退化，严格门槛未过。**F75（全程 β=.75）是本轮主要固定基线**；F00/F50 分别解释低频和混合任务。
- 16 s 计划参考频谱特征在五条开发任务上通过可识别性检查；动态参数注入的 β=0/1 端点与静态路径逐步一致。R0/R1 规则各自质量有效，但相对 F75 的开发任务平均 J 分别**恶化 937.67%/916.04%**；两规则都未晋级，原 `test_dynamic_locked` 未打开。高频纯任务的规则 β 均值约 .9984，偏离已测最佳 .75。R0/R1 的失败**不是** RL 有效或无效的证据，本 pilot 不复制它们将高频映射到 β=1 的动作逻辑。

因此本轮直接做 RL 小批量训练，同时以 F75 和一个明确的非学习规则作强对照。**64 episode 是接口与学习趋势试验，不是收敛或泛化证明。**

## 1. 可复用的实际代码与最小实现范围

| 仓库内现有对象 | 本轮使用方式 |
|---|---|
| `experiments/gs12_weight_route/runner_core.py:run_one`、`ClosedLoopSession` | **本 GS-12 协议的闭环环境基准**：使用下述 C4 `Plant`、同一 MPC、解析参考和日志；原 `run_one()` 签名及 `beta_policy(t,ref_grid,task)` 钩子保留，已提取 `advance_control_cycle(beta)` 供高层每 8 s 暂停与继续 |
| `aav_mpc/condensed_mpc.py:CondensedActuatorMPC.solve` | 使用已存在的 `alpha=[β,0],theta_hat=[.85,2],d_hat=None`；base dynamics 固定 P0 `(1,1)`，不每步重建 MPC/CasADi/OSQP、不改变 warm start |
| `experiments/gs12_joint_route/joint_reference.py:joint_task_reference` | 生成 8 维运动学一致参考，混合任务已有前 16 s 启动包络；新任务原样复用公式，仅换冻结参数 |
| `experiments/gs12_analaysis/frequency_extension_src/plant_harness.py` | 复用本协议实际使用的 `Plant`、`PlantConfig`、`diagnostic_values` 与配平/子步实现；保留 `analaysis` 历史拼写 |
| `aav_mpc/controller.py:CondensedActuatorEnv`、`simulator/batch_path_env.py:BatchedPathResidualEnv` | 仓库确有这些 Env，但它们读取 CSV 路径、按路径进度生成参考；reset/植物随机化/终止和奖励也不同于当前固定时间 C4 实验。直接换用会改变被比较的实验，不把它们当本轮数值环境 |
| `experiments/gs12_predictor_route/spectral_policy.py:features` | 复用或核对 16 s 参考 FFT 的归一化、去趋势/Hann/rFFT 口径；它只有总频谱特征，不直接提供本轮完整观测 |
| `training/ppo.py`、`experiments/gs12_gate/train.py` | 参考 on-policy 批次、GAE、checkpoint/manifest 方式；**旧 `GateActorCritic` 是连续门控动作，不可直接当离散 β 策略** |

本轮直接使用仓库的 GS-12 闭环与 PyTorch PPO 实现，**不依赖 Gymnasium 或 Stable-Baselines3**。可复用 `training.ppo.advantages()`，但其通用 `ppo_update()` 的熵项写成 `distribution(obs).entropy().sum(-1).mean()`：对于 `Categorical` 输出 `[batch]` 时会先对**整个 batch 求和**，权重与连续动作版不同。已实现 `gs12_rl_beta_pilot/categorical_ppo.py:update()`，分类 `log_prob`、entropy 均按每样本 `[batch]` 取 `.mean()`；训练入口须调用它，不能静默沿用旧连续动作损失。

在已有 `experiments/gs12_rl_beta_pilot/` 的基础上补 `tasks.py,train_ppo.py,evaluate.py,analyze.py`。现有 `runner_core.run_one()` 的一次控制周期已提取为同文件的 `ClosedLoopSession.advance_control_cycle(beta)`，旧 `run_one()` 已调用该会话跑满 64 s；`episode_beta.py` 只负责每 80 周期暂停、选高层动作、算奖励并恢复**同一个会话**。这样静态 runner 和 RL 共用仓库对应的 C4 Plant/MPC/参考/积分实现，不建第二套仿真环境，也不复制 QP 或植物。默认 `run_one()` 的旧签名和输出须保持一致，用第 3 节固定动作逐样本对齐验证这次必要重构。

## 2. 新协议和任务：在首次训练前冻结

新输出根为 `experiments/gs12_rl_beta_pilot/output/`，绝不写回 `gs12_predictor_route/output/`。`prepare` 必须一次冻结：`protocol.json`、`tasks_train.jsonl`、`tasks_dev.jsonl`、`tasks_rl_test_locked_v1.jsonl`、动作/观测/奖励 schema、源码 SHA-256、Git HEAD 和脏工作树状态、依赖版本、质量/晋级门。已有文件拒绝重写；训练/开发/测试任务的完整 JSON 哈希不能重复。

**训练清单：**用 `numpy.random.default_rng(20261111)` 在 `prepare` 时一次生成并保存 **64 条** 64 s 训练任务，四类各 16 条，批次顺序亦写入清单；第一轮 64 episode 各用一次。四类为低频深度 `k∈{1,3,4,6}`、高频深度 `k∈{25,27,29,31}`、高频东向同一高频集合、东向+深度混合。各类 k 轮转四次；相位从 `[-π,π)` 抽取并记录到 6 位小数，混合两次切换中心从预先固定的 `{[19,39],[21,43],[25,45],[27,47]}` 轮转，仍用旧混合参考的低 `k=8`/高 `k=32` 成分。主幅值 `.006`、速度 15 m/s、前 16 s 平滑启动、评分窗 `[32,64]`。本 pilot 不加 `.003/.012` 幅值，以免在小批量同时引入另一个训练维度。后续如扩到 128/256 episode，**先另版冻结增补训练任务和预算**；不改本次 dev/locked。

**固定开发和新锁定清单：**下面每条都为 64 s、15 m/s、幅值 `.006`、`window_s=[32,64]`，相位单位 rad；单频 `ω_k=2πk/64`。它们与本 pilot 训练任务的频点集合分开；旧实验用过某些频点，但本轮任务的完整参数和相位是新的，独立性只针对本轮 RL 选参过程，不能声称频点从未在整个项目出现过。

| split | task_id | kind/axis | k 或切换中心 s | phase_rad |
|---|---|---|---|---:|
| dev | `rl_dev_low_2` | single/depth | k=2 | .42 |
| dev | `rl_dev_low_8` | single/depth | k=8 | -1.05 |
| dev | `rl_dev_dhigh_26` | single/depth | k=26 | .74 |
| dev | `rl_dev_dhigh_30` | single/depth | k=30 | -1.27 |
| dev | `rl_dev_ehigh_26` | single/east | k=26 | 1.21 |
| dev | `rl_dev_ehigh_30` | single/east | k=30 | -.33 |
| dev | `rl_dev_mixed_a` | mixed/east_depth | [20,41] | .91 |
| dev | `rl_dev_mixed_b` | mixed/east_depth | [25,44] | -1.73 |
| `rl_test_locked_v1` | `rl_lock_low_5` | single/depth | k=5 | -2.04 |
| `rl_test_locked_v1` | `rl_lock_low_7` | single/depth | k=7 | 1.36 |
| `rl_test_locked_v1` | `rl_lock_dhigh_28` | single/depth | k=28 | .37 |
| `rl_test_locked_v1` | `rl_lock_dhigh_32` | single/depth | k=32 | -1.94 |
| `rl_test_locked_v1` | `rl_lock_ehigh_28` | single/east | k=28 | 2.11 |
| `rl_test_locked_v1` | `rl_lock_ehigh_32` | single/east | k=32 | -.68 |
| `rl_test_locked_v1` | `rl_lock_mixed_a` | mixed/east_depth | [18,43] | 1.89 |
| `rl_test_locked_v1` | `rl_lock_mixed_b` | mixed/east_depth | [24,47] | -2.62 |

`joint_task_reference` 为规则/策略生成到 **80 s** 的同一解析参考计划，保证末尾仍可取未来 16 s；实际闭环只运行到 64 s。**部署可得信息假设固定为 16 s 外生参考预览**，与前轮 `dynamic_protocol.json` 一致。MPC 本身只取未来 3.5 s；不声称其接口自带 16 s。若目标部署场景不能提供 16 s 预览，先另立短预览协议，不能在本协议试验中偷看不可得未来。所有方法同任务、同初态/配平、同参考数组和采样积分。通过 `joint_reference.derivative_error` 检查位置/速度，最大差 `<1e-3`；检查三组任务 JSON 及参考哈希唯一。

旧 `ps_lock_*` 已打开，不使用；旧 `pd_lock_*` 虽未运行但其定义已公开，不挪作本轮测试。`rl_test_locked_v1` 只在第 7 节条件满足、模型/checkpoint/基线/指标全部冻结后运行**一次**。

## 3. 仓库 GS-12 闭环会话、动作与数值验收

环境使用 `runner_core.py` 中从 `run_one()` 提取的 `ClosedLoopSession`，不引入 Gymnasium 适配层。会话构造时接收一条任务及 `duration_s=64., reference_fn=joint_task_reference, preview_s=16., controller="C4", a_depth=1.4, a_rudder_rate=.7, predictor_params=(1.,1.)`；初始化逻辑沿用 `run_one()`，包括 `PlantConfig(elevator=.9,rudder=.9,order="second_all")`、`Plant`、`BatchedUUVModel`、`CondensedActuatorMPC`、初始 `previous/plan` 和参考网格。每条 episode 新建一个会话，8 个高层决策共用该会话，不从上条轨迹继承 Plant、MPC warm start 或队列。

在 `runner_core.py` 内把原循环中的一次 .1 s 控制周期封装为 `advance_control_cycle(beta)`：按旧索引取 35 点参考，设置 W1 权重，调用一次 `mpc.solve(...,alpha=[beta,0.],theta_hat=[.85,2.],d_hat=None)`，记录相同的 QP/命令/耗时字段，再执行 5 次 `Plant.step(command)`、采集 5 个 .02 s 子步并更新早停诊断。`run_one()` 仍调用这个周期接口直到结束，保持原签名、输出字段和状态判定；完成与原实现的对齐验收后才用于 RL。会话的终止信息只报告 64 s 到达、QP 失败、诊断越界或非有限值，不加入另一套植物动力学或奖励。

`episode_beta.py` 用本地动作索引 `0,1,2,3 → β_target=0,.25,.5,.75`；β=1 仅保留为报告背景/诊断，不进入本轮 RL 动作。每次高层决策调用同一会话的 `advance_control_cycle(beta_actual)` 最多 80 次；完整 episode 应有 **640 次 MPC 求解、3200 个植物子步、8 个 transition**。任务/随机种子由冻结清单和训练入口传入；单个 episode 的控制周期顺序执行，不在内部并行。

RL 和 Rule-cap75 从 `β_actual=.75,β_prev_target=.75` 开始；高层目标切换时，每个 .1 s 控制周期将 β 朝目标最多移动 `.02`，到达后保持，`β∈[0,.75]`。这是本 pilot 预冻结的切换速度，不在看过 dev 结果后修改。F75/F50/F00 **从第一个 MPC 求解起分别恒为 .75/.5/0**，不从 .75 渐变，才能代表真实固定控制器。β 进入 `mpc.solve(alpha=[β_actual,0],theta_hat=[.85,2],d_hat=None)`；P0 base dynamics、W1、植物状态、执行器与命令队列持续保持。QP 失败使用现有 `fallback_mode="hold"` 的诊断记录，但本 RL episode **立即终止并施加第 5 节惩罚**，不把持有旧命令后的短轨迹当完整成绩。

质量阈值沿用现有 runner：`diagnostic_values` 的侧向位置 8 m、深度 5 m、切向速度 2 m/s、pitch 40°；命令和实际执行器达到归一化盒界 `1e-6` 内记饱和。每条任务的命令尖峰标志预注册为：候选最大归一化单步命令变化**同时** `>0.05` 且 `>1.25×同任务 F75`。F75 自己若早停、QP 失败或越界，该任务不能用于正式晋级，须先报告基线质量问题。任何方法的失败轨迹仍保存 JSON/NPZ 和首次失败时间，不能静默排除。

`episode_beta.py` 每段返回 `(obs,reward,done,info)`；到达 64 s、QP 失败、诊断越界或非有限值时 `done=True`，`info` 标明原因。训练按完整任务或真实早停收集，不在任务中途人为截断 PPO rollout；传入 GAE 时本轮 `done=terminated`，末端 `next_value=0`。每段 `info` 保存段起止时间、实际 β 路径、段代价、三通道 RMSE/峰值、QP 失败/残差、约束/饱和、最大命令/实际舵面变化、求解 p50/p95/p99；整条 NPZ 保存 0.1 s 与 0.02 s 原始数组及 8 个 action/reward/logprob/value。缺失或早停任务保留状态行。

**先过数值门再训练：**

1. 先运行第 8.1 节现成的 `verify_loop`，用旧公开轨迹检查重构后的 `run_one()` 和 `BetaEpisode`；然后在一条**新的 dev** 任务上，用 `BetaEpisode` 的固定动作 F75、F50、F00 各跑 8 段，分别与当前 `runner_core.run_one(task,"C4",1.4,.7,64.,joint_task_reference,(1.,1.),beta_policy=lambda tm,g,t,b=beta:b,preview_s=16.)` 逐周期比较命令、艇体状态、实际执行器及 QP 有效状态。最大绝对数值差 `≤2e-7`、QP 标志逐步一致；参考数组哈希一致。F75 的完整 `J_track` 应与 runner 外部评分一致。验收轨迹只作接口对齐，不占新 dev 正式配对名额。失败时排查参考索引、warm start、队列、植物 5 子步或 β 注入，**停止训练**。
2. 在同一 dev 混合任务依次试 `.75→.5→.75`、`.75→0→.75`、`.5→.75`，每目标至少保持一个 8 s 段；检查每步 `|Δβ_actual|≤.02+1e-12`、内部模型参数轨迹、QP 目标变化、命令/实际舵面连续性及求解耗时。与同任务 F75 配对，但只作为接口诊断，不当锁定性能结果。
3. 用 2–4 条 dev 任务跑随机动作，确认本地动作索引、观测 19 维/有限值、奖励/终止/日志一致，连续新建两条会话检查初态、`previous/plan` 与队列未串扰。首次使用 1 worker；一条 64 s 轨迹测时与内存后再选 1/2/4 worker，不照搬旧吞吐数值。

上述三项失败可修最小接线错误并重新跑 smoke；若修了影响控制/参考的核心代码，要给本轮协议升版本并重新验收，不能把不同版本的轨迹混算。

## 4. 观测与强非学习规则：只用部署可得信息

首版观测固定为 **19 维 float32**，先在 `obs_schema.json` 写明每个坐标和尺度，再训练。未来参考 16 s 以 .1 s 采样 160 点，对 `(E/8,(D-20)/5,(VN-15)/2)` 分通道减去均值/线性趋势、Hann 加窗、rFFT；频率 `ω=2π·rfftfreq(160,.1)`，高频阈值 `2.2 rad/s`，仅用非直流功率。每通道能量 `P_c=Σ_{ω>0}|FFT_c|²/160²`，每通道高频比 `ρ_c`；总高频比按三个通道功率合并。能量 `<1e-8` 时高频比置 0、另给低能量标志。

观测顺序固定为：3 个 `clip((log10(max(P_c,1e-12))+12)/12,0,1)`；3 个 `ρ_c`；1 个全通道 `ρ_high`；1 个全通道低能量标志；3 个当前 `(E_r-E)/8,(D_r-D)/5,(VN_r-VN)/2` 裁剪到 `[-2,2]`；3 个实际执行器 `/CONTROL_SCALE`；1 个 β_actual；1 个上一 β_target；1 个 `t/64`；1 个上一段 `clip(log1p(J_seg/1e-9),0,20)`（reset 时 0）；1 个上一段最大归一化命令变化裁剪到 `[0,1]`（reset 时 0）。总数 `3+3+1+1+3+3+1+1+1+1+1=19`。`Plant.output()` 与 `joint_task_reference(t)` 只取**当前**测量/参考；FFT 只取已知计划。不得把 `task_id`、任务类别、未来植物响应、F75 缓存 J 或真实模型参数放入观测。所有方法使用相同 16 s 预览权限。

预先固定简单非学习对照 **Rule-cap75**：每个 8 s 决策边界从同一 16 s 参考计算全通道 `ρ_high`；若总能量 `<1e-8`，沿用上一目标（初始 .75）；否则 `ρ<.2→β_target=0`，`.2≤ρ<.8→.5`，`ρ≥.8→.75`。随后使用与 RL 相同的 `.02/控制周期` 平滑。它与旧 R0/R1 不同：高频上限为已测较好的 .75，不会选 β=1；但它仍只是**待测基线**，不能预先称有效。阈值在训练前写入协议，允许在 dev 上分析其失败，不能根据 locked 改规则。F00/F50/F75 与 Rule-cap75、RL 是锁定测试的完整方法集合；旧 R0/R1 的历史 J 不能直接混入新任务配对。

## 5. 奖励、PPO 更新与实际训练预算

最终评估仍用原外部指标：在完整轨迹 `32≤t≤64 s` 的 0.02 s 子步上算

\[
J_{track}=\operatorname{mean}[((E_r-E)/8)^2+((D_r-D)/5)^2+((VN_r-VN)/2)^2],
\]

并先按任务算 J，再对任务等权平均；各通道米制 RMSE、绝对 ΔJ、相对比例、QP/越界/饱和和命令尖峰另列。**训练 reward 可以归一化，最终 J 不改。**

为避免高频任务数值压倒低频，`prepare` 后在**64 条训练任务上各跑一次 F75 完整轨迹**，生成只供训练奖励使用的 `reward_baseline_cache.jsonl`，含 task SHA、F75 run ID/trace SHA、完整 `J_F75`、QP/完整状态。缓存不进入观测；缓存任务失败时先查原因，不用替代值悄悄继续。训练 reward 对第 j 个 8 s 段定义：

```text
J_seg = 该段子步的原外部归一化误差平方和的时间平均
J_ref = max(该训练任务缓存的完整 J_F75, 1e-9)
reward_j = -min(J_seg/J_ref, 20) - 0.02*(beta_target_j-beta_target_(j-1))**2
```

初始上一目标为 .75。若在第 j 段发生 QP 失败、诊断越界或非有限值，追加 `-20*(9-j)`（j 从 1 到 8），立即 `terminated=True`；这样丢掉后续段不能让短失败轨迹因少算负 reward 而获益。`1e-9` 是**预先冻结的训练尺度下限**，不是工程容差；完整报告低频任务的真实 J/RMSE，不能只给归一化 reward。若 reward/数值出现 NaN，停止训练并记录触发任务；调整尺度需新训练版本，不改已打开的测试集。

PPO 策略为 `MLP(19→64→64→4 logits)` 加独立标量 value head，`torch.distributions.Categorical`。用 `training.ppo.advantages(rewards,values,next_values,done,terminated,gamma=.98,lam=.95)`；本轮所有训练 episode 完整结束或因失效终止，`done=terminated`，末端 `next_value=0`。分类 PPO 目标：`ratio=exp(new_logp-old_logp)`，clip `.2`，value loss 权重 `.5`，**分类 entropy 对 batch 取均值**、系数 `.01`，梯度范数裁剪 1。记录 policy/value loss、entropy、近似 KL、clip fraction、梯度范数、每类任务回报和动作占比；非有限梯度/参数即停。

首批预算固定 **64 个训练 episode = 512 个高层 transition = 40960 次 MPC 求解**（若无早停），另有 **64 条 F75 奖励缓存轨迹**及 smoke/dev 对照成本；不能把“64 episode”误算为仅 64 个 MPC 步。四类任务各 16 条；按任务清单分成 4 个批次，每批 16 episode、每类 4 条、共 128 transition，**同一批始终使用同一 policy 版本**，收集完再作一次 PPO 更新，共 4 次更新。每次更新 4 epochs、minibatch 64、Adam `lr=3e-4`；训练 seed `20261111`。可用最多 4 个 worker 并行**完整 episode**，每 worker 独立 Plant/MPC；MPC 时间步本身必须串行。若内存不足用 1/2 worker，保持同样任务顺序、batch 和更新数。保存第 2 次与第 4 次更新的 checkpoint、优化器与 RNG 状态、协议/源码/任务哈希；不要在 batch 中途更新策略。

**必须真实执行 PPO optimizer 更新**，而非只训练分类模仿器、运行固定规则或表查找。64 episode 不足以断言收敛；即使 dev 未改善，仍完成小批量 pilot 报告。若有明确 dev 趋势且质量有效，下一版本可扩到 128/256 episode、2–3 个训练 seed，新增训练任务与预算先冻结，不自动使用新锁定集调参。

## 6. 开发集对照、checkpoint 选择与新锁定门

在第 2 节 **8 条 dev** 任务上，分别运行 F00、F50、F75、Rule-cap75 和 PPO（确定性 argmax），一共 **40 条完整配对轨迹**；PPO 第 2/4 次更新 checkpoint 可在 dev 上比较，但固定对照只需各跑一次并复用相同任务的运行文件。选择 checkpoint 的顺序预先写死：质量有效优先；再选任务等权平均 J 较低者；同分选较早 checkpoint。每个方法均给 8 行任务指标，显示低频、高频、混合分组及总体值；不能仅报告总体百分比或训练 reward。

pilot 的**交付门**是：GS-12 闭环逐周期对齐、切换 smoke、随机动作 smoke、64 个真实 PPO 训练 episode、dev 的完整配对和失败分析均完成。**性能不通过也要交付**。只有选定 checkpoint 在 dev 上相对 F75 的任务等权平均 J 至少下降 **5%**、8/8 任务均不劣、且无新增 QP/边界/饱和/预注册命令尖峰时，才解锁 `rl_test_locked_v1`；Rule-cap75 也要作为强对照，但锁定门主要以 F75 为预注册基线。若只有部分任务收益，记录取舍，**不为争取正面结果打开锁定集**。5% 是工程晋级门槛，不是统计显著性。

锁定测试前写 `selection.json`：checkpoint 路径与 SHA、dev 选择表 SHA、Rule-cap75 代码/阈值、动作/观测/奖励 schema SHA、F75 基线定义、完整质量门、protocol SHA、训练 seed；测试入口必须拒绝缺失/未冻结状态。打开 8 条 `rl_test_locked_v1` 后，仍对 F00/F50/F75/Rule-cap75/RL 五方法一次性配对，共 **40 条轨迹**，以同一外部 J、通道 RMSE、QP/约束/尖峰与完整性口径报告。独立严格通过：RL 相对 F75 的任务等权平均 J 下降 ≥5%、8/8 任务不劣、无新增质量阻断；另列 RL 相对 Rule-cap75、F50 的数值，以判断 RL 是否有独立价值。若 RL 未胜 Rule-cap75，即使胜 F75，也不能称 RL 的学习机制已证明必要。

任务、checkpoint 或门槛在锁定结果打开后均不能重新调节并把同一批任务再次称为独立测试。若因技术损坏需重跑，保留原错误和修复记录，仅重跑**相同冻结策略/任务/指标**；策略改变则另建新锁定集。旧 `test_dynamic_locked` 继续保持未打开，不因新 RL pilot 而借用。

## 7. 失败分支：仍要完成已授权的 pilot

| 观察 | 直接处理 | 结果如何写 |
|---|---|---|
| 固定动作无法对齐 `run_one` | 停止训练，检查参考预览索引、初态/配平、warm start、队列、β 传参、5 子步；修复后重做 smoke | 技术接线失败，不是 RL 性能负结果 |
| `.75→.5/0` 切换引入 QP/越界/明显饱和 | 停止动态训练；可只做 episode 起点选一次 β 的 contextual-bandit smoke，但不得冒称已完成 8 段 RL 主试跑 | 报切换瞬态、首失败时刻、求解/舵面曲线；本轮 P1 未完成 |
| PPO loss/观测/reward 非有限或无 optimizer 更新 | 停止，定位任务/张量形状/归一化/分类 entropy；只在新训练版本修正并重跑 64 episode | 训练技术失败，保留日志；不输出虚构的 RL 收益 |
| PPO 几乎总选 .75 | 报每类任务动作占比、低频 J/RMSE 与 F75 差；检查任务均衡和奖励下限的作用，仍完成 64 episode 与 dev 对照 | 学到固定强基线附近，尚无上下文自适应证据 |
| 目标在 0/.75 间频繁跳，混合任务变差 | 对照同任务 F75、F50、Rule-cap75 和 `.75→.5→.75` smoke；画 β/命令/误差时序，检查切换代价 | 动态切换成本可能抵消静态选择收益；不直接加大网络 |
| RL dev 比 F75 好却不如 Rule-cap75 | 完成 pilot，明确简单规则仍是更强对照；只在开发集研究是否有额外复杂场景 | 有控制收益趋势，尚无 RL 独立价值 |
| RL dev 未达 5% 或某任务退化 | 不打开 `rl_test_locked_v1`，交付完整 pilot 和失败分析；可在新版本延长训练/改观测，但不回用旧锁定集 | 开发集未晋级，不等于所有 RL 方法无效 |
| 新锁定集未通过 | 保留全部逐任务与质量结果，停止在该集上选参；下轮另建锁定集 | 独立验证未支持此策略 |

P0「整条任务只选择一次 β」可用来诊断上下文选择是否有价值，但不能取代已授权的 P1 八段 PPO pilot；只有动态接线/质量确实阻断 P1 时才降级并如实报告。β=1 不在本轮动作空间，若日志出现 `β_actual>0.75+1e-12` 是映射错误而非策略发现。大幅机动/不同航速不是本 pilot 默认任务；出现需求时另立迁移协议，保留原硬约束，不能把受约束响应叫作无约束局部线性频响。

## 8. 文件、CLI 合同、审计和汇报

新目录**已实现** `README.md`、`episode_beta.py`、`features.py`、`reward.py`、`controllers.py`、`categorical_ppo.py`、`verify_loop.py`；**待补** `config_pilot.json`、`tasks.py`、`train_ppo.py`、`evaluate.py`、`analyze.py`。`output/` 的正式实验至少保存冻结协议与三个任务清单、`reward_baseline_cache.jsonl`、`models/*.pt`、`runs/<run_id>.json`、`traces/<run_id>.npz`、`train_log.csv`、`dev_episode_metrics.csv`、`manifest.json`、`integrity_audit.json`、`figures/`、`PILOT_REPORT.md`。每条 run JSON 记录 `run_id,protocol_sha256,task_sha256,source_sha256,split,task_id,method,seed,policy_sha,action/obs/reward_version,completion_status,first_violation_s,qp_failures,command_saturation,actual_saturation,J_track,RMSE_E/D/VN,peak_E/D/VN,solve_p95/p99_ms,trace_sha256`；训练轨迹另记每段 action/logprob/value/reward、β_target/actual。分析按预期 `task×method` 左连接，缺失、SHA 错、早停单列，不静默剔除或给不完整集合排榜。旧确定性时间步/相位不是统计独立样本，不报无依据的置信区间或实艇失效概率。

### 8.1 已实现脚本如何使用

在**仓库根目录**运行接线验收：

```bash
python3 -m experiments.gs12_rl_beta_pilot.verify_loop --out experiments/gs12_rl_beta_pilot/output
```

`verify_loop.py` 默认读取仓库已有的 `gs12_joint_route/output` 静态轨迹和 `gs12_predictor_route/output` F00/F50/F75 历史开发轨迹；只使用这些**已公开的开发数据**检查代码回归，不读取本轮新锁定集。它核对静态路径前 8 s 的命令、艇体状态与实际执行器，并核对三条固定 β 路径前 8 s 的上述数组、参考和 QP 标志，再核对连续两段 `.75→.5` 的 16 s 轨迹与原 `run_one()`，最大绝对差门槛 `2e-7`；还核对旧 F75 的外部 `J_track`。命令成功退出并写入 `experiments/gs12_rl_beta_pilot/output/smoke/verify_loop.json` 才算通过。此前实跑的固定路径差为 0，切换路径最大命令差为 `1.1724e-7`；这些是**接线验收**，不是 RL 性能结果。若历史轨迹目录位置不同，可用 `--archive <gs12_predictor_route/output目录>` 指定预测器轨迹目录；静态核对仍读取仓库内的 `gs12_joint_route/output`。

后续 `train_ppo.py`、`evaluate.py` 按以下**现有 Python 接口**接线，不能另写一套 Plant/MPC 循环：

| 已实现接口 | 调用约定 |
|---|---|
| `BetaEpisode(task, j_f75, initial_beta=.75)` | 每条任务构造一次；`task` 来自冻结清单，`j_f75` 从**同一任务完整 64 s 的 F75 轨迹**计算；训练任务按任务哈希缓存。接口验收中的 `1e-9` 只是占位，不得用于训练。F00/F50/F75 固定对照分别以 `initial_beta=0/.5/.75` 构造。 |
| `episode.observe()`、`episode.step(action_index)` | 决策前取 19 维 `float32` 观测；动作索引 `0/1/2/3` 对应 β `0/.25/.5/.75`。`step` 返回 `(next_obs, reward, done, info)`，最多推进 80 个控制周期；`done` 后 `next_obs` 为全零占位，不能再调用 `step`。8 段必须复用同一个 `episode`；`episode.result()` 给出与 `run_one()` 同格式的 `meta/arrays`。 |
| `fixed_action(beta)`、`rule_cap75_action(reference_window, previous_target)` | 固定对照每段传同一动作，且首段不从 .75 渐变；规则对照在每个决策边界取 `episode.session.ref_grid[5*k:5*k+800:5]` 的 160 个未来参考点，其中 `k=episode.session.k`，并传入上一段目标 β（首段 .75），返回动作索引。规则和 RL 的 β 平滑都由 `BetaEpisode.step()` 完成。 |
| `external_j_track(arrays)` | 只对覆盖完整 `[32,64]` s 评分窗的轨迹调用；输入 `episode.result()["arrays"]`，返回原外部 J。训练奖励用 `reward.py:segment_reward` 的段代价和任务 F75 缓存，最终比较仍用外部 J。 |
| `BetaActorCritic()`、`gae_advantages(...)`、`categorical_ppo.update(agent, optimizer, batch, epochs=4, minibatch=64)` | `batch=(observations, actions, old_logp, returns, advantages)`；形状分别为 `[N,19]`、四个 `[N]`，`actions` 为 `torch.long`，其余为有限张量。按第 5 节以 `gamma=.98,lam=.95` 算 GAE，完整/真实早停时 `done=terminated` 且末端 `next_value=0`；一个 16 episode 批次收完后调用一次 `update`，它内部执行 4 epochs 的真实 optimizer 更新并返回损失、熵、KL、裁剪比例和梯度范数。 |

Rule-cap75 的逐任务接线如下；`task` 与 `j_f75` 须先从冻结清单及同任务 F75 缓存读取：

```python
from experiments.gs12_rl_beta_pilot.episode_beta import BetaEpisode
from experiments.gs12_rl_beta_pilot.controllers import rule_cap75_action
from experiments.gs12_rl_beta_pilot.reward import exter
nal_j_track

episode = BetaEpisode(task, j_f75, initial_beta=.75)
while not episode.done:
    k = episode.session.k
    preview = episode.session.ref_grid[5*k:5*k+800:5]  # 160 点，0.1 s 间隔
    obs = episode.observe()
    action = rule_cap75_action(preview, episode.beta_prev_target)
    next_obs, reward, done, info = episode.step(action)
result = episode.result()
J_track = (external_j_track(result["arrays"])
           if result["meta"]["status"] in ("complete_valid", "complete_constrained")
           else None)  # 失败轨迹保留状态，不参与完整轨迹排名
```

为避免把旧公开开发任务误当新训练数据，上表说明的是**接线合同**；正式任务清单、F75 奖励缓存、训练循环和评估 CLI 尚需按第 2、5、6 节实现。单独使用 `BetaEpisode` 做固定动作 smoke 时，可先用旧公开开发任务和其已存的完整 F75 轨迹计算 `j_f75`，但这些轨迹不计入本轮 64 episode 训练或新 dev 结果。

以下 CLI 中 **`verify_loop` 已实现并通过运行**；其余是待实现的合同，脚本完成后以 `--help` 核对并把真实命令写进 README：

```bash
python3 -m experiments.gs12_rl_beta_pilot.verify_loop --out experiments/gs12_rl_beta_pilot/output
python3 -m experiments.gs12_rl_beta_pilot.tasks prepare --out experiments/gs12_rl_beta_pilot/output
python3 -m experiments.gs12_rl_beta_pilot.evaluate --split train_baseline --method F75 --out experiments/gs12_rl_beta_pilot/output
python3 -m experiments.gs12_rl_beta_pilot.train_ppo --episodes 64 --seed 20261111 --out experiments/gs12_rl_beta_pilot/output
python3 -m experiments.gs12_rl_beta_pilot.evaluate --split dev --with-baselines --out experiments/gs12_rl_beta_pilot/output
python3 -m experiments.gs12_rl_beta_pilot.analyze --split dev --out experiments/gs12_rl_beta_pilot/output
# 只有 dev 晋级、模型和基线冻结后才允许：
python3 -m experiments.gs12_rl_beta_pilot.evaluate --split rl_test_locked_v1 --with-baselines --out experiments/gs12_rl_beta_pilot/output
python3 -m experiments.gs12_rl_beta_pilot.analyze --split rl_test_locked_v1 --out experiments/gs12_rl_beta_pilot/output
```

执行顺序以第 3 节数值 smoke 先于 `prepare` 后正式采集为准；已有 `verify_loop` 先对齐仓库保存的旧公开开发轨迹，再用本轮新 dev 任务补固定动作、切换和随机动作 smoke，不得读取锁定结果。`prepare` 必须在首次训练和奖励缓存采集前完成。主进程预先检查 run ID 唯一，worker 只写自己的 JSON/NPZ，父进程单独写 manifest；不为理论上的极罕见并发/安全边界增设数据库、分布式锁或通用事务系统。

**复杂度准则：**若有人提议为推测性或发生概率极低的竞争/安全边界添加复杂机制，先向用户解释具体失败模式、发生可能性、实现及维护成本，取得同意后再扩展；不得悄悄扩大范围。已观察到的真实接线、失败记录或共享 manifest 写冲突，按最小修复处理。不要为了未来可能的 RL 扩展重写 MPC/植物。

最终 `PILOT_REPORT.md` 要明确分列「已完成历史实验」「本轮真实 PPO 更新与 dev 结果」「新锁定是否打开」「未运行/失败」，报告实际 episode/transition/optimizer update 数、训练耗时、完整率、逐任务 J/RMSE、动作占比和质量事件。**首个可交付里程碑即便性能未提升，也必须包含真正的 64 episode PPO 训练与配对 dev 分析；若技术或质量门阻断，准确说明停在何处。**
