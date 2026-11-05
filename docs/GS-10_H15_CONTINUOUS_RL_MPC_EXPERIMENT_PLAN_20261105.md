## 计划目标

本计划用于快速回答一个问题：在固定 15 m/s、同一条 MPC 约束和同一计算预算下，连续双门控策略是否比固定保守模型和确定性门控更好地处理高速耦合失配？它不是安全认证计划，也不先做完整 diffusion、完整不确定度理论或 CUDA-MPC 移植。先完成一个可复现的 22 状态最小原型；只有原型显示可重复信号，才扩展执行器状态、预测器和正式留出。

被测算法简称 **CAT-MPC**（causal adaptive-trust MPC）：历史适配器给出有效的舵响应修正，因果残差预测器给出 MPC 时域内的未解释体加速度残差，连续 PPO 策略输出两者的信任系数。参考速度始终为 15 m/s；RL 不选择速度、候选集合、舵角或推力。

现有强基线是 `robust_condensed_endpoint35`：MPC 点模型使用舵效 `0.85`、舵时间常数 `2.0` 的固定保守修正。历史报告中的新留出结果为 14/20，不能在本计划中重新调参后继续称为盲测。所有 pilot 结果先报告 `strict_hist`、每 0.02 s 物理违约、MPC fallback、backup/intervention 和 deadline 分项；没有完整子步事件日志时，不得把回合写成 `strict_mainline` 成功。

## 1. 最小算法实现

### 1.1 状态、时间和数据接口

| 项目 | MVP 定义 |
| --- | --- |
| 控制与积分 | MPC 每 0.1 s 求解一次；植物每周期推进 5 个 0.02 s 子步 |
| MPC 状态 | 保留已有 22 状态：艇体 10、实际控制量 3、三段最新命令队列 9 |
| 参考 | 固定目标速度 15 m/s 和现有末端切线延拓；不使用速度意图 |
| 历史窗口 | 最近 10 条 0.1 s 观测（约 1 s），只含已完成的状态、实际执行器、队列、命令、上一计划和误差 |
| 策略更新 | 每 5 个 MPC 周期（0.5 s）更新一次；更新之间零阶保持并限幅变化 |
| MPC 更新 | 每个 0.1 s 都用当前测量状态重新 rollout、计算 Jacobian、构造并求解 QP；不能因为门控保持就复用过期的状态 Jacobian |
| 扰动维度 | 先用 5 维体加速度残差 `[u_dot,v_dot,w_dot,q_dot,r_dot]`；它是未解释响应残差，不命名为真实外流 |

MVP 不把隐藏的二阶执行器 rate 加进控制器状态。目标植物仍可使用已有 `SecondOrderActuator`，因此结果只说明一个有效 22 状态修正接口是否有信号。若该接口通过 pilot，再单独做 25 状态（22 + 三个 rate）实验；不能把 22 状态结果写成已经精确建模二阶执行器。

### 1.2 历史适配器

第一轮直接复用并冻结 `experiments/h15_mainline/response_estimator.py` 的因果滑动窗估计器，避免同时训练适配器和策略。它输出有效响应参数

\[
\hat\theta_k=(\hat g_k,\hat\tau_k),
\quad \hat g_k\in[0.60,1.10],\quad \hat\tau_k\in[0.5,3.0].
\]

估计失败、激励不足、历史过短或 OOD 时返回固定 `theta0=(0.85,2.0)`。这里的参数是短窗有效模型，不是隐藏植物真值。后续才可用小型 GRU/1-D CNN 蒸馏该估计器；适配器冻结后才能比较不同 PPO 种子。

为了避免残差分支重复补偿，先在周期 $k$ 用历史 $j\leq k$ 得到 $\hat\theta_k$；周期完成后才计算

\[
r_k=\frac{\nu_{k+1}-\operatorname{Step}_{\hat\theta_k}
 (\nu_k,a^{\rm actual}_k)}{0.1},
\quad \nu=[u,v,w,q,r].
\]

残差计算使用实际执行器，而不是目标命令或下一周期更新的 \hatθ。现有 current 估计与参考补偿保持原样；不要把同一 current 同时作为位置残差和动力学前馈注入。

### 1.3 因果残差预测器

先使用最小的持久值/AR(1)基线或 2 层 GRU，输入最近 10 个 (r)，输出未来 35 个 0.1 s 的均值 \hat d_{k:k+34}。首轮只需要均值和经验绝对误差，不等待正式 99% 联合覆盖。训练数据取 4 个 train 几何 × 3 个植物/扰动种子共 12 回合，按整回合切分窗口；归一化只用 train。

预测器只能看到过去残差和当前几何，不看到未来 current、隐藏舵效、二阶 rate 或 `state_after[k]`。P08 的 diffusion 只作为后续教师/消融：如果 AR/GRU 在 dev 上没有比 persistence 明显降低 1/5/25 步误差，就删除扩散分支，不为使用复杂模型而保留它。

### 1.4 连续门控与点模型

策略观测为

\[
o_k=[\hat\theta_k,\hat d_{k:k+34},\text{预测误差摘要},
\text{路径局部几何},\text{实际余量},\text{上一门控},\text{上一 QP 状态}].
\]

不读取新门控尚未产生的 QP 余量；这些余量只能来自当前测量和上一个已完成周期。一个默认的 64×64 MLP 输出原始高斯变量 (z_k\in\mathbb R^2)：

\[
\alpha^{\rm target}_k=\sigma(z_k),\qquad
\alpha_k=\operatorname{slew}(\alpha_{k-1},\alpha^{\rm target}_k),
\quad \alpha_k=(\alpha_{\theta,k},\alpha_{d,k})\in[0,1]^2.
\]

策略动作每 0.5 s 更新并保持 5 个 MPC 周期。PPO 保存 sigmoid 前的 `z`、旧策略 `log_prob(z|o)`、value、实际应用的 alpha 和 slew/override 原因；ratio 使用同一个 raw Gaussian 的 log probability，不能对 slew 后的 alpha 重新计算高斯密度。初始策略的均值可为 0（alpha 约 0.5），但 A0 的 `(0,0)` 是单独的确定性基线，不能混淆。

MVP 的点模型以固定保守模型为零点：

\[
f^{pt}=f_{\theta_0}
 +\alpha_\theta(f_{\hat\theta}-f_{\theta_0})
 +B_d\alpha_d\hat d,\qquad \theta_0=(0.85,2.0).
\]

这样 `(alpha_theta, alpha_d)=(0,0)` 才与 B0 一致；若使用名义 `1.0/1.0` 作零点，就无法公平解释 RL 增益。`B_d` 只映射到艇体速度/角速度导数，不修改实际执行器和队列状态。建议扩展 `simulator/rti_symbolic.py` 的数值扰动输入，保持 CasADi 图和稀疏结构；不在每周期重建图。

所有 alpha 共享 B0 的命令边界、回退和子步物理审计。首轮不做 alpha 网格枚举，也不让 actor 改风险集合；固定 alpha 对照只是离线灵敏度检查，不是离散动作空间。

### 1.5 奖励、折扣和终止

每个策略决策覆盖 (m=5) 个控制周期，先把 5 个周期奖励折为

\[
R_k=\sum_{j=0}^{m-1}\gamma_c^j r_{k+j},\quad
\gamma_c=0.995,
\quad \Gamma_k=\gamma_c^m.
\]

PPO 的 GAE 使用 `R_k + Gamma_k * V(next) - V`；物理违约、QP 失败和主动 backup 是 intervention/failure，终止时不 bootstrap；行政截断才 bootstrap。奖励包括路径进度、固定 15 m/s 切向对水速度误差、横向/深度/俯仰误差、饱和、fallback、backup 和 alpha 变化，不允许通过降低目标速度取巧。每次决策即使前一周期发生 fallback 也按固定 5 周期节拍记录；不能漏掉决策间隔。

### 1.6 一次控制周期的实际调用顺序

在时间戳 $t_k$，`run_episode.py` 按以下顺序执行一次循环；这个顺序是实现契约，不能把 actor 放到新 QP 之后形成循环观测。

1. 从环境读取 `x_k[10]`、`actual_k[3]`、`queue_k[9]`、上一条实际命令、当前路径几何/误差和上一周期 QP 结果；拼成历史末行。历史窗口形状为 `[T=10,D=34]`，其中每行是 `[x(10), actual(3), queue(9), issued_command(3), residual_body_acc(5), path_errors(4)]`。残差在第一轮没有标签时填零并带 `residual_valid=0`，不能用未来状态补齐。
2. 让冻结的 response estimator 消费 $j\leq k$ 的历史，得到 `theta_hat=[gain,tau_scale]`、`theta_source` 和 `fallback_reason`。将参数投影到 `[0.60,1.10]×[0.5,3.0]`；失败就返回 `theta0=[0.85,2.0]`。
3. 让冻结的 predictor 消费最近 10 个已完成的 `r` 和当前几何，输出 `d_hat[35,5]`。MVP 的 actor 不直接使用完整 175 维预测，而使用固定的 15 维摘要 `[mean(5), first_step(5), turn_window_mean(5)]`；完整预测只传给 MPC。
4. 当 `k mod 5 == 0`，把 `[theta_hat(2), theta_radius(2), d_summary(15), margins(6), qp_status_and_last_time(2), alpha_prev(2), geometry(8)]` 组成 37 维归一化观测，送入 64×64 actor。actor 返回 raw `z[2]` 和 value；保存 raw Gaussian log-prob。sigmoid 与 slew 后的 `alpha[2]` 保持五个周期。其余周期只更新预测和 MPC，不重新采样策略。
5. 用当前 `alpha` 构造 MPC 点模型。`theta0` 对应既有 `.85/2.0` 动力学；`alpha_theta` 插值到 `theta_hat`，`alpha_d` 只把 `d_hat[k:k+35]` 作为艇体 5 维导数偏置。每个 0.1 s 都从当前 `x_k, actual_k, queue_k` rollout N35、重新求 A/B/Jacobian、构造一个 QP 并调用一次 OSQP；不枚举 alpha，也不沿用上周期 Jacobian。
6. 取优化计划第一条 `[thrust,elevator,rudder]` 命令交给植物，推进 5 个 0.02 s 子步，记录每个子步的状态、actual、queue、物理 flags 和墙钟。命令盒、回退和现有 current-reference 补偿保持 B0 一致。
7. 周期结束后，用实际执行器和预更新的 `theta_hat_k` 计算 `r_k=(nu_{k+1}-Step_theta_hat(nu_k,actual_k))/0.1`，写入下一窗口；然后才进入 `k+1`。如果物理违约、QP 失败或 backup，记录 intervention 并终止该 rollout，不把 supervisor 命令伪装成 actor 动作。

对应代码改动边界为：`simulator/rti_symbolic.py` 为 actuator-aware dynamics 增加数值 `d[5]` 输入并将其加到 hull derivative 的索引 5:10；`aav_mpc/condensed_mpc.py` 在每个预测步传入 `d_hat` 的仿射偏置；`experiments/h15_cat_mpc/run_episode.py` 实现上述时间顺序和日志。22 状态的 A/B 维度不变，CasADi 图和稀疏结构复用；25 状态 rate、diffusion、tube 和 GPU MPC 均不属于 MVP。

## 2. 并行与 GPU 执行方式

### 2.1 当前 CPU 路径

`ActuatorAwareEnv` 明确只接受 `rti_cpu`；`CondensedActuatorMPC` 在 `condensed_mpc.py` 中逐样本调用 NumPy/CasADi 和 OSQP。现有 CUDA RTI 是 10 状态名义模型，拒绝 actuator-aware 模式，不能冒充 H15 GPU 后端。因此首轮直接采用独立进程并行：

- [ ] 每个 job 固定 `(method, geometry_id, plant_seed, current/noise_seed, policy_version)`，写入唯一目录和 `run.json`；父进程只做汇总。
- [ ] 本机 10 个物理核、16 GiB 内存先用 3–4 个 worker；每个 worker 设置 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、Torch threads=1，使用 `spawn`，不要每个 job 再开线程。
- [ ] A0–A4 和 P2 checkpoint 评估可按 job 完全并行；paired 方法必须共享 immutable noise/current tape。当前 batch generator 会因 batch 顺序改变随机数消费，不能仅靠相同 global seed 保证配对。
- [ ] 汇总器按 config/source/input hash 缓存结果；改动动力学、adapter、predictor、策略或 manifest 时使缓存失效。

### 2.2 可用 GPU 时

- [ ] 只把 adapter/predictor 的 minibatch 和 PPO actor/critic 更新放到单张 CUDA/MPS GPU；保持 rollout 和 CPU OSQP 分离，收齐同一 frozen policy 的轨迹后再更新。
- [ ] 小型 64×64 actor 先比较 CPU 推理和批量 GPU 推理；若每 0.1 s CPU↔GPU 拷贝更慢，就保留 CPU inference。GPU batch 吞吐不能写成单控制器延迟。
- [ ] 需要大批量 replay 时，先把独立 scenario 维度向量化，保持 0.02 s 递推；比较 CPU/Torch 的状态、队列、accepted command、物理 flags 后再用于筛选。
- [ ] 不在 P2 为 GPU 移植 22 状态 OSQP/CasADi。若未来移植，必须先做逐周期数值 parity、Jacobian parity 和单样本端到端测量。

当前工作机没有可假设的 NVIDIA GPU；计划中的 GPU 项是条件路径。正式 p99 测量必须 batch=1、无并行 worker、预热后串行运行，并把 CUDA kernel 用同步事件计时；并行吞吐只作工程诊断。

## 3. 评价口径和停止条件

每个 job 至少保存 `config.json`、代码/输入 hash、scenario/noise seed、policy version、逐周期 alpha/raw z/logp、adapter source、predictor version、QP status、command、22 状态、子步 flags 和 timing。主表按 paired unit 输出：

| 层级 | 必报指标 | 用途 |
| --- | --- | --- |
| 物理 | full path、0.02 s 违约、首次违约类型/时刻 | 判断是否真正改善失配 |
| 控制 | MPC fallback、backup/intervention、空集、命令/执行器饱和 | 分离策略收益与接管 |
| 任务 | 切向对水速度误差、横向/深度/俯仰峰值、进度、完成时间 | 防止降速取巧 |
| 学习 | alpha 分布、变化率、raw logp、value、seed 间方差 | 判断 RL 是否学到长期门控 |
| 工程 | 周期 p50/p95/p99/max、adapter/predictor/MPC 分项 | 区分吞吐和实时性 |

按以下规则停止，不为“完成更多验证”而堆叠阶段：

- [ ] P0 等价性失败：停止，先修接口。
- [ ] P1 固定门控无命令/余量影响：停止 PPO，检查残差单位、MPC 注入和 observation。
- [ ] 预测器不优于 persistence：固定 `alpha_d=0`，继续响应支路，不阻塞整个算法。
- [x] P2 未达到任一探索门槛且学习曲线无持续改善：已报告 pilot 负结果，不扩到 8,000 或三种子。PPO 相对 A0 仅 1/6 配对严格主线改善，归一化跟踪误差中位数下降约 4.0%–4.4%（门槛 10%），切向速度误差略增；相对 A4 有 1 个改善、1 个恶化，无独立 RL 增益信号。
- [x] PPO pilot 未胜过最强确定性 gate：保留 A4 结果，当前结论为这个单策略种子没有独立 RL 增益；不运行多种子/留出扩展。
- [ ] 只靠降低实际速度、增加 backup、放宽判据或未来信息获得改善：判为无效。
- [ ] 只有 P3 稳定通过才进入 P4；P4 失败时仍保留 P3 的真实结论，不回写成算法成功。

## 4. 按顺序执行的实验步骤

下面每一项都是按依赖关系排列、可直接勾选的执行步骤。前一步未通过时，不进入下一步；括号中的数量是 pilot 预算，不是统计显著性承诺。

### P0：冻结输入并完成短接线检查（半天级）

- [x] 按用户指定的 `/bin/python3` 执行 `import numpy, torch, casadi, osqp`，检查 `torch.cuda.is_available()` 和可用 CPU；四项依赖均可导入，CUDA 不可用，CPU 计数 16。此处遵循用户指定解释器，不采用本计划原先的解释器建议。
- [x] 记录代码 commit、Python/Torch/CasADi/OSQP 版本、CPU/GPU、线程环境；固定 `h15.json`、15 m/s、N35、末端延拓和 target-domain 参数。版本为 Python 3.10.12、Torch 2.14.0+cu130、NumPy 2.2.6、SciPy 1.15.3、CasADi 3.6.5、OSQP 1.1.3；commit `46b329c1d70681f389ae82b030ac4ae78d782434`。配置在 `experiments/h15_cat_mpc/config.json`，冻结输入在 `outputs/h15_cat_mpc_20261105/freeze/quick_run_manifest.json`；线程变量初始未设置，实验 worker 配置为单线程。
- [x] 复用现有 B0：`robust_condensed_endpoint35`，固定 \hatθ=`(.85,2.0)`，target 植物使用 `fin_gain=.9`、二阶执行器、harmonic current、process noise；不调参数。配置和现存 target-domain 参数已对照冻结。
- [x] 建立 `experiments/h15_cat_mpc/`，已包含要求的配置、适配器、残差预测器、连续策略、PPO、单回合 runner、评估器和汇总器；新 runner 的子进程入口使用当前 `/bin/python3` 进程环境。
- [x] 审计历史估计和残差标签的时间戳：runner 在植物推进后才补齐 `state_after` 并生成残差；下一次 MPC/策略观测消费刚完成的历史行，选择本周期 alpha 前不读取本周期未来状态。PPO 动作每 5 周期更新一次。
- [x] 对同一 snapshot/reference/queue 运行 B0 和 CAT-MPC `(0,0)`；第一条命令和预测 22 状态最大差均为 0（同一 OSQP 求解器、相同输入；新动力学相对 commit 原实现逐点最大差也为 0）。
- [x] 对 `(1,0),(0,1),(1,1)` 各做一次有限值/Jacobian 检查；输出的 `f/A/B` 均有限，零预测残差时 `(0,1)` 不改变 B0，\hatθ=θ0 时 `(1,0)` 不改变 B0。扰动只影响艇体状态，不改执行器/队列。
- [x] 在一个简单、一个耦合、一个目标失配短轨迹各运行 40 个控制周期；三类场景都完整推进 40 周期，固定 `(1,1)` 在每周期改变第一条命令，且轨迹/预测随之改变。探针使用小型数值 d 偏置，仅验证接线，不作为性能评估。

P0 的通过条件是：零门控等价、没有未来观测、每周期仍只求解一次 MPC、alpha 不越界。当前 H15 condensed 链是 CPU NumPy/CasADi/OSQP；不要把 `device=cuda` 当作 GPU 加速开关。

### P1：冻结最小适配器和预测器，验证动作确实有作用（可并行）

固定门控对照定义如下，所有项都用同一冻结的适配器/预测器和同一场景：

| 方法 | $\alpha_\theta$ | $\alpha_d$ | 含义 |
| --- | ---: | ---: | --- |
| A0 | 0 | 0 | B0 固定 `.85/2.0`，不使用残差前馈 |
| A1 | 1 | 0 | 只使用历史响应修正 |
| A2 | 0 | 1 | 只使用因果残差预测 |
| A3 | 1 | 1 | 两个修正都完全使用 |
| A4 | 0.5 | 0.5 | 中间连续信任强度 |

- [x] 用 train 的 4 个几何 × 3 个种子收集 12 个完整回合，共 2,388 个控制周期；按回合顺序生成历史适配器输入和因果残差标签，未跨回合拼接训练窗口。
- [x] 冻结既有 response estimator 与 train-only 归一化，训练并保存 AR(1) 预测器、残差标签、模型参数、12 个输入 trace hash 和源文件 hash。预测器在 dev 汇总上 1/5/25 步 MAE 均低于 persistence，因此保留 alpha_d 支路；该结果是小样本 pilot。
- [x] 在 dev 的 2 个几何 × 3 个种子上并行运行固定门控 A0–A4，共 30 个配对回合；每回合 batch size=1、独立输出目录，同一几何/种子的环境随机数生成器。
- [x] 记录 1/5/25 步残差误差、配对命令差异、首个物理违约、MPC fallback、backup、路径进度、速度误差和完整周期时间；deadline 单独按实测周期时间记录。A4 有 1 个 strict_hist 配对改善、0 个恶化；A0/A4 strict_mainline 分别为 4/6 与 5/6。A2 的物理违约增加，固定门控不构成整体成功结论。
- [x] 固定门控在 30 个回合中均改变预测或命令，A1–A4 至少 5/6 个 paired unit 存在命令差异；接口有控制作用，继续进入 PPO pilot。此处仅通过 P1 机制门槛，不代表 PPO 有增益。

A0–A4 是固定连续值干预，不是 RL 候选动作集合；它们只用于证明接口有控制作用和提供确定性对照。P1 不跑完整 diffusion、不建立完整联合覆盖证明、不做 50 Hz telemetry 迁移。

### P2：一个种子的快速连续 PPO（主要可行性验证）

- [x] 固定 P1 的 adapter/predictor checkpoint，不在 PPO 期间更新它们；训练只让 actor/critic 学习两个 alpha。冻结 hash 见 `outputs/h15_cat_mpc_20261105/p2_seed41001/frozen_inputs.json`。
- [x] 以 4 个 train 几何 × 3 个种子收集 on-policy rollout；每次 PPO 更新前等待全部 worker 返回，同一批轨迹使用同一 policy version。
- [x] 使用策略种子 `41001` 完成 4,000 次门控决策，保存 0/1,000/2,000/4,000 checkpoint。配置为 PPO `lr=3e-4`、clip=.2、4 epochs、minibatch=64、gradient clip=1；结果是 pilot，不声称收敛。
- [x] 4 个 checkpoint 与 A0–A4 在相同 6 个 dev 单元上配对评估，共 24 个 checkpoint 回合；P1 的 30 个固定门控回合使用同一 dev 单元和 scenario/noise seed。
- [x] 已绘制 alpha 分布、raw Gaussian 标准差、fallback/backup、0.02 s 物理事件、周期延迟和 reward 曲线，见 `outputs/h15_cat_mpc_20261105/p2_dev_checkpoints/p2_diagnostics.png`。alpha 未饱和，但均值靠近 A4 固定门控，reward 曲线无持续改善。

P2 的“可用信号”满足以下任一探索条件且没有物理完成恶化：

1. 六个配对 dev 单元中至少两个由失败变为无干预物理完成，恶化数为 0；
2. 完成率持平时，共同成功回合归一化跟踪误差中位数下降至少 10%，完成时间增加不超过 3%，实际切向对水速度误差不增加；
3. 完成率和误差不恶化，backup/fallback 周期比例下降至少 20%，且改善不由单个回合独占。

这些是进入下一阶段的探索门，不是统计检验或安全阈值。若 4,000 决策无收益且学习曲线无持续改善，停止扩预算；只有仍在改善且动作确实有效时，才允许一次扩到 8,000。

### P3：多种子和初步未见留出（仅 P2 通过后）

- [ ] 追加 PPO 策略种子 `41002,41003`，每个同样 4,000 次门控决策；三个策略都在 6 个 dev 单元评估，不挑最佳种子。
- [ ] 训练同预算的标量门控 `alpha_theta=alpha_d`，作为二维动作必要性的消融；固定规则 gate 与固定 alpha 使用相同保持周期和 slew 限制。
- [ ] 冻结 adapter、predictor、策略、阈值、随机种子和 checkpoint 后，在 `h15_mpc_fix_holdout/manifest.json` 的 4 个未见几何 × 5 个植物/扰动种子上运行 A0、最强确定性 gate、三个 PPO 策略和标量策略，共 100 个配对回合。
- [ ] 报告每个方法的 `strict_hist`、0.02 s physical strict、MPC fallback、backup/intervention、deadline、速度/姿态/深度/横向误差和 p50/p95/p99/max；按几何聚类汇总改善/恶化，不把周期数当独立样本。
- [ ] 若 RL 不超过最强确定性 gate，保留确定性结果并结束 RL 扩展；若只降低实际速度、增加 backup 或放宽判据，判为失败模式。

### P4：出现稳定信号后才做的扩展

- [ ] 用独立 `dev_cal/dev_eval` 重新校准逐步残差集合、覆盖率和区间宽度；现有仓库没有这两个独立 split，必须先按整回合/几何拆分后再生成窗口。
- [ ] 若一阶有效模型仍不足，增加 25 状态（额外三维 actuator rate），用 measured rate 或滤波差分输入，不读取 `SecondOrderActuator.rate` 隐变量；先比较 B0_22 与 lifted B0_25，再评价 RL。
- [ ] 只有简单预测器显示可预测增益时，才训练 diffusion teacher 并蒸馏到 GRU/TCN；报告推理尾延迟，不引用 P08 的平均时间作为 H15 证据。
- [ ] 最后才进行 50 Hz telemetry、固定集合/tube 屏蔽、状态约束屏幕和串行实时测量；这些属于部署可信度，不阻塞 P2 的离线算法可行性判断。

### 4.6 运行入口（拟新增，不把未实现命令当现有结果）

建议新增 runner 后使用以下结构；名称是计划接口，首次实现时需写入 `--help`、config 和 manifest：

```bash
# 生成/冻结输入（已有脚本可复用，输出目录必须唯一）
python -m experiments.h15_mainline.freeze_quickstart --out <freeze-dir>

# P1 固定门控并行 screen
python -m experiments.h15_cat_mpc.evaluate \
  --config experiments/h15_cat_mpc/config.json \
  --split dev --methods A0 A1 A2 A3 A4 --workers 4 --out <screen-dir>

# P2 一个 PPO 种子；rollout worker 与 learner 分离
python -m experiments.h15_cat_mpc.ppo_train \
  --config experiments/h15_cat_mpc/config.json --seed 41001 \
  --decisions 4000 --workers 4 --device auto --out <pilot-dir>

# P2 checkpoint 配对评估
python -m experiments.h15_cat_mpc.evaluate \
  --config experiments/h15_cat_mpc/config.json --split dev \
  --policy-checkpoints <pilot-dir>/ckpt_*.pt --workers 4 --out <eval-dir>
```

这些命令只描述所需入口；本报告没有声称它们已经存在或已经产生结果。实现时必须先让 `A0` 复现现有 `h15_robust_prediction.py` 的配置，再接入 actor。

### 4.7 预期结论形式

第一轮只允许得到三种结论：

1. **接口无效**：alpha 不改变点模型/命令，或 predictor 无信号；停止并修复/删除对应支路。
2. **连续门控有机制信号但 RL 无独立增益**：固定或确定性 gate 已足够，保留确定性方法。
3. **RL 初步有效**：P2/P3 在相同信息、固定 15 m/s、无物理完成恶化且不靠 backup 取巧的情况下超过最强确定性 gate；这只允许进入 P4，不是安全证明或最终泛化结论。

报告所有结果时区分 pilot、dev 和未见 holdout；不把单个 checkpoint、教师策略、oracle 参数、并行吞吐或 P08 平均时间写成部署性能。
