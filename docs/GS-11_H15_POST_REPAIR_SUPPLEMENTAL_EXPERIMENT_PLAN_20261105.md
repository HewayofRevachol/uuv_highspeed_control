## 结论与适用范围

- [ ] 总体顺序：先重建与修正接口兼容的 adapter/predictor，确认 gate 确实改变了模型，再验证固定/规则 gate，最后有条件地训练 PPO。不要把本轮 A0/A4 相同的完成数当作完整算法无效，也不要直接恢复原来的长程训练。

执行工作区按用户指定为 `~/uuv_workspace`，下文路径均相对此项目根目录。此文档是补充实验执行计划；本轮没有运行其中的仿真或训练。

### 已有证据及边界

2026-11-05 修复执行者在 GS-11 评论中报告：12 项针对性单测与 compileall 通过，最终 parity 为 2 geometry × 3 seeds × 2 methods = 12 回合：

| 方法 | physical strict | fallback 周期 | deadline clean | 对 A0 的物理改善/恶化 |
| --- | ---: | ---: | ---: | ---: |
| A0 `(0,0)` | 5/6 | 0 | 4/6 | — |
| A4 `(.5,.5)` | 5/6 | 0 | 4/6 | +0/−0 |

A4 的 deadline 配对为 +1/−1。证据位置由该评论给出：`docs/GS-11_H15_REPAIR_VALIDATION_20261105.md` 和 `outputs/h15_cat_mpc_20261105/repair_parity_verified_20261105/`。本次规划运行可访问的 GitHub 检出仍为修正前 `c9120d0`，未取得这些修正源码/原始日志；上述数字属于执行者报告，未在本次独立复算。执行 E0 时应直接在 `uuv_workspace` 核对，不能用旧 GitHub 版本替代修正版。

修复说明同时明确：adapter/predictor 尚未重新拟合、旧 predictor 标签不兼容。旧实现中，无 adapter normalization 会回退 E0，令 `theta_hat=theta0`，因此 theta gate 无作用；无 predictor checkpoint 的默认行为是 persistence，初始无历史时为零，之后跟随最近 residual，**不是必然全程零预测**。修正版必须记录实际加载和回退情况，才能解释本轮 parity。

## 1. 固定实验契约

### 1.1 控制设置与可用信息

- [ ] 固定控制契约：参考速度 15 m/s，控制周期 0.1 s，植物 5×0.02 s 子步，MPC N35/22 状态，PPO 每 5 个控制周期更新 alpha，`theta0=(.85,2.0)`。沿用参数插值实现：

`theta_applied = theta0 + alpha_theta * (theta_hat - theta0)`；`d_applied = alpha_d * d_hat`。

- [ ] 固定信息边界：目标域包含额外 fin gain ×0.9、二阶执行器和已声明积分器/海流配置；记录真实植物参数用于离线审计。策略、估计器和 nominal rollout 只使用可观测状态/队列、已发命令、固定名义参数与因果估计，不把真实随机 tau、delay、二阶 rate 或未来扰动作为策略输入。局部分支恢复可使用模拟器内部状态，但只能用于保证同一环境。

### 1.2 指标

- [ ] `physical_strict`：完整到达且所有有效子步满足 |横向误差|≤8 m、|深度误差|≤5 m、|切向速度误差|≤2 m/s、|实际 pitch|≤40°。
- [ ] 主算法指标 `no_fallback_physical`：physical strict 且实际执行命令没有 QP fallback。若未来启用 backup，另记 backup 事件并加入“无干预”条件；未实现时 `backup_supported=false`，backup 字段为 N/A。
- [ ] deadline 独立给超时周期数、超时回合数、p50/p95/p99/max。不得用 deadline-only 胜负替代物理改善。
- [ ] 共同无干预成功回合：每回合计算 `R=sqrt(mean[(lat/8)^2+(depth/5)^2+(speed_error/2)^2+(pitch_deg/40)^2])`，报告每对 `R_candidate/R_base-1`、其分布/中位数和各自 RMSE 中位数之比；基线 R=0 的对只报绝对差，不除零。
- [ ] 对失败回合报告首个违约时间、类型、进度、约束峰值；不与全程成功回合池化平均误差。饱和率按有效执行周期计，定义任一通道达到其物理上下限（按控制量程容差 1e-6）。

配对键为 geometry ID × environment seed，包含同一初始状态、植物/海流随机流。独立性主要来自 geometry cluster；周期、重复动作抽样和相同 seed 的重复运行均不能当作新增独立场景。

### 1.3 数据拆分与种子

| 数据集 | 规模与用途 | 使用限制 |
| --- | --- | --- |
| D_diag | 原 2 geometry × seeds 32001–32003 = 6 单元 | 重现修复前后现象、局部诊断；已经看过，不能称新留出 |
| T_fit | 4 个训练 geometry × seeds 51001–51003 = 12 单元 | 重新生成 normalization、adapter 和 AR；几何可沿用原 train，不能用 D_diag 拟合 |
| D_select | 新 4 geometry × seeds 52001–52005 = 20 单元 | 选固定/规则 gate、PPO checkpoint；属于开发集，可反复使用但不能作为最终泛化证明 |
| H_final | 另新 4 geometry × seeds 53001–53005 = 20 单元 | 一次最终评估；不得与 D_select 共享轨迹或几何模板参数实例 |
| T_boundary | 4 个额外训练 geometry × seeds 54001–54003 = 12 单元 | 仅在 E5 需要时加入边界训练；从预先声明的曲率/深度范围独立生成，不复制 dev/test 失败轨迹 |

- [ ] 生成并冻结数据 split：预先覆盖弯曲/深度耦合、反转、进入机动等形态，保存完整参数和 CSV hash；检查 seed 是否已使用并在 manifest 固定。每个 split 的具体轨迹必须不同，同一路径换 seed 不算几何隔离；历史 `h15_mpc_fix_holdout` 不得充当 H_final。

## 2. E0：核实修正与支路活动（先决条件）

- [ ] 复核已有 12 回合：检查源码/产物 hash、target 参数快照、schema、子步指标，并保存 adapter/predictor 的真实路径、hash、模型版本及 fallback 原因。缺文件时停止，不得静默替换成 E0/default 后沿用同一方法名。

- [ ] 验证两个时序：target 参数在本次 reset 随机化后且首个 solve 前生效；issued command 在植物积分前捕获。父类 step 会调用下一次 solve/complete_pending，不能等整个父类 step 返回后才补写本周期命令。

- [ ] 运行 12 个短探针（各 40 控制周期）：3 类环境〔完全匹配 nominal/无扰动、匹配 nominal/已知加速度扰动、真实修正 target〕×2 补偿模式〔on/off〕×2 命令序列〔阶跃/反转〕。已知扰动仅作为测试 oracle，不能喂给策略。

验收值在运行前冻结：

- [ ] reset 重复/部分 mask 重置后，未重置槽位不变；二阶执行器 rate 对重置槽清零；额外 ×0.9 不丢失也不重复相乘。数值比较使用 float64、`rtol=1e-10, atol=1e-12`。
- [ ] `command_json`/issued、trace 和实际队列首项一致，控制量程归一化最大差≤1e-6；mpc command、限幅前补偿和最终 command 分列，不能靠重算一次低层控制来“取日志”。
- [ ] nominal rollout 与 MPC 点模型使用同一队列顺序、5×20 ms 更新和名义 delay/tau。完全匹配且同一积分器的无扰动探针，残差除以 `[1,1,1,.1,.1]` 后最大绝对值≤1e-6。target 的二阶/积分器失配残差不要求为零；它仍是总模型误差，不是纯海流。
- [ ] 新旧命令/label schema 混用必须显式失败；`alpha_used`、`theta_label_before`、forecast 和 reward 按 period 对齐，最后一个终止周期恰好完成一次。

- [ ] 记录支路活动覆盖：计算 `A_theta=max(abs(theta_hat-theta0)/[.5,2.5])>1e-3` 和 `A_d=max(abs(d_hat)/[1,1,1,.1,.1])>1e-3`（预测时域取最大值），报告每回合 E0/E1 比例、low_excitation 原因、两轴活动比例、模型增量和 command 增量。用人工非饱和正控制的 0/1 gate 确认通路连通；真实运行中 command 不变也可能来自约束饱和。

- [ ] 按 E0 结果决策：失败则停止性能比较、修接口并另开 run；某轴不活动则标记“该支路尚不可评价”并先做 E1，不据此宣布 RL 无效。

## 3. E1：重建数据、adapter 和 predictor

- [ ] 先完成 E1 数据采集，再进行 gate 性能对照：按 T_fit 的 12 单元，用 A0 分别采集低层补偿 on/off，共 **24 个完整回合**。两模式使用相同环境种子；记录 issued command、实际 actuator、queue、状态和所有有效时间步，失败回合也保留。

按以下因果顺序处理：

- [ ] 在补偿 on 的 12 个 train 回合拟合 normalization，冻结现有估计器参数；将补偿模式写入 artifact。补偿 off 的 12 个回合单独生成匹配 normalization/adapter，不能把对照模式产生的分布变化归为 predictor 收益。
- [ ] 用冻结估计器从每回合起点顺序 replay。对周期 k 用截至 k−1 的历史得到 `theta_label_before`，先生成 k 的 nominal label，再消费 k 的观测；队列/实际 actuator 可观测，真实随机植物参数不可读。在线与离线共用同一 rollout 函数，固定 label 所依据的是未被 gate 混合的因果估计 theta，不能随意切换为 theta_applied。
- [ ] nominal 标签版本记为实际修正版 schema，不预设字符串一定是 `issued_v2`；禁止用重命名把旧 artifact 伪装为新版本。保存数值公式、dt、theta 契约和输入/源码 hash。
- [ ] 每个补偿模式分别准备 zero、persistence、新 AR 三个预测器；AR 按回合建滞后对，不能跨回合连接序列。旧/新 label 可以在相同轨迹上离线比较，用于定位瞬态差异；旧 label predictor 不加载进正式修正版 runner。
- [ ] 12 个训练单元原则上各至少有 100 个有效周期；不足时记录缺口，以同规则新增训练种子补到覆盖数并单列预算。NaN/损坏行不能填零参与训练；间断序列重新起段。

- [ ] 产出两套模式明确的 `normalization/adapter/predictor`、label manifest、训练 MAE 和活动覆盖。先运行 E2 计划中补偿 on 的 6 个 A0 回合（计入 E2 的 30，不额外增加），用固定轨迹选主预测器；A0 不受预测器门控影响。随后在 D_diag 报告 1/5/25-step 分量 MAE，每单元至少 50 个有效 h25 目标对；不同量纲的五维 MAE 不直接相加。

- [ ] 选择主预测器：h5 或 h25 至少 3/5 分量相对 persistence 改善≥5%，其他分量无 >5% 退化且两个几何方向一致；否则采用 persistence（必要时 zero）并记录 AR 未过门。该门只选择预测器，不等同于 gate 闭环结论。

## 4. E2：完整固定门控与补偿消融

- [ ] 运行 E2 主消融：固定补偿 on、新 adapter 和 E1 主预测器，在 D_diag 跑 A0–A4，共 **5×6=30 回合**；A1=(1,0)、A2=(0,1)、A3=(1,1)。A0 也计算相同模块，纯 B0 开销放入 E7。

- [ ] 完成 E2 消融：运行 A4 ×〔zero/persistence/AR〕×〔补偿 on/off〕×6 单元=36 回合，加补偿 off 的 A0×6=6 回合，共42；复用前述 A4/on 六回合后最多新增36。每种补偿模式与自己的 A0 比较；AR 未过 E1 门时只能作为预注册负结果。

- [ ] 生成 E2 报告：给出物理/无干预配对表、共同成功逐对 RMSE、首个违约、gate 活动和时延；zero/persistence/AR 离线误差在同一套 A0 轨迹上评估。

- [ ] 按 E2 结果决策：不要求复现旧 A4 的 6/6；若两轴无活动则补激励/辨识或降为单轴，若活动充分但候选均恶化则停止 PPO 并报告负结果。

## 5. E3：局部干预，判别是否需要状态相关 gate

- [ ] 选取 E3 快照：从 A0 的 D_diag 每单元取两个固定时刻（首个几何变化前 0.5 s；或余量首次≤0.3，未触发则取第二几何变化前 0.5 s），只用 A0 状态选时刻，共 12 个。

- [ ] 保存并恢复完整 snapshot：hull、actual actuator、命令队列、二阶 actuator rate、海流/噪声 RNG、路径进度、MPC plan/previous、求解器 warm start、估计器/预测器历史、alpha/保持计数和补偿估计；先复跑 A0 检查恢复前后 hash 一致。

- [ ] 运行 E3 局部扫描：每快照令 `a∈{0,.25,.5,.75,1}`，测试 `(a,0),(0,a),(a,a)`，去重 13 组，共 **156 个短分支**，每分支 20 周期/2 s 或到终止；冻结 gate，其余估计和控制因果更新。

- [ ] 按共同前缀比较 E3：先按 2 s 内是否违约、违约时间、fallback 排序，再按共同前缀误差排序；共同前缀不足 5 周期不比较平均误差，避免早失败因截短获得低误差。

- [ ] 判断 E3 活动门：每轴至少 4 个活动快照且覆盖两几何；候选至少在 4 个快照上避免违约或共同前缀 RMSE 降≥5%，其余快照不新增物理违约/fallback，饱和率增幅≤5 个百分点。该门仅用于缩小候选。

- [ ] 形成候选规则：若阶段优胜 gate 可重复变化，则只使用可观测余量/激励构造规则；参数在 D_diag 固定，局部结果不作为全程最优上界。

## 6. E4：新开发集上的确定性基线

- [ ] 运行 E4 开发集：在 D_select 跑 A0、A4、最优固定 G* 和规则 G_rule，最多 **4×20=80 回合**；未形成规则则只有 3 种。规则阈值、保持时间和初值在运行前冻结，不能查看 H_final 后调整。

预注册两个可行的探索门。所有门先要求：physical strict 和 no_fallback_physical 均无配对恶化、fallback周期不增加。

- [ ] 完成门：令 b 为 A0 的无干预失败数。b>0时，新增无干预完成数≥min(2,b)，其中物理失败恢复和fallback消除分列；b=0时此门禁用，不要求不可能的新增成功。
- [ ] 精度门：共同无干预成功至少12/20；逐配对 R 相对变化中位数≤−10%，至少2/3配对不恶化；速度 MAE 的逐对变化中位数≤0，完成时间逐对相对变化中位数≤+3%。

- [ ] 冻结 E4 决策：完成门或精度门通过只表示探索继续，不能当泛化证明；选出 G_const/D* 并保留 A4 对照。若支路活动充分但没有候选超过 A0，则停止 PPO；若活动不足，回到 E1，不通过换 seed 反复试门。

## 7. E5：只在 E4 有信号时启动 PPO pilot

- [ ] 检查 E5 入口：E0/E1 通过、支路活动可评价、E4 固定/规则候选通过探索门；局部短分支门不能绕过 E4。若固定 gate 已足够好，PPO 必须超过该强基线。

- [ ] 检查训练覆盖：用 G_const 在 T_boundary 的 12 单元检查至少 3 个余量≤0.2，其中至少 2 个物理失败或余量≤0.1；若过于容易则重新生成边界并记录预算，不能降低 15 m/s 或放宽约束。确认 T_fit/T_boundary 共 24 单元，修正脚本的 manifest 约束。

- [ ] 训练首个 PPO seed 61001：保留 4000 决策，每 500 决策收集→更新，共 8 次，每次 4 epochs、minibatch64；监控 explained variance、actor/critic 梯度、KL/clip fraction、alpha 方差和 E0 比例，记录原始/保留数量并先通过截断 bootstrap 单测。

- [ ] 在 D_select 评估 0/1000/2000/4000 四个 deterministic checkpoint，共 80 回合；只用 D_select 选 checkpoint，平局选更早版本。用所选 checkpoint 在 T_fit 跑 12 回合形成 learned-mean，再在 D_select 跑 20 回合；禁止混合 checkpoint alpha，禁止使用 H_final。

- [ ] 冻结公平协议：固定/规则/PPO 均从 alpha=(0,0) 开始，每 0.5 s 更新、slew≤0.25；A4 从 (.5,.5) 起步只作为历史附加对照。

- [ ] 判断单 seed 继续门：相对 A0 通过 E4 门；相对 D*/G_const/learned-mean 无物理或无干预回归，并新增 1 个无干预完成，或共同成功 RMSE 中位数再降≥5%且速度/时间不退化；同时排除初始化本身收益。

- [ ] 只有 61001 通过后才训练 61002/61003；每个 4000 决策、每 500 更新，总上限 12000 保留决策。每 seed 80 checkpoint 评估+12 均值采集+20 learned-mean 对照，3 seed 最多 **336 个回合**；入口能力不足时先实现，不能沿用原三批更新冒充本计划。

- [ ] 冻结 PPO 是否进入最终留出：至少 2/3 seed 通过同一继续门，全部 seed 相对 D* 无物理回归；报告全部 seed，未达成则停止扩预算。该门是工程探索门，不是显著性检验。

- [ ] 如需随机性诊断，另列 108 回合预算（3 policy seeds × 3 action seeds × 6 D_diag，以及匹配的固定均值 gate），仅区分学习与探索噪声，不参与 checkpoint 选择。

## 8. E6：一次最终留出；E7：单独实时性验证

- [ ] 锁定 H_final：固定代码/依赖、target 配置、artifact、方法、checkpoint、评价模式和阈值；模型选择结束后只打开一次 H_final，不回灌失败片段。若需重新开发，下一轮另建测试集。

- [ ] 未启动PPO时：A0、A4、D*，最多3×20=60回合。
- [ ] PPO晋级时：上述3对照+3个policy seeds的deterministic策略，最多6×20=120回合。相同配置可去重，记录实际数；每个policy seed分别与基线配对，不把60次策略运行当60个独立环境。

- [ ] 完成 H_final 判定：PPO 每个 seed 相对 A0 按 E4 门、相对 D* 不得物理/无干预回归，至少 2/3 seed 达标且全部 seed 无相对 D* 物理回归；固定/规则只按相对 A0 判定。H_final 未运行初始化、G_const、learned-mean，不能宣称超过这些开发机制对照；报告 geometry 分组、全部 seed 和按 geometry 重采样区间。

- [ ] 执行 E7 实时性测试：D_diag、batch=1、workers=1、线程各 1、固定预热，最多 7 种方法（含纯 B0）×6 单元×3 次，共 **126 回合**；分别计时控制链、仿真积分、日志序列化和初始 solve。

- [ ] 生成实时性报告：每种边界给出 p50/p95/p99/max 和 >100 ms 次数；0 超时只表示本次观察，日志开销必须计入部署测量。

## 9. 执行顺序、预算与产物

| 阶段 | 最多新增完整回合 | 其他预算 | 产出/停止点 |
| --- | ---: | --- | --- |
| E0 | 0 | 12×40周期短探针；已有12 parity只复算 | 契约不通过即停止 |
| E1 | 24 | 离线label/AR拟合 | 先冻结artifact；数据缺口补采另计 |
| E2 | 30+36=66 | 同轨迹预测误差评估 | 固定门控+预测/补偿消融 |
| E3 | 0 | 12×13=156个2s分支 | 局部机制、G*候选；无活动不解释为无效 |
| E4 | 80 | D_select几何冻结 | 任一探索门通过才考虑PPO |
| E5 | 12+336=348 | ≤12000保留决策，训练回合数依终止长度实际记录 | 首seed不过即停止后两seed；可选随机诊断+108回合 |
| E6 | 60或120 | H_final只开一次 | 最终固定/规则结论或3seed PPO结论 |
| E7 | ≤126 | 串行计时 | 与物理完成分开评估 |

E0–E4主预算最多170个完整回合+短探针/分支。走完PPO主线最多764个完整诊断/评估回合（170+348+120+126），另计训练rollout和明确记录的补采；不含可选108随机诊断。没有必要一次性跑完；每个阶段按门槛停止。按已有约200周期/回合、约50ms/周期粗估，每完整回合约10s，仅用于初步排程；以E0实测速率重新估算，不能当墙钟承诺。

- [ ] 在 `~/uuv_workspace` 核对修复源码和 CLI：运行 `python -m experiments.h15_cat_mpc.evaluate --help`、`python -m experiments.h15_cat_mpc.train_predictor --help`、`python -m experiments.h15_cat_mpc.ppo_train --help`；先做 manifest/任务清单 dry-run，确认任意 alpha、补偿开关、快照分支、独立 action seed、500 决策更新和分项计时入口，再执行仿真。

- [ ] 建立归档根目录 `outputs/h15_cat_mpc_postrepair_<run_id>/`，分为 `e0_contract/`、`e1_artifacts/`、`e2_fixed/`、`e3_branches/`、`e4_dev/`、`e5_ppo/`、`e6_holdout/`、`e7_timing/`，并交付：

- [ ] `protocol.json`：阈值、控制/环境、split与seed、schema、target快照策略、补偿模式、label契约、计时边界。
- [ ] `source_manifest.json`：Git commit、dirty diff hash、实际源码/依赖/input/checkpoint hash；只记commit且有未提交修改不足以复现。
- [ ] `pairs.csv`与逐回合JSON：期望/实际任务数、重复或缺失、退出原因、首次违约、活动覆盖、物理/fallback/deadline及逐对误差。
- [ ] `decision.md`：各阶段实际通过/停止理由，修正parity与新实验分表；不将修正前后结果合并算提升率。

- [ ] 执行顺序总检查：优先完成 E0/E1/E2，确认真实支路活动和完整固定门控表现，再决定是否投入 E3 之后的预算。
