# GS-12 下一轮：预测器 × 权重联合实验执行指令

> **状态：待执行计划。** 本文件依据当前仓库代码、`REPORT.md` 和旧路线逐项核对后编写；下文的新目录、命令及任务数是实施合同，不是已完成结果。接手 agent 应从仓库根目录执行，并以实际运行记录汇报。本文中的「C5」始终指 **C5 预测器**；真实植物始终是 C4，除非在后续独立协议中明确改变。

## 0. 本轮目标和最短路径

先回答一个问题：在同一 C4 真实植物和配对任务上，C5 预测器与已验证权重 `(1.4,0.7)` 合用，是否比单独改权重、单独改预测器更好？主要比较对象是上一轮胜出的固定权重 B，不是原始控制器 A。

按 **M0 仓库/接线审计 → M1 四配置开发集 → 冻结选择 → M2 全新锁定集 → M3 可选机理/迁移 → 结论** 执行。M1 最少 16 条、M2 如有可检验新候选则最少 12 条 64 s 闭环轨迹。先保住完整配对、质量审计和新锁定测试；算力不够时删去 M3。不要先写 FFT 规则或训练 PPO/SAC。

上一轮已证实：C4 预测器加权重 W1，在旧三条锁定任务上比 C4+W0 的任务等权平均 `J_track` 低 **13.46%**；B1/B4 经验频响均值如下。它们是**历史结果**，不是本轮 D 的结果；旧 `test_locked` 已被用于提出本轮假设，不得再次充当独立测试。

| 历史固定控制器 | B1 平均 `σ₁` | B4 平均 `σ₁` |
|---|---:|---:|
| C4 预测器 + W0 | 0.006191 | 0.207564 |
| C5 预测器 + W0 | 0.016733 | 0.143589 |
| C4 预测器 + W1 | 0.005306 | 0.189107 |

历史证据提示 C5 预测器有高频收益和低频代价，W1 在已测频带有局部收益；二者是否互补仍**未知**。`σ₁` 是参考到跟踪误差的经验频响，不是本轮时域 `J_track`，也不是鲁棒稳定性证明。

## 1. 当前仓库已核对的事实与第一步

本文件编写时，实际存在：

- `aav_mpc/condensed_mpc.py`：`CondensedActuatorMPC.set_tracking_weight_scales(depth, rudder_rate)`，深度倍率进入 `state_weight[2]`，双舵命令变化倍率进入全部 `N=35` 阶段的 `rate_weight`；`mpc.solve` 每步只做一轮 RTI。
- `experiments/gs12_weight_route/reference_tasks.py`：`single_frequency`、`mixed_reference`、`band_reference`、`task_reference`。单频参考内置前 16 s raised-cosine 启动。
- `experiments/gs12_weight_route/runner_core.py`：`run_one(task, controller, a_depth, a_rudder_rate, duration_s)`；每条轨迹新建植物/MPC，记录命令、参考、艇体状态、实际执行器、QP 状态与求解时延。它目前把 `controller == "C4"` 映射为 `(1,1)`，**所有其他字符串**映射为 `(.85,2)`；联合入口必须显式白名单，不能依赖这个隐式 `else`。
- `experiments/gs12_weight_route/run_weight_route.py`：CLI 只有 `prepare`、`pilot`、`run {dev_scan,dev_rule,test_locked,long_matrix} --workers N`；输出根由 `GS12_WEIGHT_OUT` 决定，但任务由硬编码 `task_list()` 产生。**不能**仅换输出目录就把它用作本轮新任务 runner。
- `experiments/gs12_weight_route/analyze_weight_route.py`：`main()` 固定扫描旧 `dev_scan/dev_rule/test_locked`，并写 `selection.json`；`analyze_long_matrix.py` 固定三种旧方法和旧 `output/selection.json`。二者都没有真正的 `--help` 分支，直接调用会执行分析并写文件。新分析只复用它们的评分/频响公式，不直接让其读取历史 `output/` 写回。
- `experiments/gs12_analaysis/frequency_extension_src/plant_harness.py`：C4 `PlantConfig(elevator=.9,rudder=.9,order="second_all")`、`Plant`、`diagnostic_values` 和旧闭环对照。目录拼写确实是 `analaysis`，不得自行更名。
- 旧证据在 `experiments/gs12_weight_route/output/`，含 `protocol.json`、`tasks.jsonl`、`selection.json`、`test_locked_metrics.csv`、`long_matrix/long_matrices.json`、`source_manifest.json`、`integrity_audit.json`、独立 runs/traces。把整个旧目录视为只读；不要在其中运行 `prepare`、分析、绘图或重键脚本。

仓库当前 `main` 工作树已有改动/未跟踪实验目录，不能用 `git reset`、`git clean` 或覆盖旧文件来“清理环境”。先记录 `git rev-parse HEAD`、`git status --short`、Python 与 NumPy/Torch/SciPy/OSQP/CasADi 版本和目标源码 SHA-256；必要时在新输出目录保存快照。**不要求为了本轮实验把用户现有工作树提交或改分支。**

M0 的现成只读验证命令（从仓库根目录运行）：

```bash
python3 experiments/gs12_weight_route/verify_route.py
```

它检查参考导数、QP 权重接线及旧/新 runner 的 24 s C4 基线轨迹一致性。预期旧/新命令、艇体状态、输出和诊断最大差不超过 `2e-7`，QP 有效标志逐周期一致。若失败，先排查环境/源码差异，**不得**开始新对比或把接线失败解释成控制器负结果。

## 2. 精确实验矩阵与共用协议

真实植物对四配置相同：`PlantConfig(elevator=.9,rudder=.9,order="second_all")`，15 m/s；MPC 22 状态、`dt=.1 s`、`N=35`；每控制周期植物推进 `5×.02 s`。初态、配平、命令盒约束、OSQP 设置、warm start、参考和诊断边界均沿用旧 runner。只改变预测器参数和两个权重倍率。

| ID | plant_id | predictor_id | `fin_gain_scale` | `fin_tau_scale` | weight_id | `(a_D,a_Δδ)` |
|---|---|---|---:|---:|---|---|
| A | C4 | P0/C4 | 1.0 | 1.0 | W0 | `(1.0,1.0)` |
| B | C4 | P0/C4 | 1.0 | 1.0 | W1 | `(1.4,0.7)` |
| C | C4 | P1/C5 | 0.85 | 2.0 | W0 | `(1.0,1.0)` |
| **D** | **C4** | **P1/C5** | **0.85** | **2.0** | **W1** | **`(1.4,0.7)`** |

`prepare` 同时在协议的 `optional_dev_candidates` 中预登记以下**暂不运行**的补充点，避免看过 dev 后改协议/源码：E=`Pmid(.925,1.5)+W1(1.4,.7)`，E0=`Pmid+W0(1,1)`，F=`P1+(1.4,1)`，G=`P1+(1,.7)`，H=`P1+(.7,1)`。P1 五点十字扫描的另两点正是已运行的 C、D。补充批次只能选择 `pmid`（E，必要时 E0）或 `p1_weight_cross`（F/G/H），最多一类；所有补充点仍配对四条 dev 任务。未选择时它们不增加 M1 的 16 条必跑轨迹。

权重作用点是深度状态权重 `3*a_D`、每个预测阶段双舵归一化命令变化惩罚 `0.35*a_Δδ`；其他权重和硬约束不改。`mpc.dynamics.fin_gain_scale/fin_tau_scale` 是预测器核验对象；`mpc.alpha` 默认零，`solve()` 采用 dynamics 的 base 参数。D 必须同时被验证为 C5 预测器和 W1 的 QP 权重，不能只检查 JSON 名称。

所有任务采用 `[N,E,D,pitch,yaw,VN,VE,VD]` 运动学一致参考。主评价窗口为 `32≤t≤64 s`，与旧 `analyze_weight_route.summarize()` 对 0.02 s 子步记录的包含两端点口径一致；每任务四配置使用完全相同的相位、幅值和窗口。外部主指标固定为

\[
J_{\rm track}=\operatorname{mean}_{32\le t\le64}\left[
((E_r-E)/8)^2+((D_r-D)/5)^2+((V_{N,r}-V_N)/2)^2\right].
\]

不可比较随 W1 改变的 MPC 内部优化代价。主汇总为**先对每任务算 J，再对任务等权平均，最后求比率**；逐任务百分比另列，不能把百分比算术平均当作正式汇总。三个误差通道的工程单位 RMSE 及峰值必须同时报告。

## 3. 一次性冻结的任务清单

下表给出**本轮指定值**，不是留给 agent 临场挑选的示例。新目录的 `prepare` 必须在 M1 运行前一次性把 dev 和 new_locked 全部写进 `protocol.json`、`tasks.jsonl`，计算哈希并锁定。`new_locked` 任务内容在读取 dev 结果后不得变更。所有任务 `duration_s=64`、`speed_mps=15`、`amplitude=.006`、`window_s=[32,64]`，相位单位 rad；所有单频均有旧公式中的前 16 s 平滑启动。

| split | task_id | kind / axis | `k` 或切换中心 s | `phase_rad` |
|---|---|---|---|---:|
| dev | `dev_depth_b1` | single / depth | `k=8` | 0.43 |
| dev | `dev_depth_b4` | single / depth | `k=32` | -0.61 |
| dev | `dev_east_b4` | single / east | `k=28` | 1.37 |
| dev | `dev_mixed_switch` | mixed / east_depth | `[21,41]` | -0.42 |
| new_locked | `locked_depth_b1` | single / depth | `k=4` | 1.83 |
| new_locked | `locked_east_b4` | single / east | `k=26` | -0.94 |
| new_locked | `locked_mixed_switch` | mixed / east_depth | `[24,44]` | 2.27 |

单频使用现有 `single_frequency()`；`ω_k=2πk/64`，幅值 `.006`。混合任务沿用现有 `mixed_reference()` 的低频 `k=8`、高频 `k=32`、高频相位偏移 `+.37`、两次宽度 2 s 的 cubic smoothstep，但旧实现**固定幅值 `.006` 且在 t=0 没有 16 s 启动包络**。为满足本轮协议，只在新目录增加 `joint_reference.py` 适配器，按以下确定公式处理旧函数返回的东向/深度偏离直航值；不要改变历史 `reference_tasks.py` 的默认语义：

- 令 `s=amplitude/.006`；`g(t)=0` (`t≤0`)，`[1-cos(πt/16)]/2` (`0<t<16`)，`1` (`t≥16`)；`g'` 为解析导数。
- 对旧混合参考 `E_0,D_0,VE_0,VD_0`，令 `E=s*g*E_0`、`D=20+s*g*(D_0-20)`、`VE=s*(g*VE_0+g'*E_0)`、`VD=s*(g*VD_0+g'*(D_0-20))`；`N=v*t,VN=v`；由新速度重新计算 pitch/yaw。
- 单频仍直接调用现有 `single_frequency(t,k,axis,phase,amplitude,speed)`；每条任务在进入 runner 前检查 `d[N,E,D]/dt≈[VN,VE,VD]`，包含 16 s 启动与混合切换邻域。使用密网格中心差分，最大误差目标 `<1e-3`（排除差分端点）；若不满足，先修参考。

这七条任务的相位/频点/切换时刻与旧 `test_locked` 不同。`dev_depth_b1/b4` 虽复用频点 `k=8/32`，也不是独立锁定证据。若因实现缺陷必须改任务公式或阈值，**在任何 dev 运行前**改完、重新冻结协议；dev 开始后只能另立探索性增补协议，不得暗改本轮 new_locked。

## 4. 最小实现：复用旧闭环，不复制求解器

新建 `experiments/gs12_joint_route/`：`joint_reference.py`、`run_joint.py`、`analyze_joint.py`、`README.md` 和独立 `output/`。任务运行与分析必须从新目录读取协议，不调用旧 `run_weight_route.py.task_list()`、`jobs_for()` 或旧分析器的 `main()`。

1. 给 `gs12_weight_route/runner_core.py:run_one` 增加可选的 `reference_fn=None, predictor_params=None` 参数。默认参考仍是 `task_reference`、默认预测器仍按旧 C4/C5 映射，旧调用的数值行为不得变化；传入参数时分别用它生成 `ref_grid` 和构造 `ActuatorAwareCpuDynamics(.1,*predictor_params)`，并校验预测器参数为有限正数。`joint_reference.py` 只为新 mixed 任务提供上节适配器，其余委托旧函数。联合实验总是显式传入预测器数值；以后若做 `Pmid`，复用这个入口。若可用更薄的调用注入而不改旧 runner，也可采用，但须同样通过轨迹对齐。不得重写 MPC、植物或控制循环。
2. `run_joint.py` 从本轮 `tasks.jsonl` 生成四配置作业。每个 worker 调用 `runner_core.run_one`，每条轨迹独立初始化；`predictor_id` 只允许控制器矩阵中显式列出的 P0/P1（可选补充时加 Pmid），拒绝未知 ID。统一从 A/B/C/D 定义推导 `predictor_id/model_gain/model_tau/weight_id/a_depth/a_rudder_rate`，不要在多处复制手写映射或依赖旧 runner 对非 C4 字符串的隐式分支。
3. `run_one` 的原始 arrays 已有 `time_substep_s,state_substep,output_substep,reference_substep,diagnostic_substep,actual_substep,command_substep,time_control_s,command,qp_ok,qp_primal,qp_dual,a_depth,a_rudder_rate,solve_ms`；本轮继续每条保存独立 `.npz`。新 JSON 记录协议/任务/源码哈希、Git commit、配置数值、状态/首次越界、QP、饱和、p50/p95/p99 和 trace SHA-256。评分可以在新分析器从 NPZ 重算，勿在多个地方维护不同的 `J_track` 公式。
4. `analyze_joint.py` 读取**预期任务 × 预期配置**左连接 run JSON 与 NPZ；检查 JSON 与 trace SHA、协议哈希、配置数值及时间长度。缺失、重复、校验失败、早停、QP 失败均单列，并拒绝给不完整四配置配对下晋级结论。复用旧 `analyze_weight_route.summarize` 的误差列/尺度思路，但本轮窗口从任务协议读取，不使用它写死的旧阶段和输出路径。
5. 主进程先检查每批 `run_id` 唯一，再用 `ProcessPoolExecutor` 按轨迹并行；worker 只写自己 `run_id` 的 JSON/NPZ，主进程串行写 manifest。轨迹时间步不能并行，须保留 MPC warm start。数值库导入前设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；先用 1 worker 验证，再按本机内存与旧 pilot 经验在 1/2/4 中选（旧 4 worker 的 20.90 s 仅为历史吞吐）。不做无必要的并行框架改造。

新目录至少使用以下**待实现 CLI 合同**；实现后在 `README.md` 写实际运行命令并确认 `--help`，不能把这段预期接口说成现成入口：

```bash
python3 experiments/gs12_joint_route/run_joint.py verify --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/run_joint.py prepare --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/run_joint.py run dev --workers 4 --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/analyze_joint.py --stage dev --out experiments/gs12_joint_route/output
# 仅在第 6 节相应分支触发时，择一运行补充开发批次并重算 dev 分析：
# python3 experiments/gs12_joint_route/run_joint.py run dev-extra --set pmid --workers 4 --out experiments/gs12_joint_route/output
# 若 E 的机理判断需要配对 W0，再运行：
# python3 experiments/gs12_joint_route/run_joint.py run dev-extra --set pmid_w0 --workers 4 --out experiments/gs12_joint_route/output
# python3 experiments/gs12_joint_route/run_joint.py run dev-extra --set p1_weight_cross --workers 4 --out experiments/gs12_joint_route/output
# python3 experiments/gs12_joint_route/analyze_joint.py --stage dev --out experiments/gs12_joint_route/output
# 在 output/decisions.md 写明 dev 决策；以下以选中 D 为例：
python3 experiments/gs12_joint_route/run_joint.py freeze --candidate D --out experiments/gs12_joint_route/output
# 冻结候选后，才允许下面两条：
python3 experiments/gs12_joint_route/run_joint.py run new_locked --workers 4 --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/analyze_joint.py --stage new_locked --out experiments/gs12_joint_route/output
```

`verify` 使用代码中的预定任务定义生成短时 smoke，不依赖已冻结协议、不读取 new_locked 的结果，也不生成正式评分；它在单独 `smoke/` 下保存诊断。验证通过后才运行一次 `prepare`。`prepare` 若目标 `protocol.json` 已存在应报错或只读显示哈希，**不得重写**（当前旧 `prepare` 会重写协议并改变时间戳，不能照搬）。`dev-extra` 仅从预登记补充点生成作业：`pmid` 为 E，`pmid_w0` 为 E0，`p1_weight_cross` 为 F/G/H；重算 dev 分析时仍核对原 16 条的完整性，未触发分支不得运行。`freeze` 检查 `decisions.md`、dev 四配置覆盖、分析文件及哈希，写入候选及其参数；若选 E 或其他补充候选，`--candidate` 使用实际已注册 ID。`run new_locked` 必须要求 `selection.json` 已冻结并含所选候选及 dev 决策哈希。D 若被淘汰，按第 6 节决定是否还有候选；没有候选就形成 M1 负结果，不伪造 M2 晋级。

### M0 新接线验收

- 先运行现成 `verify_route.py`。新 runner 的 A 用旧 `verify_route.parity_check` 同一 24 s 参考做对齐；默认参考注入前后状态、命令、QP 标志应满足原脚本 `2e-7` 门槛。
- 用同一初态、参考和 warm start 构造 A/B/C/D；检查 `mpc.dynamics.fin_gain_scale/fin_tau_scale` 分别为 `(1,1)/(1,1)/(.85,2)/(.85,2)`，`depth_weight_scale/rudder_rate_weight_scale` 分别为 W0/W1/W0/W1。用现有 `verify_route.qp_check` 的 OSQP setup 捕获方法，在 P0 内比较 A/B、P1 内比较 C/D 的 QP 目标：深度倍率应改变 Hessian 和 gradient，舵变化倍率应改变 Hessian；零舵计划下相应 gradient 不变是预期行为。检查 A/C 的预测器参数与模型线性化输出不同。D 的短轨迹至少走过一次真实 `solve()`，不能只检查构造字典。
- `verify` 若发现 C/D 被错误映射、参考不连续或 A 不对齐，先修复并重跑验证；修复前不得进入 M1。正式协议和源码哈希在验证通过后冻结。

## 5. M1：四配置开发集与分析

按第 3 节四条 dev 任务运行 A/B/C/D，共 **16 条 64 s**；若计算时间紧，可先跑两条深度任务的 8 条并输出中间表，但正式开发结论仍须补齐东向和混合任务。每任务四配置的真实植物、参考、初态与评价窗口完全配对。

逐任务计算 `J_00`(A)、`J_01`(B)、`J_10`(C)、`J_11`(D)，保存：

- W1 在 P0 的效果 `J_01-J_00`；P1 在 W0 的效果 `J_10-J_00`；W1 在 P1 的效果 `J_11-J_10`；交互 `I=J_11-J_10-J_01+J_00`。`I<0` 仅表示有利的代价交互，**不等于** D 最优。另列 `J_D-min(J_A,J_B,J_C)` 与 D/B 比值。
- `RMSE_E[m]、RMSE_D[m]、RMSE_VN[m/s]`、三通道最大绝对误差，QP 失败、完成状态、首次边界触发、命令与实际执行器饱和率、最大归一化命令变化、实际舵面变化、求解 p50/p95/p99。旧 runner 的诊断边界为侧向 8 m、深度 5 m、切向速度 2 m/s、pitch 40°；饱和判据为归一化命令/实际执行器距盒界 `≤1e-6`。新分析沿用此口径。
- 提前终止时，先报告完成率和首次越界时间；跟踪误差只能在该任务所有配置的共同前缀比较，并单列完整轨迹的全窗口 `J_track`。**不允许**把不足 64 s 的短轨迹混入正式排名。

为使「明显命令尖峰」可执行，预先定义审查标志：同一任务候选的最大归一化单步命令变化同时 `>0.05` 且 `>1.25×B` 时标记；报告实际数值和曲线。它是本轮工程审查阈值，不是已验证的舵机安全极限。候选有任何新增 QP 失败、诊断越界或命令/实际舵面饱和样本，即不得自动晋级；先查明原因。历史 B 在旧锁定集均无此类事件，本轮仍须重测 B。

最低输出：`output/dev_metrics.csv`（16 行）、`output/dev_quality.csv`、`output/factor_effects.csv`（每任务一行及任务等权汇总）、`output/figures/dev_factor_grid.png`、`dev_interaction.png`、`dev_mixed_trace.png`。每张图应注明 `dev`、任务 ID、控制器 ID、评分窗口和基线；单频时域任务不得标作完整 B1/B4 频响。

## 6. M1 后冻结候选，M2 新锁定测试

先将实测判断写入 `output/decisions.md`，再写 `selection.json`：候选 ID/参数、依据的 dev run IDs 和分析 SHA、协议 SHA、冻结时间、是否观察到约束或通道损失。不得根据 new_locked 结果选参。

**预先定义的决策顺序：**

1. 若 D 相对 B 的 dev 任务等权平均 `J` 更低、至少 3/4 任务不恶化，且无第 5 节质量阻断，冻结 D，直接 M2，不做额外搜索。
2. 若 D 高频更好而低频/混合更差，先检查各通道和约束。可执行 `dev-extra --set pmid`，在四条 dev 任务上增加 E=`Pmid(.925,1.5)+W1` 共 4 条；必要时再加 E0=`Pmid+W0` 共 4 条，且该类补充仍只算一轮。新增运行仍是 dev，不能读取 new_locked 来选择。冻结 dev 上最有价值且无风险的一个新候选进入 M2；若只有 B/C 各有胜场而没有新候选，则记录上下文取舍，停止本轮晋级测试。
3. 若 D 无优势，但 C 与 B 各有胜场，最多执行一次 `dev-extra --set p1_weight_cross`：在 P1 下补跑 F/G/H 三种权重×四条 dev 任务，共 12 条；与已有 C/D 一起形成五点扫描。仅用 dev 选一个新候选。若已经做过中点补充，不再同时扩此扫描。补充扫描是探索性开发，须留完整 run IDs 和选择成本。
4. 若 D 和最多一轮补充均无稳健收益，也无可解释取舍，停止联合搜索，报告历史 B 收益和本轮负结果；不训练 RL。

M2 只运行**第 3 节已冻结**的三条 `new_locked` 任务，至少 A/B/C/冻结候选四种配置，共 **12 条 64 s**；新候选若是 E，用 E 替代 D 作为第四配置，仍保留 A/B/C。分析器检查每任务四配置齐全、哈希正确且全部完成后，才给正式晋级判断。主要比较 `candidate vs B`；A 解释相对原始 MPC 的累计收益，C 解释预测器单独效果。

预注册严格通过门槛：`100×[1-mean_task(J_candidate)/mean_task(J_B)] ≥5%`；三条任务各自 `J_candidate≤J_B`；候选与 B 均完成、候选无新增 QP 失败/诊断越界/饱和样本/第 5 节命令尖峰标志。若整体改善 0–5% 或有任务轻微退化，写「正向趋势但未晋级」或「任务取舍」；5% 是工程门槛，不是统计显著性。若某任务 J 极小，同时报绝对 J、工程单位 RMSE、逐任务百分比和等权绝对代价汇总，避免比例误导。

打开 new_locked 后，不得再改候选并在同三条任务上反复测试、再称之为独立留出。任何阈值/任务/方法事后变化均标为探索性，下一轮另建未使用的锁定集。旧 `test_locked` 也不能复用。

## 7. M3 可选：机理复核和工况迁移

只有 M1/M2 提供值得复核的候选或清晰上下文取舍，且时间允许时执行。先选最能回答当前问题的一种，不要求两种都做。

**B1/B4 长协议。** 复用旧 `reference_tasks.band_reference`、`analyze_long_matrix.matrix_for` 的计算口径：`k=1…8`/`25…32`，三输入轴 east/depth/speed，幅值每频线 `.006/.003`，256 s，稳态窗口 W2=`128–192`、W3=`192–256 s`，每控制器有自己的零激励基线。D 的增量为 `2×3×2+1=13` 条 256 s（准确复核基线共享）。旧 `analyze_long_matrix.py` 的输出路径和三方法标签写死；新目录只提取 `matrix_for` 公式或增加通用参数，不对旧输出调用它。只有核对真实植物、源码/参考、窗口、幅值、求解设置和零基线后，才把旧 A/B/C 矩阵作为历史参照；不兼容时在新协议下四配置全部重跑。比较 `σ₁(ω)`、B1/B4 均值/峰值、固定物理方向、W2/W3 相对差、全/半幅矩阵差、受约束标志和未激励频线能量。经验 FRF 不提供稳定性证明。

**短协议迁移。** 用 12、18 m/s 和幅值 `.006/.012`，优先深度高频与混合任务；每个新工况仍配对 A/B/C/冻结候选。变航速时 `Plant(speed, config)` 会按速度重新配平，参考 `VN=speed`；不得复用 15 m/s 初态或配平。`.012` 是拟议加倍幅值，不可当成已验证局部线性范围；报告约束、饱和、QP、各通道误差和首次越界。只有控制器排名随航速/幅值/任务发生可复核的交叉，才提出上下文调度问题。

**后续频谱规则/RL 的入口。** 本轮四配置均为静态。若确有可靠交叉，先用部署可得的航速、参考预览变化率、执行器余量设计显式规则；需要频谱时须明确未来参考可预览长度，不能读取未来真实植物状态或直接用实验频带标签。动态规则与最优固定配置比较，不能把两套固定 LTI 频响拼成动态策略 FRF。只有规则在独立任务上有可重复的独立收益且仍有清晰局限，才考虑复用 `experiments/gs12_gate/` 研究 RL；旧 PPO `dev_quick` 没通过 5% 晋级，不是扩训理由。

## 8. 文件、完整性和最终报告

新输出至少包含：

```text
experiments/gs12_joint_route/
  README.md                   # 已实现、实际验证过的命令
  joint_reference.py
  run_joint.py
  analyze_joint.py
  output/
    protocol.json            # dev/new_locked 全任务及门槛，在 M1 前一次冻结
    tasks.jsonl
    controller_matrix.json
    selection.json           # M1 后冻结；未选择时明确写无候选
    decisions.md
    manifest.json
    runs/<run_id>.json
    traces/<run_id>.npz
    dev_metrics.csv
    dev_quality.csv
    factor_effects.csv
    new_locked_metrics.csv    # 仅在 M2 实际执行后生成
    integrity_audit.json
    REPORT.md
    WEEKLY_MEETING_SUMMARY.md
    figures/
```

`run_id` 由协议 SHA、split、完整 task JSON 的哈希、predictor ID/数值、weight ID/数值、航速、幅值与相位确定；16 位十六进制哈希加唯一性断言即可，不为极小概率碰撞引入数据库。JSON 至少记录 `run_id,protocol_sha256,task_sha256,git_commit,source_sha256,split,task_id,phase_rad,amplitude,speed_mps,plant_id,predictor_id,model_gain,model_tau,weight_id,a_depth,a_rudder_rate,completion_status,first_violation_time_s,qp_failures,command_saturation,actual_saturation,solve_p50_ms,solve_p95_ms,solve_p99_ms,wall_s,trace_sha256`。分析表再附窗口、`J_track`、RMSE 和峰值。若 source hash 与冻结协议不同，停止运行并说明差异，不混合版本。

完整性审计必须给出计划/实际 run 数、每任务方法覆盖、缺项、重复、SHA 不符、早停、QP 失败、越界、饱和计数；失败记录保留，不静默过滤。`REPORT.md` 明确分列「历史已证实」「本轮开发集」「本轮新锁定集」「可选扩展/未运行」，附绝对数值、逐任务和逐通道结果、图、协议 SHA、Git commit 与限制。周会摘要须把历史 **13.46%** 和本轮新测值分开。确定性任务、频线和时间步不是独立随机样本，不报告无依据的置信区间、实艇失效概率或全局稳定结论。

每完成 M0/M1/M2，向用户报告实际阶段、运行/完整/缺失/QP/越界/饱和数、最重要的相对 B 数值、新文件路径及下一节点理由。负结果也按相同标准交付。

## 9. 范围与复杂度准则

本轮不重写求解器/植物，不新建通用配置框架、数据库、分布式锁或安全防护层，不重训 PPO/SAC，不默认重跑全部 32 频点，不把 10 状态实验数字横向混排。只有**已经观察到**的具体失败模式影响本轮配对、数据完整性或结论时，才做满足该失败模式的最小修复。

**新增复杂度前的硬规则：**对于推测性或发生概率极低的并发竞争、安全边界问题，先向用户说明具体失败模式、发生可能性、实现及维护成本，并取得同意；不得悄悄扩展范围。当前已知且需要直接处理的竞争只有：多个 worker 同时写共享 `manifest.json` 会丢记录、重复 `run_id` 会覆盖轨迹。前者由主进程单独写 manifest，后者由批次开始前唯一性断言解决；不据此建设通用事务系统。这个规则不妨碍修复实际出现的运行错误，也不要求为常规可逆的最小代码修复另行请求许可。

本轮成功的标准是：用完整、配对、可复查的数据回答模型修正与权重是否互补，以及是否存在值得后续研究的工况取舍；**不以必须得到正向数值为成功条件**。
