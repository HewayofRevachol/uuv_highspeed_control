# GS-12 下一轮 Agent 实验指示：连续预测器扫描与条件启动的频谱规则

> **状态：待执行协议。** 本文件依据当前仓库的联合实验 `output/REPORT.md`、冻结协议、原始轨迹和实际 MPC/runner 源码编写。这里给出的新任务、命令和门槛是下一轮实施要求，**不是已经取得的新结果**。执行时从仓库根目录运行命令；新结果只写入 `experiments/gs12_predictor_route/`，不得覆盖本文件所在的旧 `gs12_joint_route/output/`。
>
> **核心问题：** 固定 W1 权重后，沿 P0→P1 的一维预测器参数路径，是否存在兼顾低频深度与高频/混合任务的固定 β？只有静态扫描仍显示可靠、可从部署时可得参考识别的取舍，才试验显式频谱调度。**本轮不训练 RL。**

## 0. 已有结果、边界与执行顺序

旧固定权重实验：C4 预测器 + W1=`(a_D,a_Δδ)=(1.4,0.7)`，在当时三条锁定任务上相对 C4+W0 的任务等权平均 `J_track` 降低 **13.46%**。最近完成的联合实验：A=P0W0、B=P0W1、C=P1W0、D=P1W1；**四者面对同一个 C4 真实植物**。联合实验的新锁定集 12/12 条完整，QP 失败、诊断越界、命令/实际执行器饱和均为零。D 相对 B 的三任务等权平均 J 下降 **75.82%**，但低频深度任务从 B 的 `2.5498e-10` 增至 D 的 `2.1751e-9`；仅 2/3 任务不劣，**严格门槛未通过**。高频东向和混合任务分别改善 78.87% 和 74.74%。开发集同样见到低频深度退化；因子交互均值 `I=+7.7583e-8`，不得称已证实“模型×权重正向协同”。

这些数值来自 [`gs12_joint_route/output/REPORT.md`](REPORT.md) 和 [`integrity_audit.json`](integrity_audit.json)，只作本轮假设动机。旧固定权重 `test_locked`、联合实验的 `dev/new_locked` **都已被查看**，不能重新当独立留出任务。本轮不重新判 D 上轮是否晋级。

执行顺序和交付门：

| 节点 | 必须做什么 | 后续条件 |
|---|---|---|
| M0 | 核对源码/旧证据，完成 β=0/1 对 B/D 的同轨迹接线验证；冻结全部新任务 | 接线失败先修，不能把错误当负结果 |
| M1 | W1 固定，5 个 β × 4 条新开发任务，共 20 条 64 s 配对轨迹 | 产出性能—质量曲线和明确 G1 决策 |
| G1 | 按预注册规则选最多一个固定中间 β，或记录端点取舍 | 无静态候选不强行运行静态锁定集 |
| M2 | 若选出中间 β，用全新 4 条锁定任务检验，至少 12 条轨迹 | 锁定失败不能在同批任务上改 β 重试 |
| M3 | 仅在取舍仍可靠、16 s 参考预览可得且模型参数逐步注入验证通过时，做显式频谱规则和另一套独立锁定集 | 不满足条件则停止动态分支 |
| M4 | 可选航速/幅值/长频响迁移 | 不得挤占 M1/M2 必需配对与审计 |

完成 M1、以及实际触发的 M2/M3 后，分别向用户汇报运行/完整/缺失/失效数、相对哪个固定基线的数值、输出路径和下一门的理由。负结果也要形成完整报告。

## 1. 当前仓库核实过的接口：直接复用

- `experiments/gs12_joint_route/run_joint.py` 现有命令是 `verify`、`prepare`、`run {dev,dev-extra,new_locked}`、`freeze`；它的 `TASKS` 写死为**上一轮七条任务**，不能靠改 `--out` 直接运行本轮任务。`gs12_joint_route/analyze_joint.py` 也只识别旧 `dev/new_locked`，不能直接用于新 split。
- `experiments/gs12_joint_route/joint_reference.py:joint_task_reference` 对单频委托旧 `reference_tasks.py`，对混合参考加 16 s raised-cosine 启动；正确处理位置与速度导数。新任务继续复用该函数，参考公式不得随 β 改变。`derivative_error(task)` 可作密网格位置/速度一致性检查。
- `experiments/gs12_weight_route/runner_core.py:run_one` 目前支持静态 `predictor_params=(gain,tau)` 和可选 `reference_fn`，但每步调用 `mpc.solve(state, refs)`，**还没有 β_t 策略钩子和 β 轨迹日志**。如需新增，保持默认调用行为原样，以免改变旧证据的解释。
- `aav_mpc/condensed_mpc.py:CondensedActuatorMPC.solve(state,reference,alpha=None,theta_hat=None,d_hat=None)` 已有连续模型参数入口。`base_theta=[self.dynamics.fin_gain_scale,self.dynamics.fin_tau_scale]`，实际每步参数为 `base_theta + alpha[0]*(theta_hat-base_theta)`；`alpha[1]` 只控制残差预测。本轮设基础预测器为 P0 `(1,1)`、`theta_hat=[.85,2]`、`alpha=[β,0]`、`d_hat=None`，于是实际 gain=`1-.15β`、tau=`1+β`。该参数送入 `_one()` 的 `self.dynamics.evaluate(...,model_parameters=...)`；`simulator/rti_symbolic.py` 的 CasADi 函数把它用于双舵效力和双舵时间常数。**无需每步重建 `ActuatorAwareCpuDynamics`、OSQP 或 MPC 对象。**
- 现有 `run_one` 在构造 MPC 后设置了 `mpc.dynamics`；若使用本轮每步 α 路径，明确设置其 base 为 P0，所有轨迹 W1 保持 `(1.4,0.7)`。`mpc.previous`、`mpc.plan`、植物实际执行器和延迟队列应随时间连续，不因 β 更新而重置。
- 当前工作树已有未提交/未跟踪的实验材料，Git HEAD 本身不足以标识运行代码。新协议记录 Git commit、工作树状态和影响实验的源码 SHA-256。仓库中 `gs12_analaysis` 的拼写是既有路径，不改名。

现成的初始验证入口（**只读检查**，不写旧输出）：

```bash
python3 experiments/gs12_weight_route/verify_route.py
python3 experiments/gs12_joint_route/run_joint.py --help
python3 experiments/gs12_joint_route/analyze_joint.py --help
```

不要对旧 `analyze_weight_route.py`、`analyze_long_matrix.py` 或 `plot_results.py` 试 `--help`：这些脚本没有完整的 help 分支，直接运行可能重写旧输出。旧 `gs12_joint_route/output/`、`gs12_weight_route/output/` 与其 runs/traces 均视为只读。

## 2. 固定控制对象、主指标和安全口径

所有 M1/M2/M3 配对任务固定 C4 真实植物 `PlantConfig(elevator=.9,rudder=.9,order="second_all")`、15 m/s、同一初态/速度配平、22 状态 MPC、`dt=.1 s`、`N=35`、每周期 `5×.02 s` 植物子步、相同盒约束/OSQP 参数/warm start。只改变 β；W1 全程为 `a_D=1.4,a_Δδ=.7`，其中深度权重 `3a_D`、双舵逐预测阶段命令变化惩罚 `.35a_Δδ`，其他代价与约束不变。

\[
\beta\in\{0,.25,.5,.75,1\},\quad
g(\beta)=1-.15\beta,\quad \tau(\beta)=1+\beta.
\]

| β | gain | tau | 含义 |
|---:|---:|---:|---|
| 0 | 1.0000 | 1.00 | B=P0W1 |
| .25 | .9625 | 1.25 | 新中间点 |
| .50 | .9250 | 1.50 | 新中间点；虽在旧协议预登记为 E，**旧实验未运行它** |
| .75 | .8875 | 1.75 | 新中间点 |
| 1 | .8500 | 2.00 | D=P1W1 |

这是模型参数路径，不是两个闭环频响矩阵的线性插值；gain/tau 同时变化，不能从 β 曲线单独归因两者贡献。

沿用前两轮同一**外部**追踪代价，所有配置在相同任务的同一 `32≤t≤64 s` 子步窗口评分：

\[
J_{\rm track}=\operatorname{mean}\left[
((E_r-E)/8)^2+((D_r-D)/5)^2+((V_{N,r}-V_N)/2)^2\right].
\]

单任务报告原始 J、相对 B 和 D 的百分比及绝对 `ΔJ`；总指标是**先求各任务 J，再等任务权重取均值，最后算比率**，不能平均各任务百分比。另报 `RMSE_E[m]、RMSE_D[m]、RMSE_VN[m/s]`、三通道峰值和归一化 MSE 分量。低频深度 J 可能在 `10^-10` 量级，必须并列绝对 J、深度 RMSE 和相对变化；不能因百分比大而夸大工程意义，也不能因绝对误差小而事后改严格门槛。MPC 内部随预测器/权重改变的优化目标不能跨配置评分。

每条轨迹记录完整/早停、首次边界触发时刻和种类、QP 失败数、命令/实际执行器饱和、最大归一化单步命令变化、最大实际舵面单步变化、`mpc.solve` p50/p95/p99、wall time、参考/输出/10 维艇体状态/执行器/命令/β 与实际 `(gain,tau)`。沿用旧 runner 的边界：侧向 8 m、深度 5 m、切向速度 2 m/s、pitch 40°；饱和按归一化盒界 `1e-6` 判断。命令尖峰预注册为：同任务候选最大归一化单步变化**同时** `>0.05` 且 `>1.25×B`。任何新增严重 QP 失败、边界触发或饱和样本均阻断自动晋级；完整率和首次触发时间先于 J 排名。早停时只在该任务所有配置的共同前缀对比误差，不把短轨迹与完整全窗 J 混排。

## 3. 新任务：执行前一次冻结全部 split

下表为**本轮具体参数**，不再由执行 agent 看过 M1 结果后挑选。全部任务 64 s、15 m/s、归一化幅值 `.006`、`window_s=[32,64]`，单频 `ω_k=2πk/64`，前 16 s 平滑启动；`phase_rad` 单位 rad。混合任务沿用 `joint_task_reference` 的低 `k=8`、高 `k=32`、高分量相位偏移 `+.37`、两次宽度 2 s 的平滑切换。四组任务的 phase/频点/切换中心各不相同，并区别于旧联合实验任务。参考仍为 `[N,E,D,pitch,yaw,VN,VE,VD]`，位置导数必须与速度一致。

| split | task_id | 参考 | k / 切换中心 s | phase_rad |
|---|---|---|---|---:|
| `dev_static` | `ps_dev_depth_low` | single depth | k=3 | .58 |
| `dev_static` | `ps_dev_depth_high` | single depth | k=29 | -.76 |
| `dev_static` | `ps_dev_east_high` | single east | k=27 | 1.49 |
| `dev_static` | `ps_dev_mixed` | mixed east_depth | [19,43] | -1.24 |
| `test_static_locked` | `ps_lock_depth_low` | single depth | k=5 | 2.02 |
| `test_static_locked` | `ps_lock_depth_high` | single depth | k=31 | .37 |
| `test_static_locked` | `ps_lock_east_high` | single east | k=25 | -1.12 |
| `test_static_locked` | `ps_lock_mixed` | mixed east_depth | [23,45] | .92 |
| `dev_dynamic` | `pd_dev_depth_low` | single depth | k=2 | -.93 |
| `dev_dynamic` | `pd_dev_depth_high` | single depth | k=30 | 1.04 |
| `dev_dynamic` | `pd_dev_east_high` | single east | k=26 | -1.56 |
| `dev_dynamic` | `pd_dev_mixed_a` | mixed east_depth | [26,46] | 1.72 |
| `dev_dynamic` | `pd_dev_mixed_b` | mixed east_depth | [18,38] | -2.31 |
| `test_dynamic_locked` | `pd_lock_depth_low` | single depth | k=7 | .34 |
| `test_dynamic_locked` | `pd_lock_depth_high` | single depth | k=32 | 2.19 |
| `test_dynamic_locked` | `pd_lock_east_high` | single east | k=28 | -.24 |
| `test_dynamic_locked` | `pd_lock_mixed_a` | mixed east_depth | [27,47] | -1.87 |
| `test_dynamic_locked` | `pd_lock_mixed_b` | mixed east_depth | [17,37] | .15 |

`prepare` 一次写入上述 18 条任务、评分、质量门槛、β 集合、未来预览可用假设和源码哈希，并给 `protocol.json/tasks.jsonl` 计算 SHA-256；已有文件必须拒绝重写。任务预冻结不表示必须运行所有 split：M2/M3 仍受决策门约束。静态锁定与动态锁定是两套不同任务，互不用于对方选参。若发现参考或协议技术错误，**在首次 M1 运行前**修复并重新冻结；M1 开始后变更只能另起版本，原 run 不混入新协议。

频谱规则若启动，计划参考需在任务终点后继续按**同一已冻结解析公式**生成至 80 s，供最后时刻的 16 s 预览使用；这表示仿真明确假设部署端能获得同样的未来计划，不得把仿真生成器的未来真实植物状态交给规则。若现实部署端只能拿到 MPC 自带的 3.5 s 参考片段，停止 16 s FFT 分支，另立短预览协议；不能把 16 s 信息冒称为现有 MPC 接口自带。

## 4. 最小代码修改、接线验收与拟实现 CLI

在新目录 `experiments/gs12_predictor_route/` 写薄封装 `run_predictor.py`、`analyze_predictor.py`、`README.md`；只有触发 M3 才新增 `spectral_policy.py` 和 `dynamic_protocol.json`。复用 `joint_task_reference`、`Plant`、`diagnostic_values` 和现有 `runner_core.run_one` 的控制循环，不复制植物或重写 QP。给 `run_one` 增加默认关闭的 `beta_policy`/预览长度参数：每周期先从只读参考计划取 β，再调用 `mpc.solve(...,alpha=[β,0],theta_hat=[.85,2.0],d_hat=None)`；静态策略每步返回常数。策略启用时将参考预生成网格延长至 `duration_s+16 s`，MPC 仍只切取原来的 35 个未来 .1 s 点；默认不传策略时，旧 `run_one` 的构造与调用路径完全不变。把 `beta_t,model_gain_t,model_tau_t` 作为新 NPZ 字段；所有 β 方法使用同一 P0 base dynamics，W1 常数，不改 warm start/队列。

**M0 必须验收：**

1. 从旧 `gs12_joint_route/output/runs/*.json` 按 `(task_id,controller_id)` 找到一条 B 和 D 的同任务旧轨迹；用其**原任务定义**，分别运行本轮 β=0/1 的 24 s 前缀。对旧 NPZ 前 24 s 的命令、艇体状态、实际执行器、QP 有效标志逐步比较；数值最大差目标 `≤2e-7`，QP 标志完全一致。新旧预览取样、初态、配平与 W1 必须相同。若不对齐，定位差异后再进入新任务，不能只比较旧报告表格。
2. 在同一状态/35 步参考/相同 plan 下，捕获 β=0、.5、1 的实际 `model_parameters` 与 QP Hessian/gradient；分别应为 `(1,1)`、`(.925,1.5)`、`(.85,2)`，端点 QP 和控制动作应与旧 B/D 路径对齐。保持 `alpha[1]=0`、`d_hat=None`，不得把已有残差门控混入本轮。
3. 对 18 条参考运行 `joint_reference.derivative_error`，最大值 `<1e-3`；M3 若触发，再验证 64–80 s 参考延伸无位置/速度跳变。记录 `smoke/verify.json`、实际差值和源码 SHA。未通过则停止数据采集。

下列为**新脚本实施后的 CLI 合同**，目前仓库尚无 `gs12_predictor_route` 入口，不能直接执行或写成已完成命令：

```bash
python3 experiments/gs12_predictor_route/run_predictor.py verify --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/run_predictor.py prepare --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/run_predictor.py run dev_static --workers 4 --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/analyze_predictor.py --stage dev_static --out experiments/gs12_predictor_route/output
# G1 选择中间点后才执行；下面 .50 仅示例，须由 dev 实测决定：
python3 experiments/gs12_predictor_route/run_predictor.py freeze-static --beta 0.50 --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/run_predictor.py run test_static_locked --workers 4 --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/analyze_predictor.py --stage test_static_locked --out experiments/gs12_predictor_route/output
```

`verify` 不读取任何新锁定结果；先验收实现，后 `prepare` 一次冻结。`run` 从磁盘 `tasks.jsonl` 读任务，绝不能复用旧 `run_joint.py:TASKS`。`freeze-static` 核对完整 20 条 M1 配对和 dev 选择公式，写 `selection_static.json` 与 dev 指标哈希；若无候选，用 `--beta NONE` 明确记录，禁止 `run test_static_locked`。动态 CLI 在 G2 触发时再按第 7 节实现并写入 README。正式运行命令最终以新脚本真实 `--help` 为准。

每 worker 一条完整 64 s 串行轨迹；主进程校验 `run_id` 唯一，worker 只写自己的 JSON/NPZ，主进程统一写 manifest。数值库导入前限制 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；先 1 worker smoke，再按内存选择最多 4 worker。4 worker 的求解延迟含并行争用，单艇时延分析用 1 worker 的 p95/p99，不把该实验称为实艇实时性证明。

## 5. M1：五点静态扫描、选择门 G1

四条 `dev_static` 任务各跑 β=`0,.25,.5,.75,1` 且 W1 固定，共 **20 条完整 64 s**。先跑低频深度与高频东向 10 条可给中间进度，但 G1 必须等待四任务全部完成。A/C 权重 W0 属旧因子实验，本轮不加跑；β=0/1 已提供 B/D 同协议端点。

`analyze_predictor.py` 按任务清单左连接预期 `task×β`，检查协议/源码/trace 哈希、唯一 ID、完整时间网格、参考数组一致性与任务配对。缺失、失败、早停均显示，不为平均值静默排除。输出 20 行 `dev_static_metrics.csv`、分通道 `beta_channel_rmse.csv`、质量 `beta_qp_quality.csv`、`beta_tradeoff.png`（每任务 J 对 β，绝对数值与 log 轴）、`pareto_low_high.png`（低频深度 J/RMSE 对高频深度/东向/混合综合代价）。图中标注 β=0(B)、β=1(D)、评价窗口和数据 split。

**先预注册选择式，再看曲线：** 对每个任务 t，记 `m_t=min(J_0(t),J_1(t))`，`M_t=max(J_0(t),J_1(t),1e-12)`，定义任务遗憾 `r_β(t)=[J_β(t)-m_t]/M_t`，最差任务遗憾 `R_β=max_t r_β(t)`。端点与中间点都按**同一个**式子计算；同时计算各任务 J 等权均值 `Jbar_β`。质量有效要求四任务全部完整、无 QP 失败/诊断越界/饱和、无第 2 节命令尖峰标志。

G1-A：若一个有效中间点的 `R_β` 比两个端点中较小的 R **至少低 0.05（绝对差）**，且 `Jbar_β≤1.05×min(Jbar_0,Jbar_1)`，选择其中 R 最低者；同分先选 Jbar 低者，再选较小 β。最多冻结**一个**进入 M2。这个门只决定是否值得独立验证折中，**不是**性能晋级。另单列它是否逐任务 `J_β≤min(J_0,J_1)`；这个强支配条件不能从 R 改善自动推出。

G1-B：没有中间候选时，仅当新 dev 任务上 β=0 的低频深度 J 至少比 β=1 低 5%，且 β=1 在高频深度/东向/混合三任务中至少两条比 β=0 低 5%，所有端点质量有效，才进入 M3 **特征可识别性检查**。历史联合实验已有取舍迹象，但不能代替本轮新任务的确认。

G1-C：若无有效中间点，也未重复出上述端点交叉，停止静态扫描和频谱规则，报告负结果。若 β 曲线反常且需要区分 gain 与 tau 的贡献，允许**一次**开发诊断：在 `ps_dev_depth_low` 和 `ps_dev_east_high` 上分别跑 `(gain=.85,tau=1)` 与 `(gain=1,tau=2)`，共 4 条；标为 `diagnostic`，不把它们包装成新的五点扫描结果或本轮独立锁定收益。诊断不自动解锁更多网格。

## 6. M2：新的静态锁定测试（只在 G1-A 后）

在 `selection_static.json` 保存 β*、W1、G1 公式及各任务数值、dev run IDs、指标与源码哈希后，才打开 `test_static_locked`。四任务各跑 β=0、β=1、β*，共 **12 条**；三方法完全配对。若 β* 等于端点，说明 G1 实现有误，停止而非重复计数。

测试前预先指定**一个可部署固定比较器** `F_dev=argmin_{β∈{0,.25,.5,.75,1},质量有效} Jbar_β`，同分取较小 β；若它既不是 0/1 也不是 β*，在四条锁定任务额外运行 F_dev，成为 **16 条**，不能在打开锁定结果后挑较弱比较器。`min(J_B,J_D)` 的逐任务包络只作诊断图，它不是一个单一可部署控制器。

并列报告三种判定，不能互相替换：

1. **历史严格 B 门槛**：β* 相对 β=0(B) 的任务等权平均 J 下降 ≥5%，四任务各自不劣于 B，且无新增质量阻断。
2. **最佳固定比较器门槛**：β* 相对预先指定 F_dev 的任务等权平均 J 下降 ≥5%，四任务各自不劣于 F_dev，且无新增质量阻断。若 F_dev=β*，本项记为「同一配置、无独立超越」，不能算通过。
3. **端点包络诊断**：逐任务列 `J_β*−min(J_B,J_D)`、通道 RMSE 与绝对误差；是否兼顾端点由实际各任务判断，不拿均值替代。

完整性和质量先判：不齐全、早停、QP、越界、饱和时不出「通过」。打开静态锁定结果后，禁止在这四条任务上调整 β、门槛、参考相位或指标再称独立测试。若 M2 未通过而 M1 已满足 G1-B 的端点交叉，可继续 M3，但**只能**用第 3 节独立的 `dev_dynamic/test_dynamic_locked`；M2 不可变成动态规则开发集。若只是代码/记录损坏，记录具体技术原因、作废受影响 run，修复后用原候选和原协议重跑；不能顺便重新选参。

## 7. M3：条件启动的显式频谱规则

**只有 G1-B 成立，或 M2 失败但 M1 的端点交叉仍成立，才做本节。** 规则用 W1 常数和同一 β 参数化，不叫 RL。先做「只算特征、不控制」检查；再核对 β 接口；最后才开发规则。任何一步失败都停止动态分支并报告原因。

### 7.1 参考特征与可识别性门

假设部署端可在每个决策时刻取得未来 16 s 的**外生计划参考**，且已知计划延续到任务 80 s；MPC 自身只接收未来 35×.1=3.5 s，不能把二者混写。每 `0.5 s` 取从当前时刻开始的 **160 个 .1 s 样本**（终点不含），对 `(E/8,(D-20)/5,(VN-v)/2)` 分通道去均值和线性趋势、乘 Hann 窗、做 `rfft`。用 `rfftfreq(n=160,d=.1)` 转成 `ω=2πf`；只对非直流频线求和，定义 `E_total=Σ_{ω>0}|FFT|²/n²`、`ρ_high=Σ_{ω≥2.2}|FFT|²/Σ_{ω>0}|FFT|²`（三个通道汇总）。`E_total<1e-8` 时只标为低能量，β 维持前值，不计算不稳定比值。保存每个决策窗的起止时间、能量和 ρ；不得使用未来真实植物状态、实验任务标签或尚未发生的饱和。

先在五条 `dev_dynamic` 的**计划参考**上离线计算，逐任务和混合任务的低/高阶段画出 `ρ_high(t)`。进入控制试验的最低门槛：在单频任务 `t=32…48 s`，高频深度/东向各自的中位 ρ 比低频深度至少高 `0.30`；`pd_dev_mixed_a` 的高平台取预览起点 `t=28…30 s`、切回后的低平台取 `t=48…56 s`，`pd_dev_mixed_b` 对应取 `t=20…22 s` 和 `t=40…48 s`，两条混合任务的高平台中位 ρ 均须比各自低平台高至少 `0.20`。这些窗口在看特征结果前固定；16 s 预览会提前看到即将发生的切换，因此不能按当前时刻的频段标签随意划窗。低能量窗单列，不拿来凑中位数。若不满足，最多改做一次明确、可部署的参考导数能量特征对照；仍不可区分就停止。16 s 窗的频率分辨率约 `0.393 rad/s`，不能声称精确分辨 B1 全部频线。

### 7.2 模型参数逐周期注入验收

把 P0 base dynamics 固定为 `(1,1)`，每周期 `mpc.solve(alpha=[β_t,0],theta_hat=[.85,2],d_hat=None)`；**不改** `mpc.dynamics.fin_*` 属性、不重建 CasADi 函数、MPC、warm start、物理执行器或延迟队列。当前源码已表明 `model_parameters` 进入 `_one → dynamics.evaluate`，但仍须用实际轨迹验证：

1. 动态代码路径恒 β=0 与静态 B、恒 β=1 与静态 D 在同一新开发任务逐步对齐；命令/状态/实际舵面差 `≤2e-7`，QP 有效标志完全一致。若因不同数值调用路径达不到门槛，调查和记录原因，未验证前不评估规则性能。
2. 在 `pd_dev_mixed_a` 人工施加预先定义的 `β_t:0→.1→.2→.1`（每 .5 s 更新，其他时刻保持），捕获实际 `model_parameters` 和至少一组 QP 目标差异；确认 `mpc.previous/plan`、植物状态与队列没有因 β 变化被重置。记录 β、命令与实际舵面切换瞬态、QP 和求解时延。
3. 现有 `fallback_mode="hold"` 处理 QP 失败；本轮记录失败并阻断晋级，**不额外添加一个未验证的备用控制器或切换安全层**。因技术接线失败产生的轨迹不得当作规则负结果。

### 7.3 最小规则、开发批次和动态锁定门

初始规则 R0：

\[
\beta_{\rm raw}=\operatorname{clip}((\rho_{\rm high}-.25)/.50,0,1),\qquad
\beta_t=\operatorname{clip}(\beta_{t-.5}+\operatorname{clip}(\beta_{\rm raw}-\beta_{t-.5},-.1,.1),0,1).
\]

每 .5 s 更新一次，其余四个 .1 s MPC 周期保持 β；初始 β 取 `F_dev`，低能量窗保持上次 β。若 R0 在 `dev_dynamic` 质量有效但未达到开发晋级，可**最多再试一个预登记规则 R1**，只把 ρ 的线性区间改为 `[.35,.65]`，其余不变；不得无限调阈值、预览窗和更新周期。若 7.1 的特征门失败而改用导数特征，另写 `dynamic_protocol.json` 明确公式和只允许一次候选，不能仍称其为“16 s FFT 规则”。

每条 `dev_dynamic` 任务对 R0（可选 R1）及**去重后的**固定 β=0、β=1、β*=若 M2 曾选择、F_dev 配对运行。对照方法使用同一任务和同一未来参考可得范围；固定控制器不依赖额外预览。开发晋级要求规则相对 F_dev 的五任务等权平均 J 至少低 5%、每任务不劣、无新增 QP/越界/饱和/命令尖峰。最多冻结一个规则，保存代码/阈值/端点/初始化值/频谱特征定义与 dev run IDs 到 `selection_dynamic.json`。固定控制器的 LTI 频响矩阵不能拼成时变规则的频响；报告轨迹误差谱、非激励频线和 β/舵命令瞬态。

只有开发晋级且完整性审计通过，才在**五条 `test_dynamic_locked`** 上一次性运行冻结规则与同一组固定对照。严格动态通过：相对预先固定的 F_dev，等权平均 J 降低 ≥5%；五任务逐个不劣；无新增质量阻断。另列相对 B、D、β* 的值和低频绝对 RMSE。未通过后不在这五条任务上修改规则重测；无独立收益就停止，不开展 RL 扩训。

M3 开始时创建 `dynamic_protocol.json`，引用主协议 SHA 和未变的 core runner/植物/MPC 源码哈希，并冻结新 `spectral_policy.py`、特征定义、R0/R1、门槛、任务引用。这样可以在 G1 后才实现规则，同时保留 M1/M2 的代码版本；若核心动力学或 runner 控制循环改变，必须重新验证端点一致性并另立协议，不把不同核心版本的轨迹直接混算。

若 M3 触发，新脚本还须实现并在 README 验证以下入口：`feature-probe --out ...`（只写 `spectral_features.csv`，不跑闭环）、`verify-dynamic --out ...`（恒 β 端点和人工阶梯）、`run dev_dynamic --policy R0|R1 --workers N --out ...`、`freeze-dynamic --policy R0|R1 --out ...`、`run test_dynamic_locked --workers N --out ...`、`analyze_predictor.py --stage dev_dynamic|test_dynamic_locked --out ...`。`run test_dynamic_locked` 在 `selection_dynamic.json` 和 `dynamic_protocol.json` 冻结、dev 晋级与接口验收均通过前必须拒绝执行。R1 未运行时不得把它写为已比较方法。

## 8. 可选 M4、输出与复杂度边界

M1/M2 的结果若值得继续解释，优先在独立 `transfer` split 做 12/18 m/s（`Plant(speed,config)` 重新配平，`VN_ref=speed`）或幅值 `.003/.012` 的短任务配对。`.012` 是新压力值，不是已证实线性范围；出现饱和时报告受约束闭环性能，不称无约束局部线性 FRF。256 s B1/B4 三轴复矩阵仅在固定候选通过新锁定或明确需要机理解释时做；复用历史 W2/W3、全/半幅、各控制器独立零激励基线和 3×3 复矩阵算法。动态 β_t 不按固定 LTI 矩阵解释。迁移任务和协议需在运行前单独冻结，不能把新结果与旧确定性频线当成独立随机样本合并置信区间。

新输出至少包括：`protocol.json`、`tasks.jsonl`、`controllers.json`、`runs/<run_id>.json`、`traces/<run_id>.npz`、`manifest.json`、`dev_static_metrics.csv`、`beta_channel_rmse.csv`、`beta_qp_quality.csv`、`selection_static.json`（即使无候选）、`decisions.md`、`integrity_audit.json`、`figures/beta_tradeoff.png`、`figures/pareto_low_high.png`、`REPORT.md` 和可实际运行的 `README.md`。触发 M2/M3 才增加对应测试 CSV、频谱特征 CSV、β 时序图、`selection_dynamic.json`、`dynamic_protocol.json` 和动态接口验收记录。每条 run JSON 至少含 `run_id,protocol_sha256,task_sha256,source_sha256,git_commit,split,task_id,method_id,beta_or_policy,model_gain,model_tau,a_depth,a_rudder_rate,speed_mps,amplitude,phase_rad,completion_status,first_violation_time_s,qp_failures,command_saturation,actual_saturation,solve_p50/p95/p99_ms,trace_sha256`；NPZ 保留第 2 节规定的逐步数组。`run_id` 至少由协议 SHA、split、完整 task 哈希、β/规则版本、W1、航速/幅值/相位确定；worker 独立文件，父进程单独写 manifest。审计应逐 split 给出计划/实际数、缺失、重复、SHA 不符、早停、QP、越界、饱和，并拒绝在配对不齐时出晋级判定。

**复杂度准则：** 对推测性或发生概率极低的并发竞争、安全边界问题，先向用户说明具体失败模式、发生可能性、实现与维护成本，并取得同意；不得悄悄扩展范围。当前已知的并行写冲突只需由父进程写 manifest、预先断言 `run_id` 唯一；不引入数据库、分布式锁或通用事务框架。发现真实接线或完整性错误时，做最小修复并保留错误记录；不为未来可能的 RL 改写现有求解器。

最终报告分清「两轮历史已证实」「本轮开发」「本轮新锁定」「未运行/失败」，保留所有任务与通道的反例。确定性任务、频线和时间步不是统计独立样本；本轮不提供实艇部署、安全概率或全局鲁棒稳定性结论。**成功标准是得出可信的静态折中/上下文取舍判断，而不是必须找到正向配置。**
