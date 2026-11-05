# GS-11：H15 补充实验代码逻辑审查

审查日期：2026-11-05。审查对象：GitHub 分支 `codex/gs11-supplemental-experiments`，提交 `7d3d5e1b1fd0ba8f0ce88ad5bded2c58de8b14df`。

**结论：这版实现不能直接判定为逻辑正确，也不建议现在据此开始正式性能比较或重训 PPO。** 命令记录、reset 安装顺序和指标拆分已有实质修正；但残差标签、预测器选择、工件匹配、评估方法标识和 PPO 统计门仍有问题。应先修正下列问题，再重跑具有足够激励的 E0，随后重建受影响工件。

本轮审查的是实际拉取的新提交，不是重复检查旧版 `c9120d0`。没有修改实验实现，也没有执行 E1/E2 全量实验、PPO 训练或最终留出实验。文件位置均相对于仓库根目录；用户工作区为 `uuv_workspace`，复现时在该工作区对应仓库中操作即可。

## 1. 已完成的验证及边界

| 检查 | 本轮结果 | 能证明什么 |
|---|---|---|
| target-domain、residual、response-estimator、cycle-log、strict-contract 相关现有单测 | **16 passed，4 subtests passed** | 覆盖已有测试定义；不能替代独立模型对照 |
| `compileall`、`git diff --check` | 通过 | 语法和格式检查 |
| PPO/AR 七项性质检查，真实导入原函数 | 全部通过 | 五周期 reward、终止/截断 bootstrap、raw-z logprob、初始 slew、确定性动作、回合内 AR 滞后对 |
| 统计脚本、evaluate 编排、E2 编排/聚合的合成输入复现 | 确认下述缺陷 | 调用真实入口，仅替换仿真派发或输入日志；不作为闭环性能数据 |
| T_fit/D_diag 路径与 manifest 对照 | 六条路径 hash 均匹配，12/6 单元及 seed 隔离正确 | 没有发现这两份现有数据已经污染 |

验证环境为独立 CPU 环境：Python 3.11.18、PyTorch 2.15.0、NumPy 2.4.6、CasADi 3.6.5、OSQP 1.1.3。系统默认 Python 3.9 无法导入项目已有的 `str | None` 注解，因此切换至 3.11 后完成测试；这不是本次补充代码引入的算法错误。没有据此评判部署机器的 deadline 或实时性。

## 2. 必须在相应实验阶段前修复的问题

### F1 — P1：残差名义模型将舵时间常数重复缩放

**位置：** `experiments/h15_cat_mpc/residual_predictor.py:17`、`:38`；`run_episode.py:59`、`:127`；`train_predictor.py:36`、`:41`。

`nominal_delay_tau(fin_tau_scale=theta[1])` 已生成 `tau_base × theta_tau`。调用方随后又把同一 `theta_tau` 传给 `nominal_period_rollout`，后者在二阶执行器分支再次相乘。在线残差和离线标签均走这条路径。

例如默认 `theta_tau=2` 时，残差模型实际使用 `4 × tau_base`；`matched_nominal` 植物在 `run_episode.py:329` 只使用 `2 × tau_base`。因此执行器发生瞬态时，即使植物参数完全匹配、没有外部扰动，也会产生伪残差。在线和离线都犯同一错误，不能通过“二者结果一致”证明标签正确。

一个实际 40 周期 `matched_nominal/off/step` 回合的 issued fin command 全程为零，所以该回合的 scaled residual 恰为 0；这不是对舵瞬态的覆盖。独立非零舵输入复现（theta_tau=2、无外部扰动）得到：植物 fin tau=0.16 s，当前标签有效 tau=0.32 s，最大缩放残差约 **0.04617**；只缩放一次时与独立二阶植物推进完全一致。相同输入下一阶 RK4 MPC 点模型与一阶名义 rollout 的状态差约 `4.3e-19`，说明修复时还应保持标签和控制器使用同一执行器模型契约，而不能用“二阶标签与一阶 MPC 不同”掩盖重复缩放。

现有 `tests/unit/test_h15_residual_predictor.py:25` 用 `predicted - predicted` 构造零残差，没有独立执行植物，因此该测试必然通过，无法发现此问题。

- [ ] 明确 `tau` 参数是基础值还是最终值，只允许在一个位置应用 `theta_tau`。
- [ ] 用独立植物推进与非零舵阶跃/反向输入比较，覆盖 `theta_tau=0.5/1/2/3`，不能再用同一预测结果减自身。
- [ ] 标签公式修正后升级 label contract/version，重新生成标签与 predictor；重建或重新验证依赖这些标签的后续工件和策略。

### F2 — P1：PPO 分析仍会把 fallback 完成算成“无干预”收益

**位置：** `experiments/h15_cat_mpc/analyze_ppo.py:28`、`:35`、`:48`、`:112`。

新版已把 deadline 从物理改善中分离，这是正确修正。但 `improved_to_unintervened_physical_finish` 只判断 `physical_strict`，没有使用已有的 `no_fallback_physical`；共同精度集合也只判断双方 `success`。因此有 fallback 的完成仍可进入精度比较，且 precision rule 不要求无干预结果不恶化。

调用真实分析入口的两个合成例子：

1. 两个原先失败的回合变为物理完成，但候选各有一次 fallback：输出 `improved=2`、`rule1=true`，实际新增无 fallback 完成为 **0**。
2. 两个原先无 fallback 完成的回合，候选虽然 RMSE 较低却各新增一次 fallback：输出 `rule2=true`，实际无干预完成发生退化。

这些是统计逻辑反例，不是实际实验收益。E2 的 `physical_pair()` 已正确限制共同精度集合为双方 `no_fallback_physical`，两个分析入口目前口径不一致。

- [ ] 分列 physical、no-fallback、deadline 的改善与恶化；“无干预”字段必须使用 `no_fallback_physical`。
- [ ] 所有继续门先检查 physical/no-fallback 无配对恶化，以及 fallback 不增加，再判断完成或精度门。
- [ ] 共同精度集合使用双方无干预成功，并按新方案计算**逐对相对变化的中位数**；当前 `median(candidate)/median(baseline)-1` 不是同一个统计量。
- [ ] 给上述两种反例加回归测试；旧 P2 pilot 阈值和新 E4/E5 阈值分版本保存，不能把旧分析输出直接当新阶段验收。

### F3 — P1：方法标签与实际策略可以不一致

**位置：** `experiments/h15_cat_mpc/evaluate.py:30`、`:52`、`:59`、`:71`；`run_episode.py:361`、`:393`。

`evaluate` 对所有 `--methods` 都套用 checkpoint，runner 则按 checkpoint 是否存在执行策略，不按方法标签限制。仅传 `--policy-checkpoint` 而省略 `--methods` 时，默认 A0–A4 的 60 个 train 任务全部携带同一个 PPO checkpoint。它们会执行策略，却保留 A0–A4 标签；这些固定方法的初始 alpha 还不相同。反过来，`--methods PPO` 不传 checkpoint 也被接受，runner 会保持 alpha=(0,0)。

真实入口编排复现确认上述两种调用都未被拒绝。这样产生的“基线”和“PPO”结果不能按名称解释。

- [ ] 固定 A0–A4 禁止携带 policy checkpoint；PPO 必须携带 checkpoint；未知方法直接报错。
- [ ] 校验放在创建输出目录及启动任务之前，并在底层 runner 再校验一次。
- [ ] 日志分开保存 method family、checkpoint hash、初始 alpha 和实际动作生成方式，避免用名称代替执行身份。

### F4 — P1：E2 主实验没有执行主预测器选择门

**位置：** `experiments/h15_cat_mpc/e2_matrix.py:118`、`:210`；补充实验计划 `:83`、`:85`、`:89`。

计划要求先运行补偿 on 的六个 A0 回合，在同一批轨迹上比较 zero/persistence/AR；AR 达到预注册分量 MAE 门才成为主预测器，否则退回 persistence。当前入口无条件将 30 个主矩阵任务设为 `ar_or_persistence` 并传入 AR checkpoint；同轨迹误差到全部 66 回合结束后才计算。这里的 `ar_or_persistence` 不代表按误差自动择优：有 checkpoint 就加载 AR。

真实任务编排复现：**30/30 个 main 任务均使用 AR checkpoint**。所以目前 66 回合计数正确，但不等同于执行了计划中的 E1→E2 选择流程。

- [ ] 先运行并复用六个 on/A0，产生包含 h5/h25 分量误差、覆盖数和判定的选择记录。
- [ ] 将选择结果显式传入其余主矩阵；未达门的 AR 保留在预注册消融中。
- [ ] 用 AR 明显劣于 persistence 的合成误差验证主矩阵确实切换；记录选择记录 hash。

### F5 — P1：工件同模式不代表同一残差定义，缺少 normalization 绑定校验

**位置：** `experiments/h15_cat_mpc/e2_matrix.py:80`；`residual_predictor.py:99`；`train_predictor.py:112`。

predictor 工件已经保存 `manifest.normalization_sha256`，但加载入口只检查补偿模式和 label version，不比较传入 normalization 的实际 hash。normalization 会影响估计器的有效性、回退和因果 theta，从而影响残差定义；同为 `on` 的两套文件不能随意互换。

复现中，给 on/off predictor 填入与实际 normalization 不同的 hash，真实 E2 入口仍接受并生成全部 **66 个任务**。这是防错缺陷，不表示仓库现有工件已经发生串用。

- [ ] 在 E2、PPO 和直接 runner 的共用加载层绑定 normalization、adapter 参数/源码、label contract 和训练集合 hash。
- [ ] mode 一致但 normalization hash 不一致时，在启动第一回合之前失败。
- [ ] 不只验证冻结 JSON 文件自身 hash；还需核对运行时真正使用的估计器代码和参数。

## 3. 其他已确认问题

| 优先级 | 位置 | 触发与影响 | 修复验收 |
|---|---|---|---|
| P2 | `e2_matrix.py:151`、`:180` | 用 `"main" in out.parts` 推断组；合法输出目录 `outputs/main` 会把 36 个 ablation 也归为 main，66 回合结束后聚合 `StopIteration`。已用合成 episode 真实执行聚合复现。 | 在 job 元数据显式保存 group；不同目录名应得到相同分组。 |
| P2 | `evaluate.py:75`、`:83` | 有 checkpoint 时强制 deterministic=True，但 run.json 默认记录 False；随机 action-seed 诊断无法通过该入口实现。 | 明确 deterministic/stochastic 选项并记录实际执行值。 |
| P2 | `evaluate.py:45`、`ppo_train.py:167` | `evaluate --split train` 仍取旧 train seeds 31001–31003，PPO 已取 T_fit 51001–51003。直接用该命令做新方案的 T_fit learned-mean 会跑错单元。 | 增加显式 manifest/split 选择并校验、归档 hash。 |
| P2 | `run_episode.py:213` | 推力已裁剪至归一化 [0,1]，但饱和惩罚判断 `abs(thrust-0.5)>0.5`，上下边界永远不触发；后面的 saturation 日志用另一套判断。 | 用统一的物理边界判定计算 reward 和日志；分别测试推力与舵面边界。 |
| P2 | `collect_training.py:34`、`e2_matrix.py:91`、`ppo_train.py:174` | 未统一核验 manifest 的 CSV hash；T_fit/T_boundary 主要按 ID 判互斥，改名引用相同路径仍可绕过。 | 启动前校验路径内容 hash、唯一单元和集合隔离；本轮六条现有 CSV hash 均正确。 |
| P2 | `batch_path_env.py:320`–`:380`、`run_episode.py:202`–`:230` | 物理阈值在一个控制周期的 5 个子步全部执行后才汇总；越界后的剩余子步仍进入 residual 和日志。当前语义是“周期级干预”，不是首个 20 ms 子步立即停止。 | 明确并记录 period-level 语义，或在首个越界子步截断；增加首个越界时间和截断状态。 |
| P2 | `run_episode.py:466`–`:470`、`:542` | `strict_substeps` 的 pitch 是绝对状态 pitch，`strict_hist` 的 pitch 来自相对路径误差；两个 strict 指标不可直接比较。`physical_strict` 重算子步标志的口径是正确的。 | 统一 pitch 定义或改名，禁止把 `strict_hist` 当作物理子步指标。 |
| P2 | `train_predictor.py:30`–`:45` | 离线重标从 theta 重新生成 delay/tau/rate，没有消费 episode 记录的 nominal delay/tau/rate；修复重复缩放后仍可能与在线标签契约漂移。 | 直接读取并校验 `nominal_*_json`，或对重算值逐字段断言相等，再拟合 predictor。 |

`freeze_inputs.py:23` 仍接受旧 P1 的 30 回合/6 配对格式。可以通过旧分析入口转换 E2 主矩阵，但这个文件本身**不能证明** E1 预测器门或 E4 探索门已经通过。E3–E7 在实现说明中也被标明是后续条件阶段，不能因存在 PPO 入口就认为这些阶段已实现或验收。

## 4. 已修正或本轮未发现错误的逻辑

- target-domain modifier 现通过 reset 后、第一次 solve 前的 hook 应用；二阶执行器 rate 在 reset 时清零。旧版“安装后被 reset 覆盖”的路径已修正。
- issued command 现在在最终裁剪后、植物执行前记录；兼容 `command_json` 指向 issued，并检查与实际 command 对齐。日志命令不再沿用旧版 base command。`low_level_compensation` 当前是最终 issued−MPC 的净差值，若要分析限幅前补偿，还应另记原始分量。
- runner 分列 `physical_strict`、`no_fallback_physical`、`deadline_clean`、`strict_mainline`，backup 不支持时用 null。分析层仍需按 F2 统一。
- target 轨迹使用二阶执行器和 midpoint 积分，而 condensed MPC 点模型使用一阶执行器和 RK4；计划允许这种已知 target mismatch 进入 residual，因此 `d_hat` 不能解释成纯海流/外部扰动。修复 F1 时应保持该设计边界，同时只消除重复 tau 缩放。
- PPO 把 raw-z 当作概率空间，采样和重算 logprob 一致。sigmoid/slew 是确定性的执行映射，不应把“没有对 alpha 另算 Jacobian”直接判为 PPO 错误。
- 每个决策的五周期折扣 reward、完整回合终止 bootstrap=0、500 决策截断使用下一决策 value，这些性质检查通过；没有做完整 PPO 收敛性或泛化验证。
- 新 AR 拟合按 episode 构造滞后对，没有把上一回合末尾与下一回合开头拼成训练样本。
- E2 的 on/off 各与本模式 A0 配对；预测器离线误差使用同一 A0 轨迹。30+36 的回合复用计数在 AR 主预测器这一设定下正确。

## 5. 修复和复验顺序

- [ ] 先处理残差公式和名义模型契约，加入真实非零舵激励、独立植物对照；重新通过 E0 后再采集/训练。
- [ ] 加上方法/checkpoint 和工件 hash 的启动前校验，修复输出目录分组；先用合成输入验证拒绝行为，避免跑完才发现无效。
- [ ] 完成六个 A0 的预测器选择流程，再启动其余 E2 任务；选择失败时按原方案退回 persistence。
- [ ] 统一 PPO/E2 的物理与无干预配对口径、逐对精度统计及继续门；让 fallback 反例不能过门。
- [ ] 补齐新 T_fit/D_select 入口、确定性/随机模式和阶段证据绑定后，才进入条件式 PPO。
- [ ] 所有报告记录实际 commit、工件版本和 schema。修复前的实验输出保留为历史，不能仅修改标签版本后重用为新结论。

## 6. 可复核材料

随报告提供的证据包包括实际函数检查结果和可重跑诊断脚本。入口/统计脚本用合成数据复现逻辑，不会训练策略或跑全量仿真：

```bash
python verify_statistics_evaluation.py --repo /path/to/repository --out statistics_evaluation_evidence.json
python verify_supplement_pipeline.py --repo /path/to/repository --out supplement_pipeline_evidence.json
```

应使用上述审查提交及 Python ≥3.10，安装项目 CPU 测试依赖。脚本成功表示“在被审查版本上复现了记录的行为”，不表示实验代码通过验收。

本轮现有单测命令：

```bash
python -m pytest -q tests/unit/test_h15_target_domain.py tests/unit/test_h15_residual_predictor.py tests/unit/test_h15_response_estimator.py tests/unit/test_h15_cycle_log.py tests/unit/test_h15_strict_contract.py
python -m compileall -q experiments/h15_cat_mpc experiments/h15_mainline simulator aav_mpc
git diff --check
```
