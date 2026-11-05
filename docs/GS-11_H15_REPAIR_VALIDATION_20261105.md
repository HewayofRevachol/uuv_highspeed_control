## 代码审计后的追加修复（2026-11-05）

- [x] F1：`nominal_delay_tau()` 输出已缩放的 effective tau；`nominal_period_rollout()` 不再接收或重复应用 fin scale。在线与离线 label 使用 `nominal_5x20ms_issued_tau_once_v3`，旧 v2 工件会被拒绝。
- [x] 离线重标按新 normalization 从回合起点重放估计器，同时用重放 theta 计算 gain/tau，并递推该轨迹的 nominal rate。采集记录的 delay/tau 按原 theta 校验，原 rate 只作为采集证据，避免混合新 theta 与旧执行器状态。
- [x] F3：A0–A4 禁止 policy checkpoint；PPO 必须提供 checkpoint；未知方法和不匹配标签在建输出目录之前失败。评估默认 policy mean，`--stochastic-policy` 显式采样；run/episode 记录实际动作来源和 checkpoint hash。
- [x] F2：分析脚本分列 physical、no-fallback、deadline；“无干预”改善只使用 no-fallback。继续门先检查逐对物理/无干预结果无恶化且 fallback 次数不增加，精度使用双方无 fallback 成功集合中的逐对相对变化中位数。旧 P2 门另记 `legacy_p2_audit_repaired_v2`，不能当作 E4/E5 验收。
- [x] F4：先运行六个补偿 on/A0，按相同轨迹计算 zero/persistence/AR 分量 MAE。h25 每回合至少 50 个目标对，h5 或 h25 在两个几何上有同一组至少三维改善 ≥5%，其余维无 >5% 退化，才选 AR；否则用 persistence，AR 保留消融。零基准误差、非有限误差、单个回合覆盖不足均有回归测试。
- [x] E2 剩余主矩阵显式使用选择结果，复用 A0/A4，总数仍为 66；`predictor_selection.json` 保存选择证据与输入 hash，group 使用 job 字段。
- [x] F5 的 normalization 绑定：E2、evaluate 和直接 runner 检查 predictor 的 train split、补偿模式、v3 label 及 normalization 实际 hash。E2/evaluate 校验 manifest 中路径 hash，train 评估默认 T_fit；旧 holdout manifest 的实际路径 hash 会写入 run.json。

选择门的分几何 MAE 按有效目标数加权；最终 summary 的总体预测误差仍按回合等权。判断选择门应读取 `predictor_selection.json`，不能从总体均值反推。

### 本次验证

- 全部 `tests/unit`：**120 passed，13 subtests passed**；`compileall` 与 `git diff --check` 通过。41 条警告均为已有 OSQP `raise_error` 默认值将变更的提示。新增用独立植物推进的非零舵阶跃/反向对照，覆盖 tau scale=0.5/1/2/3；同时验证在线与离线多周期一致，以及采集 theta 与重放 theta 不同的场景。
- E2 真实 CLI 编排测试用合成 episode 替换仿真派发，分别验证 AR 通过/失败时先 6 后 60、共 66 个唯一任务，主矩阵 30 回合、24 个固定门配对、36 个消融配对，以及输出路径祖先名含 `main` 时分组正确。这是编排验证，不是 66 个闭环实验结果。
- 运行实际 E0：12 个短回合全部完成 40 周期，协议检查通过；匹配名义模型的最大缩放残差为 0，人工 0/1 gate 正控制在 40/40 个周期改变命令。输出目录为 `outputs/h15_cat_mpc_20261105/tau_v3_contract_probe_20261105`；该短实验不替代上面的非零舵独立模型测试。
- 尚未重新采集完整 E1、拟合正式 v3 predictor、运行完整 E2、重训 PPO 或使用最终 holdout；本次没有新的闭环性能优越性结论。

### 复验与迁移

- [ ] 在 Python 3.11 环境运行 `python -m pytest -q tests/unit`。
- [ ] 如需复现 E0，使用全新输出目录执行 `python -m experiments.h15_cat_mpc.e0_contract_probe --out outputs/h15_cat_mpc_20261105/e0_v3_recheck --workers 4`。
- [ ] 按补充实验计划重新生成 on/off 的 E1 trace、normalization 和 v3 predictor，禁止只改旧 artifact 的版本字符串；然后将同模式且 hash 匹配的工件传给 E2。
- [ ] F5 的源码/估计器参数自动拒配及训练集合全链路绑定、采集/PPO 入口的 manifest 隔离、reward 饱和边界和物理子步终止/pitch 口径等审计项仍需单独处理；本次不宣称审计全部关闭。

## 早期修复记录（v3 追加修复前）

- 在环境 reset 完成植物参数随机化后、首次 MPC solve 前应用 target-domain 舵效缩放；记录 reset 参数快照，并在每次 reset 清零二阶执行器内部速率。
- 将命令记录点放在父环境最终裁剪之后、植物执行之前。日志分开记录 MPC 命令、低层补偿和实际发出命令；兼容字段 `command_json` 指向 issued command。新日志标记 `issued_v2`，缺少新 schema 的分析/训练输入会被拒绝。
- 把因果残差标签改为共用的 `nominal_period_rollout`：5 个 20 ms 子步，显式推进延迟队列和执行器；使用可观测初值和 issued command，不注入隐藏扰动、洋流或采样船体参数。早期 trace 标记为 `nominal_5x20ms_issued_v2`，其中的重复 tau 缩放随后由本次 v3 修复，不能混用两个版本。
- 分开记录物理完成、无 fallback 物理完成、deadline clean 和 strict mainline。未实现的 backup 计数记为 `null`。新增配对复算程序，逐回合核对指标、命令链和 reset 快照。

## 早期验证结果（不作为 v3 代码的性能结果）

- 目标域、命令日志、响应估计与实验接口的针对性单测：12 项通过；涉及 Python 文件 compileall 通过。
- 最终 parity 在 `outputs/h15_cat_mpc_20261105/repair_parity_verified_20261105`，覆盖 2 个 geometry × 3 个 seed × A0/A4，共 12 回合。摘要和文件 hash 清单在同目录的 `parity_summary.json`。
- A0 与 A4 均为 5/6 physical strict、5/6 无 fallback physical、4/6 deadline clean；均无 QP fallback。6 个配对单元中 physical strict 改善 0、恶化 0；deadline 仅 1 次改善、1 次恶化，不能视为稳定收益。代码按此结果停止扩大 PPO。
- 首次验证发现低层命令会在父环境二次裁剪，之后又发现 episode 终止时日志收尾晚于校验；采集点和收尾顺序均已修正，并以最终代码重跑完整 parity。

## 后续实验边界

历史 predictor 使用旧标签，没有 `label_version`，修复后的 runner 会拒绝加载它，避免与新标签混用。尚未重新拟合 adapter normalization/calibration 或 residual predictor，也未重训 PPO；当前 parity 没显示 A4 的物理完成优势，因此按方案先停在接口修复与对照验证阶段。
