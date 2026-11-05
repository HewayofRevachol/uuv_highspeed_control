## 目的与边界

本方案针对 GS-11 代码审计中已确认的接口问题，目标是先得到可复现、可归因的 CAT-MPC/PPO 对照，再决定是否继续训练。修复期间保留旧输出目录，不覆盖旧 JSON；所有新结果使用新的 run id 和统一 source hash。当前运行时工作区中实际检出的项目目录为 `uuv_highspeed_control`；没有发现另一个名为 `uuv_workspace` 的本地目录，因此以下路径均相对于该项目根目录。

## P0：修正 target-domain 安装顺序并记录实际参数

问题位置：`experiments/h15_cat_mpc/run_episode.py:228-240,302-304`、`simulator/batch_path_env.py:187-209`、`aav_mpc/h15_target_domain.py:38-55`。当前 `install_target_domain()` 在 `env.reset()` 前执行，而 reset 会重写随机化后的 `parameters`、`delay_steps` 和 `tau`；额外的 fin gain `×0.9` 因而不生效。

实施步骤：

1. 不要简单地把 `install_target_domain()` 挪到 `env.reset()` 之后：`BatchedPathResidualEnv.reset()` 在参数随机化后会立即调用 `_solve_mpc()`，这样首个 command 仍会使用旧模型。应在 `simulator/batch_path_env.py:208` 参数/延迟/tau 写入之后、`simulator/batch_path_env.py:221-225` 首次 solve 之前增加受控 hook，例如 `_after_reset_parameters(mask)`。
2. `install_target_domain()` 改为注册 target-domain 配置和 hook；hook 只对本次 `mask` 应用 fin gain、delay scale，并在 reset 前安装 `plant_integrator`/`SecondOrderActuator`。这样随机化参数已经生成，且首个 MPC solve 前 target 参数已生效。`DDATPathEnv.reset()` 的调用链必须覆盖该 hook。
3. 将 `install_target_domain` 改为返回快照，至少包括 `parameters[3:5]`、`delay_steps[1:]`、原始 `tau[1:]`、`fin_gain_scale`、`fin_tau_scale`、`second_order` 和 `integrator`。在 `run.json` 和 `episode.json` 保存快照及 `target_domain_applied_after_reset=true`。
4. 快速断言：hook 后 `parameters[3:5] == reset_parameters[3:5] * .9`（容差 `1e-12`）；首个 solve 使用同一快照；后续 reset 不能丢失缩放；二阶执行器的 tau scale 必须在实际更新中生效。若不希望改 base env，也必须提供一个等价的 reset 内部扩展点，不能依靠外层 reset 后补丁。

验收：用固定环境 seed 运行 `target_domain=False`、修复后的 `target_domain=True` 各 1 回合；检查参数快照、首个 command 和 source hash。若 `.9` 没有出现在快照中，禁止进入后续 PPO 重训。

## P0：分离 MPC、低层补偿和实际发出命令

问题位置：`experiments/h15_cat_mpc/run_episode.py:47-104,150-186`、`simulator/ddat_env.py:195-240`。`_start_pending()` 当前保存的是 `base_command`，而 `_low_level_command()` 随后增加 disturbance compensation；`response_estimator.py` 却把 `command_json` 当作实际输入。

实施步骤：

1. 在 pending 结构中将旧字段改名为 `mpc_command_before`，并保留 `mpc_plan_commands`。
2. `GateEnv.step()` 调用父类 step 后，立即从 `info["command"]` 保存本周期实际发出的 `issued_command`；同时保存 `low_level_compensation = issued_command - mpc_command_before`。必须在下一次 `_solve_mpc()` 的 `_complete_pending()` 前完成赋值。
3. 新日志字段固定为：`mpc_command_json`、`low_level_compensation_json`、`issued_command_json`、`command_json`。其中 `command_json` 兼容性地等于 `issued_command_json`；估计器和 predictor 只读取 issued command。增加 `command_schema="issued_v2"`，旧日志标为 `base_v1`，禁止混训。
4. 对 issued command 做饱和、有限值和单位检查，并在每周期断言 `issued = clip(mpc + compensation)` 与 `trace.command` 一致（控制尺度误差小于 `1e-6`）。

验收：在同一状态快照下开启/关闭低层补偿各跑 1 回合；统计 `issued - mpc` 的非零比例，并确认估计器读到的 command 与 trace 完全一致。只有通过后才重新生成 estimator normalization、adapter 和 residual predictor。

## P1：重建无混淆 residual label

问题位置：`run_episode.py:88-103`、`train_predictor.py:29-38`。当前 label 用周期开始的 `actuator_before` 固定积分 0.1 s，但真实环境每 0.02 s 更新二阶执行器并经过命令队列；label 因此混入执行器瞬态，不能直接解释为外部扰动。

实施步骤：

1. 每个周期记录 `state_before`、`actuator_before`、`queue_before`、`delay_steps`、`tau`、`issued_command`、`theta_used` 和 `parameters_used`。
2. 新增 `nominal_period_rollout()`：按控制器同一 5×0.02 s 步长、队列、延迟、名义执行器模型和积分器推进 22 状态；禁止读取隐藏随机植物参数、disturbance 或 current。输入只使用周期开始可观察快照和 issued command。
3. 定义 `residual_body_acc = (state_after[5:10] - nominal_state_after[5:10]) / 0.1`，并将 label 版本写入 manifest。旧 `train_ar1` 输出不与新 label 混合。
4. 做四格诊断：旧/新 label × 传统低层补偿开/关；另加“匹配 nominal、无扰动、改变 command”的零外扰测试。如果零外扰新 label 仍有系统性大残差，优先修 nominal rollout，而不是调整 predictor 网络。
5. 在 train-only 12 回合拟合 AR(1)，在独立 dev 回合按 1/5/25 步及几何阶段报告 persistence/AR；训练和验证都按回合等权。

## P1：修正指标和配对统计

问题位置：`experiments/h15_cat_mpc/analyze_ppo.py:24-68`、`run_episode.py:325-377`。现有 `strict_mainline` 把物理完成、QP fallback 和 deadline 混在一起，`strict_hist` 还使用周期末相对路径 pitch；`backup_periods` 固定为 0。

实施步骤：

1. 输出四个独立字段：`physical_strict`、`no_fallback_physical`、`deadline_clean`、`strict_mainline`。主完成指标用 `physical_strict`，fallback、deadline 单独给周期数和回合数。
2. 物理标志只从 5 个子步的实际 pitch/depth/speed/lateral 计算；删除 `strict_hist` 对主结论的作用，或明确标注为 legacy diagnostic。
3. 在 `analyze_ppo.py` 中把 `improved`/`worsened` 改为 `physical_strict` 配对；另生成 deadline-only 配对表。没有实现 backup 时将其写为 `null`/N/A，不把硬编码 0 当作“无备份干预”的证据。
4. RMSE 同时输出共同成功回合的逐配对变化中位数与“候选中位数/基线中位数−1”，禁止只报告后者。失败回合的截短轨迹单列，不能与全程成功回合池化。

验收：在已保存 P1/P2 JSON 上运行独立标准库复算，要求 `flag_recomputation_mismatches=[]`，并能复现 A0/A4/PPO 的物理完成表。旧统计脚本结果保留为 legacy，不覆盖。

## P2：统一 runner 后再重跑和训练

完成顺序：

1. 将以上 P0/P1 修改合并到唯一 runner；恢复或明确记录旧 P1 runner，保证 P1/P2 使用同一代码契约。
2. 先做 12 回合短 parity：A0、A4 × 2 geometry × 3 seeds；检查 target snapshot、issued command、residual manifest、四类指标。
3. 修复后冻结 normalization/adapter/predictor，并记录每个输入文件 SHA256；重新生成 12 个 predictor train 回合和 6 个 dev 单元。
4. 运行 R2 局部 gate 扫描：在同一状态快照测试 `a∈{0,.25,.5,.75,1}`，先验证 A4 的收益是否存在邻域稳定性；不要先扩大 PPO。
5. 若固定 gate 仍在困难单元稳定优于 A0，再训练短程 PPO：训练几何覆盖 dev 的深度/半径边界，至少每批更新而非只做三次大更新；评估 deterministic mean、匹配 A4 的初始 alpha/slew，并用独立 policy seeds。

## 验收门槛、回滚和交付

- 代码单测：target reset 顺序、issued command 一致性、nominal rollout 零扰动一致性、指标分类各 1 个测试。
- 数据门槛：新日志无缺字段/NaN；`command_schema`、label version、runner SHA、environment/policy/action seed 齐全。
- 实验门槛：先通过 12 回合 parity；随后报告物理完成、fallback、deadline、共同成功逐配对误差。未通过 parity 时停止训练和扩展 seed。
- 回滚：旧输出只读保留；若新 label 或 target snapshot 不通过，恢复到旧 runner 仅用于复现历史结果，不能把旧结果和新结果合并统计。
- 交付：将新报告、audit JSON、source manifest 和 run summary 一起归档到新的输出目录，并在报告中同时列出旧结果与修复后结果。
