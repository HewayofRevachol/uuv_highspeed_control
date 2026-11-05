## 修复内容

- 在环境 reset 完成植物参数随机化后、首次 MPC solve 前应用 target-domain 舵效缩放；记录 reset 参数快照，并在每次 reset 清零二阶执行器内部速率。
- 将命令记录点放在父环境最终裁剪之后、植物执行之前。日志分开记录 MPC 命令、低层补偿和实际发出命令；兼容字段 `command_json` 指向 issued command。新日志标记 `issued_v2`，缺少新 schema 的分析/训练输入会被拒绝。
- 把因果残差标签改为共用的 `nominal_period_rollout`：5 个 20 ms 子步，显式推进延迟队列和执行器；使用可观测初值和 issued command，不注入隐藏扰动、洋流或采样船体参数。训练 trace 必须标记 `nominal_5x20ms_issued_v2`。
- 分开记录物理完成、无 fallback 物理完成、deadline clean 和 strict mainline。未实现的 backup 计数记为 `null`。新增配对复算程序，逐回合核对指标、命令链和 reset 快照。

## 验证结果

- 目标域、命令日志、响应估计与实验接口的针对性单测：12 项通过；涉及 Python 文件 compileall 通过。
- 最终 parity 在 `outputs/h15_cat_mpc_20261105/repair_parity_verified_20261105`，覆盖 2 个 geometry × 3 个 seed × A0/A4，共 12 回合。摘要和文件 hash 清单在同目录的 `parity_summary.json`。
- A0 与 A4 均为 5/6 physical strict、5/6 无 fallback physical、4/6 deadline clean；均无 QP fallback。6 个配对单元中 physical strict 改善 0、恶化 0；deadline 仅 1 次改善、1 次恶化，不能视为稳定收益。代码按此结果停止扩大 PPO。
- 首次验证发现低层命令会在父环境二次裁剪，之后又发现 episode 终止时日志收尾晚于校验；采集点和收尾顺序均已修正，并以最终代码重跑完整 parity。

## 后续实验边界

历史 predictor 使用旧标签，没有 `label_version`，修复后的 runner 会拒绝加载它，避免与新标签混用。尚未重新拟合 adapter normalization/calibration 或 residual predictor，也未重训 PPO；当前 parity 没显示 A4 的物理完成优势，因此按方案先停在接口修复与对照验证阶段。
