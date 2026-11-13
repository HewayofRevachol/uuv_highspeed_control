# GS-22 设计所用证据节选

这些文件来自 2026-11-13 GS-21 共享工作空间归档：`experiments/gs20_gs12_continuation/`。它们是已有运行数据与接口的副本，本次没有新增仿真或训练。主报告为该归档的 `reports/20261113_results/REPORT.md`，本目录仅收录支持设计取舍的表格和接口文本，避免将整个实验包误当作新实现。

| 文件 | 内容与口径 |
| --- | --- |
| [A_frequency.csv](A_frequency.csv) | 匹配/组合失配对象、三种方法、七频点的归一化误差频响奇异值；共 42 行 |
| [B_per_task_beta.csv](B_per_task_beta.csv) | 两结构×16 条开发任务的 β=0/0.5/1 全程评分和 E/D/VN 物理 RMSE；共 32 行；`best_beta` 对纯速度近乎并列项的末位差异不应解释为有效偏好 |
| [C_D_pairs.csv](C_D_pairs.csv) | C 迁移与 D 切换的逐任务配对；共 16 行；变化比例为候选/基线−1 |
| [D_window_scores.csv](D_window_scores.csv) | 原 `window_scores.csv` 中 `stage == D_light` 的全部行和列；含全程、稳态、三阶段及过渡窗口 |
| [closed_loop.py.txt](closed_loop.py.txt) | 历史 Session 的 19 维观测、8 秒动作、β 限速、MPC 调用、记录与奖励接口；仅作原样文本快照 |
| [policy.py.txt](policy.py.txt) | 历史 19 维 PPO 策略接口；未在轻量实验中完成训练，不是已验证的有效策略 |
| [config.json.txt](config.json.txt) | 历史对象、频点、周期与参数配置；不是 GS-22 新模块的可执行配置 |
| [provenance.json](provenance.json) | 来源相对路径、SHA-256、节选转换及未复制参考文件的哈希 |

CSV 中 `R2` 是归一化误差平方和的时间平均，越小越好；E/D/N 单位为米，VN/VE/VD 为 m/s，角度为 rad。B 的所有任务等权平均后选固定 β，不能把平均后的改善率视为逐路径一致改善。D 的东向规则动作与固定法相同，故两条平局不构成两个退化案例。

原始 NPZ 轨迹和完整 GS-21 报告仍在原工作空间归档，未随本次设计提交复制。这里的来源哈希确认文件身份，不等于仓库已包含重建完整实验的一切数据。接口快照的导入路径也不构成仓库内可运行代码。
