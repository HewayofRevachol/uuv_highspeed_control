# RL–MPC 研究报告索引

| 文件 | 内容 |
| --- | --- |
| [顶会文献扫描](RL_MPC_top_tier_20261101.md) | 早期 RL–MPC 先例整理。 |
| [完整选题综述](UUV_RL_MPC_INNOVATION_REVIEW_20261102.md) | 候选路线、文献和实验边界。 |
| [推荐主路线](RECOMMENDED_ROUTE_ACTUATOR_AWARE_RL_MPC_20261102.md) | 执行器状态与长期价值增强 MPC 的研究假设。 |
| [证据核验](VERIFY_ACTUATOR_AWARE_RL_MPC_EVIDENCE_20261102.md) | 对照原始逐回合记录，区分已观察现象与待证机制。 |
| [仿真与现实性审查](AUDIT_UUV_MECHANISM_REALISM_20261102.md) | 重跑控制回合、动力学响应和路径稳态筛查。 |

基础阶段报告在 [`report1030.md`](../report1030.md)。现实性审查的[复现脚本](../high_speed_experiments/audit_uuv_mechanisms.py)和[运行结果](../outputs/reports/uuv_mechanism_audit_20261102/)按工作区约定分别放在实验脚本与输出目录；运行目录中的 `run.json` 记录了输入、版本和种子。
