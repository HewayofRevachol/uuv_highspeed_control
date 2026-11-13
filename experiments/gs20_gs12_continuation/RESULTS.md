# 实验结果报告入口

2026-11-13 更新：已汇总 GS-12 延续实验的阶段 A 与轻量版 B–D，并说明 E/F 按预设门槛未触发的原因。

- 正文：`reports/20261113_results/REPORT.md`
- 单文件阅读版：`reports/20261113_results/REPORT.html`
- 全部逐任务数据表：`reports/20261113_results/tables/`
- 核验结果与哈希：`reports/20261113_results/audit.json`、`data_manifest.json`
- 原始轨迹：仓库根目录下 `outputs/gs20_continuation/`

报告独立复核260条正式记录；轻量B–D共128条完整记录。固定β=0.5在指定开发/迁移任务上改善误差，所测规则调度为0胜、2平、2负，因此没有开展PPO和策略留出评价。
