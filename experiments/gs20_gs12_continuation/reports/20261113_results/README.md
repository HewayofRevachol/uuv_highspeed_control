# 报告包说明

优先打开 `REPORT.html`：图片已内嵌，可离线阅读。`REPORT.md` 是同内容的Markdown源稿；`figures/` 同时提供PNG和矢量PDF，`tables/` 提供原单位的完整CSV。

本包包含260条正式记录的汇总表、分窗结果、逐任务配对、核验结论、源数据哈希及运行代码/输入快照。约120 MB的原始运行资料已归档到共享工作空间的 `outputs/gs20_continuation/`，不重复放进这个精简报告包。早期不完整的原大网格尝试也在那里单独保留。

在共享工作空间中，报告目录为 `experiments/gs20_gs12_continuation/reports/20261113_results/`；从仓库根目录运行：

```bash
python3 experiments/gs20_gs12_continuation/reports/20261113_results/build_report.py
```

该命令只离线重算数据表、图和核验记录，不运行闭环或训练。它读取归档原始运行资料，以及 `experiments/gs12_analaysis/GS-12_frequency_data.zip` 中的历史频响参照。两者均已放入共享工作空间。报告正文是人工整理的解释，不由脚本自动改写。

原始数据的形状与单位：每条 `traces/*.npz` 保存0.02秒步长的时间、状态、参考、输出和误差，以及0.1秒步长的命令、β和QP记录。八个误差分量依次为N/E/D位置（m）、pitch/yaw（rad）、VN/VE/VD（m/s）。`R2_valid` 留空表示没有完整有效全程评分；`R2_observed` 和分窗数据只能用于该已观测区间，不得把提前终止轨迹当作完成。

源数据归档、统计复核和报告生成日期：2026-11-13。E/F按预设门槛未触发，本包不含PPO策略或策略留出效果。
