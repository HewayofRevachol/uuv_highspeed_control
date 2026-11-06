# GS-13 SR-PPO r1 实验决策记录

- 与 r0 的唯一区别：`residual_limit` 从 0.25 降到 0.20；其余配置和 444 个场景保持不变。
- 执行：真实 GS-12 `PlantHarness` 闭环；41002 seed，8,192 PPO decisions，16 次更新，40,947 个 0.1 s control steps，182 个 rollout episodes。
- 训练设备：受限运行环境中 CUDA 不可见，实际为 CPU。`/usr/bin/python3` 在沙箱外可以看到 RTX 4060 Laptop GPU；现有训练实现默认不将模型迁移到 CUDA。
- Teacher：8 个 train 场景，336 个监督样本；dataset SHA256 `271d8b0713d97b770869a26e0f8c9035e177db2b864e9b3e786e990b9e86647c`。
- Dev：完整配对 24 场景、4 方法、96 行；无截断。feasible 层（22 场景）SR-PPO 失败 2 次，与 fixed_best/rule_best 持平；成本 7.7732，fixed_best 为 7.7769，rule_best 为 7.0923。相对更优 rule_best 的配对成本增益为 -9.60%，未达到 +5% 晋级门槛。
- 次级指标：SR-PPO 相对 rule_best 的 QP failure delta 为 0，deadline miss delta 为 +2，饱和比例差为 +0.0182。
- 决策：已由 dev 选择终点 checkpoint，但不进入 test；残差上限下调没有使 checkpoint 达到晋级门槛。
- 限制：结果是仿真闭环结果，不构成实艇安全保证。高 shield override 率需结合逐项原因解释。

## 主要产物

- `run_manifest.json`、`cases.jsonl`、`obs_schema.json`
- `teacher/summary.json`、`teacher/dataset.npz`
- `training/41002/updates.jsonl`、`training/41002/training_summary.json`、两个 checkpoint
- `evaluation/dev/14d543ceb10e/paired_metrics.csv`、`training_summary.json`、`report.md`
- `selected_checkpoint.json`
