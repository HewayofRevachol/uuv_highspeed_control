# GS-13 SR-PPO r0 实验决策记录

- 数据集：全量 train/dev/test manifest 已固定；train 300、dev 24、test 120，场景文件共 444 行。
- 执行：真实 GS-12 `PlantHarness` 闭环；41002 seed，8,192 PPO decisions，16 次更新，40,952 个 0.1 s control steps，181 个 rollout episodes。
- 训练设备：该次训练在受限运行环境中执行，PyTorch 看不到 CUDA，实际为 CPU。沙箱外 `/usr/bin/python3` 可见 RTX 4060 Laptop GPU；这份代码目前默认将策略留在 CPU，换解释器不会自动把网络迁到 GPU。
- Teacher：8 个 train 场景，336 个监督样本；dataset SHA256 `58bf9af04b444af0072dc5cae9a03c013a9c51cb3cf03ff1da641b7da29c1b42`。
- Dev：完整配对 24 场景、4 方法、96 行；无截断。feasible 层（22 场景）SR-PPO 失败 2 次，与 fixed_best/rule_best 持平；成本 7.7732，fixed_best 为 7.7769，rule_best 为 7.0923。相对更优 rule_best 的配对成本增益为 -9.60%，未达到 +5% 晋级门槛。
- 决策：已由 dev 选择终点 checkpoint，但不进入 test；不运行 test 或依据 test 选 seed/checkpoint。
- 频域 pilot：fixed_best、rule_best、SR-PPO 共 18 个探测均有效；这是分轴经验诊断，不作统一 LTI/MIMO 结论。
- 限制：结果是仿真闭环结果，不构成实艇安全保证。训练期高 shield override 率需结合逐项原因解释。

## 主要产物

- `run_manifest.json`、`cases.jsonl`、`obs_schema.json`
- `teacher/summary.json`、`teacher/dataset.npz`
- `training/41002/updates.jsonl`、`training/41002/training_summary.json`、两个 checkpoint
- `evaluation/dev/bd9a57ea2828/paired_metrics.csv`、`training_summary.json`、`report.md`
- `selected_checkpoint.json`、`frequency/pilot/bd9a57ea2828/summary.json`
