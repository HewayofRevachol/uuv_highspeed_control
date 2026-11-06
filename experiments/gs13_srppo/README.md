# GS-13 SR-PPO

这是 GS-12 混合 MPC 的安全残差 PPO 实现，包含真实 `PlantHarness` 闭环适配器、分层场景采样、教师门控 warm-start、四列约束代价、QP/边界 safety shield、truncation-aware GAE、Lagrange PPO、配对报告和频域诊断。

先阅读 `EXPERIMENT_STEPS.md`。核心入口：

```bash
python3 -m experiments.gs13_srppo.run prepare --out outputs/gs13_srppo/r0 --config experiments/gs13_srppo/config.json
python3 -m experiments.gs13_srppo.run smoke --run outputs/gs13_srppo/r0
python3 -m experiments.gs13_srppo.run teacher-grid --run outputs/gs13_srppo/r0 --limit 8
python3 -m experiments.gs13_srppo.train --run outputs/gs13_srppo/r0 --seed 41002 --decisions 8192 --update-every 512
```

训练输出是 `training/<seed>/`，评估输出是 `evaluation/<split>/<checkpoint-hash>/`。代码会拒绝修改后的 manifest、teacher 数据集或 checkpoint；不要把短 smoke run 当作算法评估结论。
