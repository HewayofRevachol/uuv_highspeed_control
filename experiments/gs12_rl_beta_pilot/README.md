# GS-12 β pilot

本目录基于 [RL 小批量试跑指令](../gs12_predictor_route/output/GS-12_RL小批量试跑_执行与失败分支.md)。真实植物、MPC、参考索引和诊断仍由 `gs12_weight_route/runner_core.py` 提供；该文件将原 `run_one()` 循环抽成可暂停的 `ClosedLoopSession.advance_control_cycle()`，原入口继续调用它。

- `features.py`：按冻结的 16 s 外生参考预览和当前测量生成 19 维观测。
- `episode_beta.py`：每个动作最多推进 80 个 MPC 周期；同一条 64 s 任务持续使用同一个 Plant/MPC 会话，β 每周期最多变化 0.02。固定基线要以对应 β 初始化并每段选择同一动作。
- `reward.py`：段代价、F75 归一化训练奖励和原 32–64 s 外部评分公式。
- `controllers.py`：固定 β 与 Rule-cap75 的预注册动作映射。
- `categorical_ppo.py`：复用仓库 GAE 的离散动作 actor/critic 更新；分类熵按样本取均值。
- `continuous_ppo.py`、`train_ppo_continuous.py`：连续 β 目标的高斯 PPO 训练。
- `CONTINUOUS_TRAINING_AND_ADMM.md`：连续 β PPO 的计算流程、CPU/GPU 分工及与参考 ADMM 批量环境的差异。
- `verify_loop.py`：用仓库保存的旧静态及 F00/F50/F75 轨迹核对前 8 s，再将连续的 `.75→.5` 两段与 `run_one` 对齐，同时核对外部 J 评分。

运行接线验收：

```bash
python3 -m experiments.gs12_rl_beta_pilot.verify_loop --out experiments/gs12_rl_beta_pilot/output
```

`BetaEpisode(task, j_f75)` 的 `j_f75` 必须来自该训练任务完整的 F75 轨迹；验收脚本的 `1e-9` 仅用于检查接线，不得作为正式训练缓存。离散 PPO 试跑结果见 `output/PILOT_REPORT.md`；连续动作训练是独立实验，不能覆盖该目录下的冻结结果。

连续动作训练（需要已有训练任务清单、F75 奖励缓存和 `tqdm`）：

```bash
python3 -m experiments.gs12_rl_beta_pilot.train_ppo_continuous \
  --source experiments/gs12_rl_beta_pilot/output \
  --out experiments/gs12_rl_beta_pilot/output_continuous \
  --episodes 64 --update-episodes 16 --workers 4 --device auto
```

网络输出高斯 latent，经 sigmoid 映射为连续 `beta_target ∈ [0, 0.75]`；原闭环的 β 限速仍为每个 MPC 周期 0.02。现有 GS-12 Plant/MPC 是 CPU/NumPy 会话，因此完整 episode 在 CPU worker 间并行；CUDA 可用于 PPO 批量参数更新，但不能直接让这个 MPC 求解器在 GPU 上批量运行。进度由 `tqdm` 显示，每批保存 checkpoint，结束时保存 `final_actor.pt`。建议先测量 worker 数与吞吐，不要假设小批量 GPU 更新一定比 CPU 更快。
