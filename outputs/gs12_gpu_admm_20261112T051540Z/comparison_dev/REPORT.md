# GS-12 GPU ADMM：MPC 与 RL+MPC 配对评估

任务：`experiments/gs12_rl_beta_pilot/output/tasks_dev.jsonl`；策略：`outputs/gs12_gpu_admm_20261112T051540Z/formal_1h/final_actor.pt`。各方法使用同一批任务和同一 GPU ADMM 环境。
RL 使用高斯均值经 sigmoid 得到的确定性 β 目标。
MPC_beta0 从 β=0 开始并保持 0；MPC_beta075 从 β=0.75 开始并保持 0.75。
J_track 是 32–64 秒窗口的原始跟踪代价，越低越好；失败轨迹不参与均值。

| 方法 | 完成/任务 | 平均 J_track | 中位 J_track |
| --- | ---: | ---: | ---: |
| MPC_beta0 | 8/8 | 4.26402e-07 | 4.65442e-07 |
| MPC_beta075 | 8/8 | 2.06999e-08 | 1.92396e-08 |
| RL_beta | 8/8 | 7.86644e-09 | 4.37438e-09 |

与 MPC_beta0 的同任务完整轨迹比较：

- MPC_beta075：8 对完整轨迹，6 对 J_track 更低，配对均值变化 -95.15%。
- RL_beta：8 对完整轨迹，6 对 J_track 更低，配对均值变化 -98.16%。

与 MPC_beta075 的同任务完整轨迹比较：

- RL_beta：8 对完整轨迹，8 对 J_track 更低，配对均值变化 -62.00%。

## 逐任务结果

| task_id | MPC β=0 | MPC β=0.75 | RL+MPC | 最低 J |
| --- | ---: | ---: | ---: | --- |
| rl_dev_low_2 | 5.079e-11 | 1.514e-10 | 1.097e-10 | MPC_beta0 |
| rl_dev_low_8 | 1.122e-10 | 6.074e-09 | 3.107e-09 | MPC_beta0 |
| rl_dev_dhigh_26 | 3.771e-07 | 5.569e-08 | 3.270e-08 | RL_beta |
| rl_dev_dhigh_30 | 8.427e-07 | 2.983e-08 | 9.178e-09 | RL_beta |
| rl_dev_ehigh_26 | 3.250e-07 | 5.725e-09 | 3.505e-09 | RL_beta |
| rl_dev_ehigh_30 | 5.807e-07 | 1.404e-08 | 3.360e-09 | RL_beta |
| rl_dev_mixed_a | 5.538e-07 | 2.444e-08 | 5.244e-09 | RL_beta |
| rl_dev_mixed_b | 7.318e-07 | 2.965e-08 | 5.724e-09 | RL_beta |

两条低频任务上，MPC β=0 的 J_track 最低；总体均值主要受高频及 mixed 任务影响。
本次仅包含 8 条 dev 任务，未运行锁定测试。
失败数和逐任务 β 目标见 `summary.json`、`episodes.csv` 与 `episodes.jsonl`。
