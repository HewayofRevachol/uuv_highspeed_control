# GS-18 full-state experiment

> 本文记录已完成的历史实验。包含 MPC 权重变化的结果不能归因于 β；当前仅调整 β 的后续方案见 `PLAN_BETA_ONLY.md`，尚无该方案的新闭环结果。

## Frozen before holdout evaluation

The development comparison used `tasks_dev.jsonl`, the frozen GS-12 actor, GPU CuSADi ADMM (100 iterations, rho 10, tolerance 0.001), and the `J10` definition in `README.md`. All 12 tasks completed for every tested method. The first-round best fixed controller used beta 0.55 with original MPC weights (`J10=0.001663959`); the old RL policy scored `0.001664375`. The taskwise fixed-beta oracle gains only 0.383% over beta 0.55. Starting beta 0.55 at 0.75 changed its mean score by less than 0.001%.

The second-round development sweep kept beta 0.55 and tested attitude weight multipliers 2, 4, and 8, plus velocity multiplier 2. Their mean `J10` values were 0.001632630, 0.001577657, 0.001489482, and 0.001681021 respectively. At attitude multiplier 8, a new beta sweep selected fixed beta 0.5 (`J10=0.001488890`); the old RL policy scored 0.001491815. Thus the candidate is a fixed MPC controller, not an RL improvement. The candidate changes only the MPC theta and psi weights from 100/30 to 800/240; all other settings remain frozen.

Before opening the holdout tasks, freeze the following acceptance rule: compare this candidate with the original-weight, fixed-beta-0.55 controller on the same holdout tasks. Require at least a 5% reduction in task-equal mean `J10`, no additional incomplete episodes, QP failures, or command saturation, and no increase in any ten-state task-equal mean RMSE beyond these absolute tolerances: 0.01 m for each position, 0.002 rad for each attitude angle, 0.01 m/s for each body velocity, and 0.002 rad/s for each angular rate. These tolerances bound engineering regressions; four holdout tasks cannot establish statistical significance. `J_track` is a diagnostic only. No parameters will be changed after the holdout result.

Development artifacts: `outputs/gs18_dev_20261112/`, `outputs/gs18_beta055_start075_20261112/`, and `outputs/gs18_dev_att{2,4,8}_20261112/`, `outputs/gs18_dev_vel2_20261112/`, `outputs/gs18_dev_att8_sweep_20261112/`.

## Holdout result

The frozen candidate **did not pass** the predeclared acceptance rule. Both controllers completed all four holdout tasks with zero QP failures, command saturations, and actuator saturations. The original controller's mean `J10` was 0.007737282; the candidate's was 0.007314548, a 5.46% reduction. However, the task-equal mean east-position RMSE rose from 0.001528 m to 0.012366 m, an increase of 0.010838 m against the frozen 0.010 m limit. The other nine mean RMSE changes stayed within their limits. The old three-channel `J_track` rose from 1.647e-6 to 1.451e-5; it was not an acceptance metric.

| Holdout task | Original `J10` | Candidate `J10` | Reduction |
| --- | ---: | ---: | ---: |
| depth_17 | 0.000154675 | 0.000096728 | 37.46% |
| east_7 | 0.028579504 | 0.027302283 | 4.47% |
| speed_7 | 0.000003745 | 0.000003745 | ~0% |
| mixed | 0.002211203 | 0.001855436 | 16.09% |

This is evidence of a lower aggregate full-state error on these four tasks, but not acceptance of the proposed controller under the frozen regression gate. The small holdout set also does not support a statistical claim. No further controller parameters were tuned after the holdout evaluation. Holdout artifacts: `outputs/gs18_holdout_baseline_20261112/` and `outputs/gs18_holdout_candidate_20261112/`.

## 继续调参：取消单状态容差后

按后续指示，取消东向位置 RMSE 的 10 mm 退化门槛；`J10` 定义、任务成功条件、无 QP 失败/饱和检查与求解器容差（ADMM `1e-3`）未改。由于原 4 条留出任务此前已揭盲，本轮将它们仅作为复测，并另用预先固定的 8 条 `tasks_validation_v2.jsonl` 作验证。调参只使用 12 条开发任务。

开发集依次比较姿态权重倍率 8/16/32/64/128，再在 ×128 下比较 β=0/0.25/0.5/0.75 及位置权重倍率 0.25/1/4。最低开发 `J10` 配置为固定 β=0.75、姿态权重 ×128、位置权重 ×0.25，均值 `0.000902875`，12/12 完成，零 QP 失败和饱和。相较上一候选（β=0.5、姿态 ×8、位置 ×1，`J10=0.001488890`）降低 39.4%。开发集东向位置平均 RMSE 为 61.1 mm，明显高于原基线；用户已明确取消该项容差门槛，因此作为代价保留报告，而不据此否决候选。

| 数据集 | 基线 J10 | 候选 J10 | J10 降幅 | 完成情况 | QP 失败/饱和 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 已揭盲留出集复测（4 条） | 0.00773728 | 0.00465686 | 39.81% | 两者 4/4 | 两者均 0 |
| 新验证集（8 条） | 0.00479484 | 0.00303266 | 36.75% | 两者 8/8 | 两者均 0 |

新验证任务逐条均改善，降幅为 11.12%–64.41%。候选的十状态任务等权平均 RMSE（N,E,D,theta,psi,u,v,w,q,r）依次为 18.1 mm、111.0 mm、7.13 mm、0.165 mrad、2.12 mrad、7.02 mm/s、104.7 mm/s、17.7 mm/s、0.488 mrad/s、2.36 mrad/s。对照基线东向位置平均 RMSE 为 1.80 mm；候选为 111.0 mm。该策略明确把综合分数的改善换成了显著更大的东向位置误差。旧 `J_track` 只作诊断，不参与选择。

结论仅针对固定 MPC 参数：当前证据支持这组高姿态、低位置权重在所测验证任务上降低 `J10`，不支持 RL 优势，也不构成统计显著性结论。所有配置和逐任务轨迹在 `outputs/gs18_tuning_v2_20261112/`；搜索协议和最终配置分别为 `protocol.json`、`selected.json`。
