# GS-18 full-state GPU experiment

**当前实验入口：只调整 β。** 使用 `beta_only.py develop` 做配对扫描，使用 `beta_only.py validate` 验证冻结候选。完整计划和命令见 [PLAN_BETA_ONLY.md](PLAN_BETA_ONLY.md)。原 MPC 权重倍率固定为 1，所有方法从 β=0.75 起步，主指标仍为十状态 J10。本次仅补充代码与计划，未运行新的闭环实验。

下文保留原评估接口和历史权重实验的用法；当前 β 实验使用上面的专用入口。

Run the first, paired development comparison from the repository root:

```bash
python3 experiments/GS-18_full_state/run.py --out outputs/gs18_dev
```

The runner uses the frozen GS-12 GPU ADMM checkpoint and its saved solver settings. It compares the deterministic RL policy with fixed beta 0, 0.5, 0.55, 0.6, 0.65, and 0.75. Each fixed beta starts at its own value; the policy starts at 0.75. All methods use the same plant, initial states, tasks, and 0.02 s sampling. Use `--checkpoint` to select another compatible checkpoint, `--tasks` for a different task JSONL, and `--batch-size` to change only batching. `--methods beta_0 RL_beta` can select a subset for a diagnostic run.

For controller comparisons, `--attitude-scale` multiplies the MPC theta/psi tracking weights, `--velocity-scale` multiplies its inertial-velocity tracking weights, `--position-scale` multiplies position weights, and `--fixed-initial-beta` sets a common starting beta for fixed-beta runs. Their values are saved in `summary.json`. The first development and holdout outcomes are in `RESULTS.md`.

## 历史 MPC 权重实验（已结束，不属于当前 β 计划）

用户已取消单状态 RMSE 退化容差。继续以任务等权平均 `J10` 选择配置，保留完整率、QP 失败和饱和检查；不改变 `J10` 的定义或 QP 数值容差。各状态误差仍原样报告。

```bash
python3 experiments/GS-18_full_state/tune.py --out outputs/gs18_tuning_v2_20261112
```

先在 12 条开发任务上搜索姿态权重倍率 8/16/32/64（若仍改善则补 128），再搜索 β=0/0.25/0.5/0.75，以及位置权重倍率 0.25/1/4。每个配置独立保存轨迹和汇总，开发集结果保存为 `development.json`，选中配置保存为 `selected.json`。参数冻结后才运行旧留出集复测，以及预先固定的 8 条 `tasks_validation_v2.jsonl` 新验证任务；这两组结果不参与选参。原 `RESULTS.md` 保留历史验收结果，本轮结果另记。

The 12 development tasks are the eight original GS-12 tasks plus northward-speed changes and paired left/right eastward turns. For the new `single` tasks, the speed amplitude 0.5 means ±1 m/s northward speed variation, and the east amplitude 0.25 means ±2 m eastward displacement; both retain the original 16 s ramp. The four-task `tasks_holdout.jsonl` was opened in the first round and now serves as a retest set. The second round fixes a new eight-task `tasks_validation_v2.jsonl` before tuning and evaluates it only after candidate selection. These small deterministic sets support an engineering check, not a significance claim.

`summary.json` reports the complete-run means and the strongest fixed beta among methods that finish *every* task. `episodes.jsonl` preserves every status and task score; failed runs have no complete-trajectory `J10`. `traces/*.npz` hold sampled hull state, derived reference, actual actuator, command, beta, QP validity/residual, and step-valid mask. `J_track_recomputed` is checked against the original environment's 32–64 s three-channel score on every complete run. Command saturation, actuator saturation, command slew, and QP failure counts are separate diagnostics.

The proposed `J10` is the 0.02–64 s mean of four group means: position 0.40, attitude 0.20, body velocity 0.25, and angular rate 0.15. The state order is `[N,E,D,theta,psi,u,v,w,q,r]`; scales are `[8,8,5]` m, `[10,10]` degrees, `[2,1,1]` m/s, and `[5,5]` degrees/s. Yaw error wraps at ±π. Body velocity references come from rotating the existing inertial velocity; `q_ref=d(theta_ref)/dt` and `r_ref=cos(theta_ref)*d(psi_ref)/dt`, sampled by a second-order finite difference. `rmse` and `max_abs` are always in native SI units, including radians and radians/s. The first 0 s sample is an initial condition and is excluded from `J10`; the old score includes 32 s exactly.

Inspect group and per-state errors before changing MPC weights, add angular-rate targets only if q/r are the bottleneck, and retrain RL with ten-state observations/reward only if there is a useful task-dependent beta tradeoff. The current 19-input policy and old reward cache cannot serve as a retrained ten-state policy. A 5% mean `J10` improvement over the original best fixed beta, with no added failures or saturation, remains the aggregate improvement check; per-state RMSE is reported without a regression cutoff per the user's updated instruction. An improvement from a fixed controller must be reported as such.
