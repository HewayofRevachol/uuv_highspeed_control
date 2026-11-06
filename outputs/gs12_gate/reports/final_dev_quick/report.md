# GS-12 final report

Protocol: `gs12_gate_v1`. Evaluation split: `dev_quick` (tag: `None`).
Paired cases: 24. Methods: hybrid_zero, fixed_best, rule_best, ppo.

Executed: 4572 gate decisions, 22859 MPC periods, 114293 plant substeps; episode wall time 847.7 s.
Capacity-limited cases: 2; missing common-prefix RMSE rows: 0.

此报告使用 dev_quick 开发集；未运行 T6 冻结留出测试或 T7 完整频域/消融确认。频域文件仅含 pilot 指标，NaN 对应无效饱和回合。

训练记录：seed 41001，8,192 决策 / 16 次 PPO 更新，40,940 个 MPC 周期、204,677 个植物子步，墙钟 2,209.4 s；138 个训练回合中 25 个失败、缺失 0。按 dev_quick 选择 checkpoint_008192.pt（中点 1 次失败、终点 0 次失败）。评估为 24 场景×4 方法，共 96 条；其中 1 条失败、缺失 0，汇总仿真 22,859 MPC 周期、114,293 子步。版本见 run_manifest.json；选择基线 fixed:1,1,1,0 与 rule5。

## Paired results

| Method | Failures | Common-prefix RMSE | Mean cost integral | QP failures |
|---|---:|---:|---:|---:|
| hybrid_zero | 1 | 0.14097 | 1.71434 | 0 |
| fixed_best | 0 | 0.14016 | 1.84012 | 1 |
| rule_best | 0 | 0.14071 | 1.82302 | 0 |
| ppo | 0 | 0.14002 | 1.81173 | 1 |

## Decision

开发集 pilot 未达到晋级条件（相对较强调参基线的代价改善不足 5%）；保留固定/规则门控，不进入五 seed 确认。低频方向响应三方法均因饱和无效；速度方向 pilot 有效，但不能替代完整频域确认。按预设停止规则，不继续扩大 RL 训练。

The compared gates share the same prediction structure and causal estimator. Their paired difference can support a gate-decision claim only; this report does not assign the hybrid-model or estimator benefit to RL.

Common-prefix RMSE uses each case's shortest observed duration across methods.
Failures and capacity-limited cases remain in the table.
Dynamic-gate frequency matrices assembled from separate axes are diagnostic empirical responses.
They are not a single LTI MIMO FRF or a stability guarantee.

## Artifacts

- `paired_metrics.csv`: all paired case-method metrics.
- `frequency_metrics.npz`: available complex-response diagnostics.
- `families.png`, `tracking_and_failures.png`, `frequency.png`, `gate_and_innovation.png`.
