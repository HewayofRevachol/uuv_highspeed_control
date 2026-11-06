# GS-12 pilot report

Protocol: `gs12_gate_v1`. Evaluation split: `dev_quick` (tag: `None`).
Paired cases: 24. Methods: hybrid_zero, fixed_best, rule_best, ppo.

Executed: 4572 gate decisions, 22859 MPC periods, 114293 plant substeps; episode wall time 847.7 s.
Capacity-limited cases: 2; missing common-prefix RMSE rows: 0.

## Paired results

| Method | Failures | Common-prefix RMSE | Mean cost integral | QP failures |
|---|---:|---:|---:|---:|
| hybrid_zero | 1 | 0.14097 | 1.71434 | 0 |
| fixed_best | 0 | 0.14016 | 1.84012 | 1 |
| rule_best | 0 | 0.14071 | 1.82302 | 0 |
| ppo | 0 | 0.14002 | 1.81173 | 1 |

## Decision

首轮未达到约 5% 且不增失败的晋级条件（代价变化 -0.9%）。协议最多允许一次明确修正；当前不声称 RL 有额外价值。

The compared gates share the same prediction structure and causal estimator. Their paired difference can support a gate-decision claim only; this report does not assign the hybrid-model or estimator benefit to RL.

Common-prefix RMSE uses each case's shortest observed duration across methods.
Failures and capacity-limited cases remain in the table.
Dynamic-gate frequency matrices assembled from separate axes are diagnostic empirical responses.
They are not a single LTI MIMO FRF or a stability guarantee.

## Artifacts

- `paired_metrics.csv`: all paired case-method metrics.
- `frequency_metrics.npz`: available complex-response diagnostics.
- `families.png`, `tracking_and_failures.png`, `frequency.png`, `gate_and_innovation.png`.
