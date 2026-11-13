# GS-18 β-only results

MPC weight multipliers = 1; all methods start at beta=0.75.

| Method | Complete | Eligible | Mean J10 (complete only) |
| --- | ---: | --- | ---: |
| beta_0 | 4/4 | True | 0.02506004491100015 |
| beta_0.25 | 4/4 | True | 0.024290443598921154 |
| beta_0.5 | 4/4 | True | 0.02413138370414481 |
| beta_0.525 | 4/4 | True | 0.024139744642759368 |
| beta_0.55 | 4/4 | True | 0.02415181382594482 |
| beta_0.575 | 4/4 | True | 0.024167420382997934 |
| beta_0.6 | 4/4 | True | 0.024186434528863477 |
| beta_0.65 | 4/4 | True | 0.024234152198592446 |
| beta_0.75 | 4/4 | True | 0.024364925664465617 |
| RL_beta | 4/4 | True | 0.024233160562114664 |

Best fixed beta: beta_0.5

Best fixed beta vs beta=0.55:
```json
{
  "baseline": "beta_0.55",
  "candidate": "beta_0.5",
  "reduction_fraction": 0.0008459042433518471,
  "wins": 1,
  "rmse_change": {
    "N": 2.223062292070724e-05,
    "E": 0.0004600345447962783,
    "D": 0.00029843852871685474,
    "theta": 5.622767173425747e-07,
    "psi": -2.457289218663755e-05,
    "u": 5.532912078712402e-06,
    "v": 0.001045684700325969,
    "w": 0.0009843478858720944,
    "q": -9.385681120771538e-07,
    "r": -7.876665172864075e-05
  }
}
```

Taskwise constant-beta diagnostic (not a bound on dynamic beta):
```json
{
  "mean_J10": 0.023853757783951147,
  "per_task_best_fixed": {
    "dyn_screen_east_low_p0": "beta_0.75",
    "dyn_screen_east_high_p0": "beta_0.25",
    "dyn_screen_depth_low_p1": "beta_0.75",
    "dyn_screen_depth_high_p1": "beta_0.75"
  },
  "reduction_fraction": 0.011504765893137758,
  "interpretation": "taskwise constant-beta diagnostic, NOT a bound on time-varying beta"
}
```

See summary.json for all state/group means and comparisons; episodes.jsonl for each task's
J10, groups, ten-state RMSE/peaks and quality events; traces/ for sampled trajectories.
