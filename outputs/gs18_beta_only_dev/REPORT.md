# GS-18 β-only results

MPC weight multipliers = 1; all methods start at beta=0.75.

| Method | Complete | Eligible | Mean J10 (complete only) |
| --- | ---: | --- | ---: |
| beta_0 | 12/12 | True | 0.0017112185246728562 |
| beta_0.25 | 12/12 | True | 0.0016768008154998783 |
| beta_0.5 | 12/12 | True | 0.0016644622791348958 |
| beta_0.525 | 12/12 | True | 0.001664137394930477 |
| beta_0.55 | 12/12 | True | 0.0016639537712414534 |
| beta_0.575 | 12/12 | True | 0.0016639059321341659 |
| beta_0.6 | 12/12 | True | 0.0016639890132039573 |
| beta_0.65 | 12/12 | True | 0.0016645306498525006 |
| beta_0.75 | 12/12 | True | 0.0016669919954955545 |
| RL_beta | 12/12 | True | 0.0016643747930173841 |

Best fixed beta: beta_0.575

Best fixed beta vs beta=0.55:
```json
{
  "baseline": "beta_0.55",
  "candidate": "beta_0.575",
  "reduction_fraction": 2.8750262245469393e-05,
  "wins": 8,
  "rmse_change": {
    "N": -9.277891062520888e-07,
    "E": -5.717926096886336e-05,
    "D": -1.872588304523918e-05,
    "theta": -2.545384238505227e-07,
    "psi": 2.1966717661837942e-06,
    "u": 1.0655653914564897e-07,
    "v": -0.00011345444099598173,
    "w": -0.00010409467069528999,
    "q": -2.162074802442883e-07,
    "r": 7.99711022643132e-06
  }
}
```

Taskwise constant-beta diagnostic (not a bound on dynamic beta):
```json
{
  "mean_J10": 0.001657042722899965,
  "per_task_best_fixed": {
    "rl_dev_low_2": "beta_0",
    "rl_dev_low_8": "beta_0.75",
    "rl_dev_dhigh_26": "beta_0.75",
    "rl_dev_dhigh_30": "beta_0.75",
    "rl_dev_ehigh_26": "beta_0.6",
    "rl_dev_ehigh_30": "beta_0.25",
    "rl_dev_mixed_a": "beta_0.5",
    "rl_dev_mixed_b": "beta_0.5",
    "gs18_speed_4": "beta_0.65",
    "gs18_speed_12": "beta_0.525",
    "gs18_turn_left": "beta_0.75",
    "gs18_turn_right": "beta_0.75"
  },
  "reduction_fraction": 0.004124757957559511,
  "interpretation": "taskwise constant-beta diagnostic, NOT a bound on time-varying beta"
}
```

See summary.json for all state/group means and comparisons; episodes.jsonl for each task's
J10, groups, ten-state RMSE/peaks and quality events; traces/ for sampled trajectories.
