# GS-18 β-only results

MPC weight multipliers = 1; all methods start at beta=0.75.

| Method | Complete | Eligible | Mean J10 (complete only) |
| --- | ---: | --- | ---: |
| beta_0 | 4/4 | True | 0.06489340909673577 |
| beta_0.25 | 4/4 | True | 0.06386022979079865 |
| beta_0.5 | 4/4 | True | 0.06339520569324879 |
| beta_0.525 | 4/4 | True | 0.06337074852944401 |
| beta_0.55 | 4/4 | True | 0.06334958533078912 |
| beta_0.575 | 4/4 | True | 0.06333157805984094 |
| beta_0.6 | 4/4 | True | 0.06331657586927408 |
| beta_0.65 | 4/4 | True | 0.0632950537954905 |
| beta_0.75 | 4/4 | True | 0.06328261362330234 |
| RL_beta | 4/4 | True | 0.06332885701936705 |

Best fixed beta: beta_0.75

Best fixed beta vs beta=0.55:
```json
{
  "baseline": "beta_0.55",
  "candidate": "beta_0.75",
  "reduction_fraction": 0.0010571767303143575,
  "wins": 3,
  "rmse_change": {
    "N": -0.0002321644949374326,
    "E": 1.5645644897300387e-05,
    "D": 0.0019544008645512702,
    "theta": -7.795768983038555e-05,
    "psi": 5.24888120685893e-05,
    "u": 0.00010791742474458577,
    "v": -0.00461149536803207,
    "w": -0.003796111464973856,
    "q": -2.1178368454895236e-05,
    "r": 0.00029701840567797344
  }
}
```

Taskwise constant-beta diagnostic (not a bound on dynamic beta):
```json
{
  "mean_J10": 0.06277144502548508,
  "per_task_best_fixed": {
    "dyn_screen2_east_low_p0": "beta_0.75",
    "dyn_screen2_east_high_p0": "beta_0.25",
    "dyn_screen2_depth_low_p1": "beta_0.75",
    "dyn_screen2_depth_high_p1": "beta_0.75"
  },
  "reduction_fraction": 0.008077551930140814,
  "interpretation": "taskwise constant-beta diagnostic, NOT a bound on time-varying beta"
}
```

See summary.json for all state/group means and comparisons; episodes.jsonl for each task's
J10, groups, ten-state RMSE/peaks and quality events; traces/ for sampled trajectories.
