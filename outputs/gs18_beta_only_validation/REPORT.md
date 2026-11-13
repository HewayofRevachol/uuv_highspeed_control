# GS-18 β-only results

MPC weight multipliers = 1; all methods start at beta=0.75.

| Method | Complete | Eligible | Mean J10 (complete only) |
| --- | ---: | --- | ---: |
| beta_0.55 | 8/8 | True | 0.002904083064069911 |
| beta_0.575 | 8/8 | True | 0.0029034759394840435 |
| RL_beta | 8/8 | True | 0.0029028023427134313 |

Frozen candidate vs beta=0.55:
```json
{
  "baseline": "beta_0.55",
  "candidate": "beta_0.575",
  "reduction_fraction": 0.00020905896025458226,
  "wins": 4,
  "rmse_change": {
    "N": -1.7892544860870219e-06,
    "E": -5.203818803417668e-05,
    "D": -1.6105073504504494e-05,
    "theta": -3.8804506245250503e-07,
    "psi": 7.395713922733749e-07,
    "u": 7.280385589799807e-07,
    "v": -8.651140757692799e-05,
    "w": -0.00011092664242586675,
    "q": -1.8856295921432578e-07,
    "r": 6.22407310149186e-06
  }
}
```

Passes frozen acceptance: False

See summary.json for all state/group means and comparisons; episodes.jsonl for each task's
J10, groups, ten-state RMSE/peaks and quality events; traces/ for sampled trajectories.
