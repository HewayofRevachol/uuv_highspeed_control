## Scope

This change implements the runnable E0–E2 supplemental workflow described in
`GS-11_H15_POST_REPAIR_SUPPLEMENTAL_EXPERIMENT_PLAN_20261105.md`, together with
the repaired command, target-domain, residual-label, and metric contracts that
those experiments depend on. It also tightens the conditional PPO pilot entry
point to require frozen inputs and a disjoint `T_boundary` manifest.

## Implemented entry points

- `python3 -m experiments.h15_cat_mpc.e0_contract_probe --out <project-relative-output-dir>`
  runs the 12 short contract probes (three environment profiles, compensation
  on/off, step/reversal actions) and a synthetic gate positive control.
- `python3 -m experiments.h15_cat_mpc.collect_training --out <project-relative-output-dir>`
  collects the paired 24-episode `T_fit` dataset from the frozen
  `experiments/h15_cat_mpc/manifests/t_fit.json` manifest.
- `experiments/h15_mainline/fit_normalization.py` now reads and verifies the
  manifest recorded in the collection run, and refuses non-`T_fit` inputs.
  `experiments/h15_cat_mpc/train_predictor.py` builds mode-specific causal
  residual labels and AR predictors from those training traces.
- `python3 -m experiments.h15_cat_mpc.e2_matrix --help` exposes the registered
  66-episode fixed-gate, predictor, and compensation ablation matrix, including
  paired physical/deadline outputs and same-trace offline forecast errors.
- `experiments/h15_cat_mpc/ppo_train.py` requires the frozen P1 input record,
  the 4×3 `T_fit` split, and a separate predeclared 4×3 `T_boundary` split;
  it uses 500-decision updates and records the requested optimization and
  truncation diagnostics.

## E0 verification

The E0 probe completed all 12 jobs at 40 periods each. All contract checks
passed: command schema and issued-command/queue-head alignment, period/label
alignment, target gain snapshot, and scaled nominal residual. The maximum
scaled residual on the matched-nominal probes was `0.0`. The synthetic 0/1
gate control changed the command in 40/40 cycles (maximum physical command
delta `695.4341`). The machine-readable report is
`outputs/h15_cat_mpc_postrepair_e0verified_20261105/e0_report.json` in the
working run; raw episode directories are intentionally not part of this source
commit.

The focused unit suite passed (`3 passed`), `compileall` passed for the affected
Python packages, all E0/E1/E2/PPO CLI help entry points loaded, and `git diff
--check` passed.

## Stage boundary

E1's 24 full episodes and E2's 66 full episodes have not been run. They are
longer, data-producing stages and should be run only after reviewing the E0
report and then freezing the resulting mode-specific normalization, adapter,
and predictor artifacts. E3 onward remains conditional on the experiment
plan's stop/go criteria. The PPO script deliberately fails closed until the
separate `T_boundary` dataset and frozen P1 inputs exist.
