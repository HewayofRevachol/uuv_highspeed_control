# H15 mainline experiment

Run all commands from the repository root with `/bin/python3`. New run artifacts
belong under `outputs/experiments/<algorithm>_<model>_<scenario>_s<seed>/<run>/`.
This directory contains experiment definitions and scripts, not measured results.

## Stage 0 / A inputs

- `aav_mpc/configs/h15.json`: historical H15 task configuration. Its bytes are
  hashed in each freeze record; do not edit it during a paired run.
- `data/datasets/h15_mpc_fix_holdout/manifest.json`: previously used repair
  holdout, now a **historical regression set**, never a new blind test set.
- `task_contract.json`: frozen evaluation terms and stop thresholds.
- `manifests/historical_holdout.json`: explicit, hash checked historical cases.

The earlier detailed-plan baseline rebuild command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m aav_mpc.h15_robust_prediction \
  --manifest data/datasets/h15_mpc_fix_holdout/manifest.json --split holdout \
  --seeds 102061 102062 102063 102064 102065 \
  --methods robust_condensed_endpoint35 \
  --out outputs/experiments/mpc_uuv_h15_b0hist_s102061/rebuild_20261104
```

The historical regression is for compatibility diagnosis only. It is not for
hyperparameter selection or final claims. The quickstart plan does not make
this rebuild a prerequisite for method development.

## Quickstart freeze and engineering smoke

The new paths and manifests are generated once; existing files with different
bytes are rejected. Test paths and manifest are made read-only. The smoke
runner checks input hashes and writes full 0.1 s and 0.02 s traces under a
separate output run. Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.freeze_quickstart \
  --out outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.run_smoke \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --out outputs/experiments/mpc_uuv_h15_quickstart_s31001/smoke_replay_20261104_2
```

Smoke uses a fixed 15 m/s and deterministic 14 m/s intent on three cases.
It has no candidate filter, response estimator, state-constrained MPC, or RL.
It cannot be treated as any formal `B0_shared`/`B2_gov`/`B3_RL` comparison.
The current evidence and open stage-gate items are in the quickstart checklist
and `outputs/reports/h15_quickstart_smoke_20261104.md`.

## Train-only observation normalization

The deployable feature contract uses state errors, measured actuator values,
issued commands and the prior command queue. It excludes true target-domain
parameters, future disturbances and oracle labels. Collect the four frozen
train geometries with three seeds, then fit and freeze the 13 feature statistics:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.collect_train_observations \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --out outputs/experiments/mpc_uuv_h15_trainobs_s31001/collect_rampfix_20261104
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.fit_normalization \
  --collection outputs/experiments/mpc_uuv_h15_trainobs_s31001/collect_rampfix_20261104 \
  --out outputs/experiments/mpc_uuv_h15_trainobs_s31001/normalization_rampfix_20261104/normalization.json
```

Both scripts reject changed frozen inputs or conflicting outputs. These
observations are a training input, not a fitted response estimator or a
formal `B0_shared` comparison.

## Reference ramp and smoke profiling

`ProgressReferenceEnv` advances the reference from the previous value by at
most 0.05 m/s per control period, including a change of intent mid-episode.
The six profiled reference-chain smoke traces and their timing summary can be
reproduced with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.run_smoke \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --out outputs/experiments/mpc_uuv_h15_quickstart_s31001/smoke_rampfix_final_20261104
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.summarize_smoke_timing \
  --smoke outputs/experiments/mpc_uuv_h15_quickstart_s31001/smoke_rampfix_final_20261104 \
  --out outputs/experiments/mpc_uuv_h15_quickstart_s31001/smoke_timing_rampfix_20261104/timing_report.md
```

The timing report separates plant integration, next-cycle MPC and measured
step wall time. It does not include estimator, candidate filtering or RL time.

## E_selected train/dev preparation

`response_estimator.py` implements the first bounded sliding-window E1 fit.
It uses measured state transitions, actual actuators, issued/queued commands
and the prior full N35 MPC command plan. It returns E0 `.85/2.0` on short, low
excitation, stale, nonfinite or out-of-distribution histories. This is an
equivalent response fit, not a measurement of true hidden plant parameters.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.collect_dev_calibration \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --out outputs/experiments/e1_uuv_h15_devcal_s32001/collect_20261104
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.probe_response_estimator \
  --collection outputs/experiments/mpc_uuv_h15_trainobs_s31001/collect_rampfix_20261104 \
  --normalization outputs/experiments/mpc_uuv_h15_trainobs_s31001/normalization_rampfix_20261104/normalization.json \
  --out outputs/experiments/e1_uuv_h15_trainprobe_s31001/probe_h35_20261104
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.calibrate_response_intervals \
  --collection outputs/experiments/e1_uuv_h15_devcal_s32001/collect_20261104 \
  --normalization outputs/experiments/mpc_uuv_h15_trainobs_s31001/normalization_rampfix_20261104/normalization.json \
  --out outputs/experiments/e1_uuv_h15_devcal_s32001/calibration_h35_20261104
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.freeze_response_estimator \
  --normalization outputs/experiments/mpc_uuv_h15_trainobs_s31001/normalization_rampfix_20261104/normalization.json \
  --calibration outputs/experiments/e1_uuv_h15_devcal_s32001/calibration_h35_20261104/calibration.json \
  --out outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_h35_20261104
```

The empirical radii use the same six dev episodes and observed future
commands in offline replay. They need independent false-feasible evaluation
before use in candidate screening. See
`outputs/reports/h15_e1_response_probe_20261104.md`.

## Candidate solve budget diagnostic

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.benchmark_candidate_solves \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --out outputs/experiments/mpc_uuv_h15_candidatebench_s31001/benchmark_20261104
```

This benchmark measures five candidate-specific N35 QP solves on 12 frozen
train states. It is a lower bound on screening time: nonlinear replay and
state/terminal checks are absent. The measured five-solve wall time exceeds
the frozen 5 ms screening budget in every state. See
`outputs/reports/h15_candidate_budget_20261104.md`.

## Offline candidate-specific screen

The offline screen solves each candidate's own N35 QP and replays its accepted
command plan from the measured state, actuator and queue at 0.02 s resolution.
It checks command changes, actuator boxes, H15 physical limits and a frozen
terminal margin under five parameter scenarios. Dev replay residual radii
provide fixed empirical tightening. The result separates physical screening
from the 5 ms online budget decision.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.run_offline_candidate_probe \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --checkpoint outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_h35_20261104/checkpoint.json \
  --config experiments/h15_mainline/candidate_filter_config.json \
  --train-collection outputs/experiments/mpc_uuv_h15_trainobs_s31001/collect_rampfix_20261104 \
  --out outputs/experiments/gov_uuv_h15_offlineprobe_s31001/probe_initialcheck_20261104
```

This is a train-state mechanism probe, not a closed-loop or online method
comparison. The B2 selector currently has no budget-accepted candidate.

## Gate0 run and cycle log contracts

`run_record.py` fixes the `run.json` top-level fields and rejects a formal
`strict_mainline` count when candidate events or timing stages are unknown.
`cycle_log.py` fixes per-period fields and requires candidate decisions,
fallback events and complete stage timings for formal B0/B2/B3 rows.
Reference-chain smoke rows keep those unrun stages explicitly `null`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.build_smoke_run_record \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --smoke outputs/experiments/mpc_uuv_h15_quickstart_s31001/smoke_rampfix_final_20261104 \
  --out outputs/experiments/mpc_uuv_h15_quickstart_s31001/schema_cycle_writer_20261104
```

This validates 1233 reference-chain periods. The Gate0 audit command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.verify_gate0 \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --normalization outputs/experiments/mpc_uuv_h15_trainobs_s31001/normalization_rampfix_20261104/normalization.json \
  --schema-run outputs/experiments/mpc_uuv_h15_quickstart_s31001/schema_cycle_writer_20261104 \
  --out outputs/experiments/mpc_uuv_h15_quickstart_s31001/gate0_audit_20261104
```

It verifies frozen configuration, manifests, train-only normalization, read-only
test inputs, run schema and cycle log schema. Gate0 passes as a contract freeze.
Actual estimator/filter/policy events and complete method timing must be
measured in later gates; diagnostic `null` fields do not represent method runs.

## Gate1 initial-state budget diagnostic

Before a formal `B0_shared` smoke run, measure the frozen five-candidate screen
on the three selected geometries:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.probe_gate1_budget \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --checkpoint outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_h35_20261104/checkpoint.json \
  --config experiments/h15_mainline/candidate_filter_config.json \
  --out outputs/experiments/gov_uuv_h15_gate1budget_s31001/probe_ratefix_20261104
```

This probe uses each geometry's reset state. Its mismatch case uses a randomized
reset, not the reference smoke's full target-domain perturbations. It is a
budget and filter-threshold diagnostic, not a formal closed-loop episode.
The candidate command-rate limits were checked against all 12 frozen train
traces, including the first command after reset. The corrected initial-state
probe has three physically accepted candidates per geometry, but none meet
the frozen 5 ms screening budget.

For diagnostic hot spots only, run the same script through `cProfile` and
store its output with the run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m cProfile \
  -o outputs/experiments/gov_uuv_h15_gate1profile_s31001/profile_20261104/profile.pstats \
  -m experiments.h15_mainline.probe_gate1_budget \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --checkpoint outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_h35_20261104/checkpoint.json \
  --config experiments/h15_mainline/candidate_filter_config.json \
  --out outputs/experiments/gov_uuv_h15_gate1profile_s31001/profile_20261104
```

Profiler timing is not an online deadline measurement.

## Gate1 offline closed-loop mechanism comparison

The user prioritizes algorithm performance over the 5 ms screening budget.
`run_offline_governor_smoke.py` therefore uses the same candidate-specific
N35 QP and nonlinear screen, but selects from physically accepted candidates
without requiring completion within 5 ms. It records the measured time and
does not claim online deployability. Gate1 currently uses the E0 `.85/2.0`
fallback with frozen dev residual radii. The three paired cases have been run;
the coupled case aborts when the fixed intent is rejected or all candidates
are empty. The paired audit, measured results and next action are in
`outputs/reports/h15_gate1_offline_paired_smoke_20261104.md` and
`docs/H15_MAINLINE_HANDOFF_20261104.md`.
