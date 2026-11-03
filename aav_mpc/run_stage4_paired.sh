#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
mkdir -p aav_mpc/data/stage4/paired
for path in DDAT_trajectories/train/s_reversal_02.csv DDAT_trajectories/train/s_reversal_03.csv; do
  name=$(basename "$path" .csv)
  for seed in 101026 101027; do
    for method in osqp25 endpoint40 condensed25 condensed_endpoint40; do
      OPENBLAS_NUM_THREADS=1 /bin/python3 aav_mpc/diagnose.py --task T1 --method "$method" \
        --path "$path" --speed 6 --seed "$seed" --randomize-plant --current-max 1 \
        --out "aav_mpc/data/stage4/paired/T1_${name}_s${seed}_${method}.csv" \
        >> aav_mpc/data/stage4/paired/run.log 2>&1
    done
  done
done
for path in trajectories/stress_15mps/cold_helix_r49_15p9mps.csv \
            trajectories/stress_15mps/coupled_turn_depth_16mps.csv \
            trajectories/stress_15mps/ramped_helix_r40_15mps.csv \
            trajectories/stress_15mps/rapid_depth_wave_15mps.csv; do
  name=$(basename "$path" .csv)
  for seed in 101026 101027; do
    for method in osqp25 curv6 cap15 condensed25; do
      OPENBLAS_NUM_THREADS=1 /bin/python3 aav_mpc/diagnose.py --task T3 --method "$method" \
        --path "$path" --speed 18 --seed "$seed" --randomize-plant --current-max 1 \
        --out "aav_mpc/data/stage4/paired/T3_${name}_s${seed}_${method}.csv" \
        >> aav_mpc/data/stage4/paired/run.log 2>&1
    done
  done
done
