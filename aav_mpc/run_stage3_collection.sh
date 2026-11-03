#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
mkdir -p aav_mpc/data/stage3/raw
while read -r split family path speed seed; do
  if [[ "$speed" == 18 ]]; then task=T3; else task=T1; fi
  stem="${split}_${family}_v${speed}_s${seed}"
  OPENBLAS_NUM_THREADS=1 /bin/python3 aav_mpc/diagnose.py \
    --task "$task" --method condensed25 --path "$path" --speed "$speed" --seed "$seed" \
    --out "aav_mpc/data/stage3/raw/${stem}.csv" \
    >> aav_mpc/data/stage3/collection.log 2>&1
done <<'CASES'
train turn DDAT_trajectories/train/constant_turn_02.csv 6 20261
train turn DDAT_trajectories/train/constant_turn_03.csv 18 20261
train coupled DDAT_trajectories/train/coupled_axes_02.csv 12 20261
train coupled DDAT_trajectories/train/coupled_axes_03.csv 6 20261
train speed DDAT_trajectories/train/speed_envelope_02.csv 18 20261
train speed DDAT_trajectories/train/speed_envelope_03.csv 12 20261
val depth DDAT_trajectories/train/depth_return_02.csv 6 20262
val depth DDAT_trajectories/train/depth_return_03.csv 18 20262
test sturn DDAT_trajectories/train/s_reversal_02.csv 6 20263
test sturn DDAT_trajectories/train/s_reversal_03.csv 12 20263
test stress trajectories/stress_15mps/coupled_turn_depth_16mps.csv 18 20263
CASES
