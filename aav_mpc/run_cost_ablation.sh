#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
for case in T1:osqp25 T1:osqp40 T3:osqp25; do
  task=${case%%:*}
  method=${case#*:}
  /bin/python3 aav_mpc/diagnose.py --task "$task" --method "$method" \
    --speed-conditioned-cost --out "aav_mpc/data/stage0/${task}_${method}_cost.csv" \
    >> aav_mpc/data/stage0/cost_ablation.log 2>&1
done
