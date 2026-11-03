#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
mkdir -p aav_mpc/data/stage2
for case in T1:aware25 T1:aware40 T3:aware25; do
  task=${case%%:*}
  method=${case#*:}
  OPENBLAS_NUM_THREADS=1 /bin/python3 aav_mpc/diagnose.py --task "$task" --method "$method" \
    --out "aav_mpc/data/stage2/${task}_${method}_v2.csv" \
    >> aav_mpc/data/stage2/run_v2.log 2>&1
done
