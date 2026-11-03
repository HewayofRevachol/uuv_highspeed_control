#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
mkdir -p aav_mpc/data/stage0
for case in T1:osqp25 T1:osqp40 T1:rti25 T1:preview15 T3:osqp25 T3:rti25 T3:preview15; do
  task=${case%%:*}
  method=${case#*:}
  /bin/python3 aav_mpc/diagnose.py --task "$task" --method "$method" \
    --out "aav_mpc/data/stage0/${task}_${method}.csv" \
    >> aav_mpc/data/stage0/run.log 2>&1
done
