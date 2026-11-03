#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
for case in T1:condensed25 T1:condensed40 T1:condensed_endpoint25 T1:condensed_endpoint40 T3:condensed25; do
  task=${case%%:*}
  method=${case#*:}
  OPENBLAS_NUM_THREADS=1 /bin/python3 aav_mpc/diagnose.py --task "$task" --method "$method" \
    --out "aav_mpc/data/stage2/${task}_${method}.csv" \
    >> aav_mpc/data/stage2/condensed.log 2>&1
done
