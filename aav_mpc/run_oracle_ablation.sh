#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
for task in T1 T3; do
  /bin/python3 aav_mpc/diagnose.py --task "$task" --method osqp25 --oracle-no-lag \
    --out "aav_mpc/data/stage0/${task}_osqp25_oracle_no_lag.csv" \
    >> aav_mpc/data/stage0/oracle_no_lag.log 2>&1
done
