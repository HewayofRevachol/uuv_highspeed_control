#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
mkdir -p aav_mpc/data/stage3/model
while [[ $(find aav_mpc/data/stage3/raw -maxdepth 1 -name 'test_*.json' | wc -l) -lt 3 ]]; do
  sleep 5
done
for seed in 17 23 31; do
  OPENBLAS_NUM_THREADS=1 /bin/python3 -m aav_mpc.tail_learning --seed "$seed" \
    --out-dir aav_mpc/data/stage3/model \
    >> aav_mpc/data/stage3/model/train.log 2>&1
done
