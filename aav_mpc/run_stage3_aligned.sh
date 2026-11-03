#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
mkdir -p aav_mpc/data/stage3/model_aligned
for seed in 17 23 31; do
  OPENBLAS_NUM_THREADS=1 /bin/python3 -m aav_mpc.tail_learning --seed "$seed" \
    --log-value --balanced-episodes --select-risk --epochs 180 \
    --out-dir aav_mpc/data/stage3/model_aligned \
    >> aav_mpc/data/stage3/model_aligned/train.log 2>&1
done
