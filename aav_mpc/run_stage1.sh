#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
mkdir -p aav_mpc/data/stage1
OPENBLAS_NUM_THREADS=1 /bin/python3 aav_mpc/offline_feasibility.py \
  --horizon-s 25 --starts 3 --out aav_mpc/data/stage1/t1_25s.json \
  > aav_mpc/data/stage1/t1_25s.log 2>&1
OPENBLAS_NUM_THREADS=1 /bin/python3 aav_mpc/offline_feasibility.py \
  --horizon-s 63 --starts 3 --out aav_mpc/data/stage1/t1_full.json \
  > aav_mpc/data/stage1/t1_full.log 2>&1
