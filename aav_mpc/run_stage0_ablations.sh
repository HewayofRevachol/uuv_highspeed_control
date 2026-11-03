#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
mkdir -p aav_mpc/data/stage0
/bin/python3 aav_mpc/diagnose.py --task T1 --method osqp25 --randomize-plant \
  --out aav_mpc/data/stage0/T1_osqp25_random_body_v2.csv \
  > aav_mpc/data/stage0/random_body.log 2>&1
/bin/python3 aav_mpc/diagnose.py --task T1 --method osqp25 --current-max 1 \
  --out aav_mpc/data/stage0/T1_osqp25_current_v2.csv \
  > aav_mpc/data/stage0/current.log 2>&1
/bin/python3 aav_mpc/diagnose.py --task T1 --method osqp25 --process-noise \
  --out aav_mpc/data/stage0/T1_osqp25_noise_v2.csv \
  > aav_mpc/data/stage0/noise.log 2>&1
/bin/python3 aav_mpc/diagnose.py --task T1 --method osqp25 --nominal-body \
  --out aav_mpc/data/stage0/T1_osqp25_nominal_body_v2.csv \
  > aav_mpc/data/stage0/nominal_body.log 2>&1
