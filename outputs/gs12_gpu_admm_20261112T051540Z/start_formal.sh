#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
exec python3 -u -m experiments.gs12_rl_beta_gpu_admm.train \
  --source outputs/gs12_gpu_admm_20261112T051540Z/formal_source \
  --out outputs/gs12_gpu_admm_20261112T051540Z/formal_1h \
  --episodes 2560 --update-episodes 64 --predictor cusadi \
  --admm-iterations 100 --admm-rho 10 --admm-tolerance 0.001 \
  --seed 20261112 \
  >> outputs/gs12_gpu_admm_20261112T051540Z/formal.log 2>&1
