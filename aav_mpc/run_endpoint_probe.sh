#!/usr/bin/env bash
set -euo pipefail
cd /home/prufrock/uuv_workspace
for method in endpoint40 aware_endpoint40; do
  OPENBLAS_NUM_THREADS=1 /bin/python3 aav_mpc/diagnose.py --task T1 --method "$method" \
    --out "aav_mpc/data/stage2/T1_${method}.csv" \
    >> aav_mpc/data/stage2/endpoint_probe.log 2>&1
done
