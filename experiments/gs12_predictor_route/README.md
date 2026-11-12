# GS-12 continuous predictor route

This experiment evaluates the frozen P0→P1 predictor parameter path under W1. It reuses the existing plant, reference generator, and MPC runner. New results live in `output/`; the previous joint-route output is read-only.

## Reproduction

Run from the repository root:

```bash
python3 experiments/gs12_predictor_route/run_predictor.py verify --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/run_predictor.py prepare --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/run_predictor.py run dev_static --workers 1 --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/analyze_predictor.py --stage dev_static --out experiments/gs12_predictor_route/output
```

After M1 analysis, `selection_static.json` freezes β* and F_dev. If G1-A passes, freeze the M2 task/method list, then run the separately auditable locked runner:

```bash
python3 experiments/gs12_predictor_route/run_static_locked.py --workers 4 --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/analyze_static_locked.py
```

M3 is conditional. For this run the M2 result and endpoint-cross gate opened M3. The executed commands were:

```bash
python3 experiments/gs12_predictor_route/run_dynamic.py feature-probe --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/run_dynamic.py verify-dynamic --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/run_dynamic.py run dev_dynamic --policy R0 --workers 4 --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/analyze_dynamic.py --stage dev_dynamic --policy R0
python3 experiments/gs12_predictor_route/run_dynamic.py run dev_dynamic --policy R1 --workers 4 --out experiments/gs12_predictor_route/output
python3 experiments/gs12_predictor_route/analyze_dynamic.py --stage dev_dynamic --policy R1
python3 experiments/gs12_predictor_route/dynamic_lifecycle.py freeze-dynamic --policy R1
```

R1 is allowed once only because R0 was quality-valid but not promoted. Both rules failed the development gate. `selection_dynamic.json` records that failure, and `run_dynamic.py run test_dynamic_locked` plus `analyze_dynamic.py --stage test_dynamic_locked` reject access while no rule is frozen and promoted.

Each trajectory uses the same C4 plant, 15 m/s reference, 64 s duration, W1 weights `(1.4, 0.7)`, and the same initial state. The predictor parameters are `gain=1−0.15β`, `tau=1+β`. `run_predictor.py verify` compares β endpoints with matching saved legacy B/D traces and checks reference position/velocity consistency before any new split is prepared.

The analyzers report external tracking cost on 32–64 s, per-channel RMSE and peaks, QP/constraint quality, command slew, and per-split integrity audits. These deterministic trajectory cases are not independent statistical samples. `REPORT.md` is the readable summary; JSON/CSV/NPZ files retain the full protocol, decisions and traces.
