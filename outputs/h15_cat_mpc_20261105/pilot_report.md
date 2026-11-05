# CAT-MPC H15 pilot report

Date: 2026-11-05  
Python: `/bin/python3`  
Source commit at input freeze: `46b329c1d70681f389ae82b030ac4ae78d782434`  
Device: CPU, 16 logical CPUs reported, CUDA unavailable.

## Decision

P0, P1, and the single-seed P2 pilot are complete. The 22-state continuous gate changes the condensed MPC point model and commands, and the train-only AR(1) residual predictor beats persistence on the paired dev traces at 1, 5, and 25 control periods. A4, the fixed `(0.5, 0.5)` gate, had the strongest deterministic pilot result. PPO seed `41001` did not meet any P2 exploration condition and did not outperform A4 consistently. Stop here: do not expand to 8,000 decisions, additional PPO seeds, holdout evaluation, or P4.

## P0: interface and timing evidence

- Existing actuator-aware dynamics evaluated through the new symbolic input interface match the pre-change implementation exactly on the parity snapshot: maximum absolute difference in `f`, `A`, and `B` was `0`.
- On the same state/reference/queue, B0 and CAT-MPC `(0,0)` produced identical first command and 22-state prediction (both maximum differences `0`); OSQP status was `solved`.
- `alpha_d=1` with zero residual and `alpha_theta=1` with `theta_hat=theta0` both reproduce B0. Nonzero residual changes the hull state while leaving actual-actuator and queue states unchanged.
- The simple, coupled, and target-mismatch wiring probes each ran 40 cycles. The gated model changed the first command on every cycle. These probes used a small synthetic disturbance only to exercise the numerical input and are not performance results.

Evidence: `p0_probe_final/interface_check.json`, `p0_probe_final/p0_probe.json`.

## P1: predictor and fixed gates

The response estimator normalization used 2,388 periods from 12 train episodes (4 train geometries × 3 seeds). Residual labels and AR(1) fitting remained split by episode. Dev contains 6 paired geometry-seed units; A0–A4 ran 30 episodes with full 0.02 s substep logs and a shared seed per pair.

The average dev MAE across 30 episodes was:

| Horizon | AR(1), `[u̇,v̇,ẇ,q̇,ṙ]` | Persistence, `[u̇,v̇,ẇ,q̇,ṙ]` |
| --- | --- | --- |
| 1 | `[0.0587, 0.0400, 0.0316, 0.00485, 0.00606]` | `[0.0592, 0.0401, 0.0317, 0.00489, 0.00610]` |
| 5 | `[0.1392, 0.1257, 0.0787, 0.01186, 0.01873]` | `[0.1653, 0.1332, 0.0874, 0.01332, 0.02024]` |
| 25 | `[0.1470, 0.1364, 0.0953, 0.01364, 0.01938]` | `[0.1699, 0.1717, 0.1282, 0.01859, 0.02573]` |

`strict_hist / physical strict / strict_mainline` completions out of 6 were A0 `5/5/4`, A1 `5/5/4`, A2 `5/5/3`, A3 `5/5/3`, and A4 `6/6/5`. A3 had one MPC fallback. A4 improved one strict-history pair over A0 with no paired worsening. These are small pilot counts, not a statistical claim.

Evidence: `train_collection/`, `adapter/normalization.json`, `predictor/predictor.json`, `p1_dev_strict/summary.json`, `p1_dev_strict/paired_analysis.json`.

## P2: one PPO seed

Seed `41001` completed 4,000 gate decisions (about 20,000 MPC periods). Adapter and predictor stayed frozen. PPO used `lr=3e-4`, clip `0.2`, 4 epochs, minibatch 64, gradient clipping 1, and a 5-period action hold. Checkpoints are saved at 0, 1,000, 2,000, and 4,000 decisions.

Training mean decision return stayed approximately `7.371` from 1,000 through 4,000 decisions. At 4,000, applied alpha mean was `[0.421, 0.453]` with SD `[0.156, 0.166]`; it did not saturate, but remained close to the fixed A4 gate.

Each checkpoint completed 5/6 dev episodes under strict mainline accounting, with no backup or MPC fallback. Against A0, each checkpoint improved one paired strict-mainline unit and worsened none, below the required two improvements. On common successes, normalized tracking RMSE fell about 4.0–4.4%, short of 10%; mean absolute tangential speed error increased slightly. Against A4, each checkpoint had one paired improvement and one worsening. P2 therefore fails its exploration gate; the learning curve also shows no sustained improvement.

Evidence: `p2_seed41001/training_metrics.json`, `p2_dev_checkpoints/summary.json`, `p2_dev_checkpoints/p2_analysis.json`, `p2_dev_checkpoints/p2_diagnostics.png`.

## Limits

This is a six-unit dev pilot, not a significance test, safety result, or holdout generalization claim. The GPU path was not used. The predictor uses a small AR(1) model. The 22-state interface does not model hidden second-order actuator rates. P3 and P4 were not run because P2 did not pass.
