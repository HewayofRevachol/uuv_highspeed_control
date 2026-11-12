# GS-12 predictor × weight joint experiment

This directory is the isolated implementation for the registered joint experiment. It reuses the existing C4 plant/MPC loop through `experiments/gs12_weight_route/runner_core.py`; it does not alter the plant, solver, or legacy outputs.

## Implemented commands

Run from the repository root:

```bash
python3 experiments/gs12_joint_route/run_joint.py --help
python3 experiments/gs12_joint_route/run_joint.py verify --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/run_joint.py prepare --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/run_joint.py run dev --workers 4 --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/analyze_joint.py --stage dev --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/run_joint.py freeze --candidate D --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/run_joint.py run new_locked --workers 4 --out experiments/gs12_joint_route/output
python3 experiments/gs12_joint_route/analyze_joint.py --stage new_locked --out experiments/gs12_joint_route/output
```

`verify` stores its reference derivative report and a short D closed-loop smoke under `output/smoke/`. `prepare` refuses to overwrite frozen protocol/task files. The run command checks source hashes before starting; the parent process is the only manifest writer, and each batch checks run ID uniqueness. `run dev-extra --set {pmid,pmid_w0,p1_weight_cross}` is available for the preregistered optional development candidates. It should only be used if the registered decision branch calls for it.

`freeze --candidate D` is accepted only after complete dev analysis recommends D under the registered rule. `freeze --candidate NONE` records a no-promotion result; in that case new_locked is not run. The analyzer reports all registered tasks/configurations, run/trace hash failures, incompleteness, early stops, QP failures, boundary events and saturation.

The M3 long-frequency and operating-condition transfer extensions are optional and are not part of the mandatory M0–M2 answer.
