# GS-12 10-state experiment

All new source code, traces, matrices, and reports for this run live in this directory. The protocol input is read from `../gs12_analaysis/GS-12_10状态局部频响与典型工况实验方案.md`; that existing analysis directory is not an output location.

## Runner

```bash
python src/run_experiment.py verify --repo /home/prufrock/uuv_workspace --out output
python src/run_experiment.py pilot --repo /home/prufrock/uuv_workspace --out output --workers 1
python src/run_experiment.py precheck --repo /home/prufrock/uuv_workspace --out output --workers 2
python src/run_experiment.py straight-15 --repo /home/prufrock/uuv_workspace --out output --workers 2
python src/run_experiment.py straight-half-phase --repo /home/prufrock/uuv_workspace --out output --workers 2
python src/run_experiment.py straight-other-speeds --repo /home/prufrock/uuv_workspace --out output --workers 2
python src/run_experiment.py regime-precheck --repo /home/prufrock/uuv_workspace --out output --workers 2
python src/run_experiment.py turn-slope --repo /home/prufrock/uuv_workspace --out output --workers 2
python src/run_experiment.py periodic-probes --repo /home/prufrock/uuv_workspace --out output --workers 2
python src/run_experiment.py new-regime-half --repo /home/prufrock/uuv_workspace --out output --workers 2
python src/analyze_experiment.py --out output
```

`verify` freezes the experiment protocol and records input ZIP integrity, source hashes, environment, model and reference checks. If the source changes after freezing, use a new output directory. Incomplete or failed records are retained and are not overwritten. This deliverable focuses on the 10-state experiment; 10/22 comparisons are outside its report scope.

`output/runs` contains summaries; `output/traces` contains the complete 0.02 s plant logs and 0.1 s controller samples; `output/matrices/all_matrices.json` is recomputed from those traces. `output/manifest.json` records stage completion.

The 10-state deliverable is summarized in [`output/REPORT.md`](output/REPORT.md), with figures in `output/figures/` and the scoped trace audit in `output/scope_10state_integrity.json`. After scope clarification, the separate 22-state comparator campaign was stopped at 140/489 runs and is excluded from the 10-state report; its partial records and protocol remain under `output/paired22_campaign_v3/` for provenance.
