# Manifest status

The earlier reference to `historical_holdout.json` was a plan; that file was
not present. The historical repair cases remain in
`data/datasets/h15_mpc_fix_holdout/manifest.json` and are not a new blind test.

The quickstart freeze creates `train.json`, `dev.json`, and `test.json` once.
Each lists a path SHA-256, geometry parameters, and paired RNG seeds. The test
manifest and CSV files are read-only. The smoke runner verifies the frozen
manifest and path hashes. The current path instances are disjoint, but they
share a geometry generator and do not yet establish independent plant or
disturbance families. This limitation must be resolved before a formal final
holdout claim.
