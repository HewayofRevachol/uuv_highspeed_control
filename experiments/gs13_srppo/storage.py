"""Frozen run contracts shared by the GS-13 entry points."""
from pathlib import Path

from experiments.gs12_gate.data_io import (digest, file_hash, immutable_json,
                                           immutable_jsonl, read_cases, read_json)

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]


def source_hashes():
    paths = list(PACKAGE.glob('*.py')) + [PACKAGE / 'config.json']
    for directory in ('experiments/gs12_gate', 'experiments/h15_mainline', 'simulator', 'aav_mpc'):
        paths += list((ROOT / directory).glob('*.py'))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def load_run(run):
    run = Path(run)
    manifest = read_json(run / 'run_manifest.json')
    for key, name in (('cases_hash', 'cases.jsonl'), ('schema_hash', 'obs_schema.json')):
        if manifest[key] != file_hash(run / name):
            raise ValueError(f'run {key} mismatch')
    if source_hashes() != manifest['source_hashes']:
        raise ValueError('source changed after prepare; create a new run directory')
    return manifest, read_cases(run / 'cases.jsonl')


def artifacts(manifest):
    # First pilot deliberately freezes the GS12 engineering scales. No dev/test
    # fitting is hidden here; calibrated artifacts can be added in a new run.
    from experiments.gs12_gate.features import schema
    from experiments.gs12_gate.episode import bootstrap_estimator_normalization
    return {'bootstrap': True, 'observation_scale': schema()['scale'],
            'estimator_normalization': bootstrap_estimator_normalization(),
            'residual_limits': [1., 1., 1., .1, .1],
            'teacher_gate': manifest['config']['teacher_gate']}
