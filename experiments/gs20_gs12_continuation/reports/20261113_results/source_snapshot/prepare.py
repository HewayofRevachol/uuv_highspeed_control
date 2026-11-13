"""Write frozen input lists; this command does not simulate or train."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .common import CONFIG, provenance, write_json

AXES = [('east', 0.), ('depth', 0.), ('speed', 0.),
        ('coupled', 0.), ('coupled', float(np.pi/2))]


def build_sets():
    sets = {}

    def add(group, split, kind, **fields):
        rows = sets.setdefault(group, [])
        task = dict(task_id=f'{group}_{len(rows):04d}', split=split, kind=kind,
                    speed_mps=15., duration_s=144., plant='combined', amplitude=.006)
        task.update(fields)
        rows.append(task)

    for plant, axis in itertools.product(('matched', 'combined'), ('zero', 'east', 'depth', 'speed')):
        add('bridge', 'reproduction', 'bridge', plant=plant, axis=axis)
    for speed, plant, maneuver in itertools.product((12., 15., 18.),
            ('matched', 'combined', 'combined_tau'), ('straight', 'horizontal', 'depth', 'coupled')):
        add('maneuvers', 'reproduction', 'maneuver', speed_mps=speed, plant=plant,
            maneuver=maneuver, duration_s=24.)
    for split, phases in [('dev', (.31, 1.88)), ('test', (.93, 2.51))]:
        for (axis, offset), k, phase in itertools.product(AXES, CONFIG['bins'], phases):
            add(f'static_{split}', split, 'single', axis=axis, k=k,
                phase_rad=phase, depth_phase_offset=offset)
    for speed, amplitude, (axis, offset), k in itertools.product(
            (12., 18.), (.003, .006), AXES, (2, 8, 24)):
        add('range_dev', 'dev', 'single', axis=axis, k=k, speed_mps=speed,
            amplitude=amplitude, phase_rad=.31, depth_phase_offset=offset)
    for (axis, offset), k in itertools.product(AXES, CONFIG['bins']):
        add('half_dev', 'dev', 'single', axis=axis, k=k, amplitude=.003,
            phase_rad=.31, depth_phase_offset=offset)
    for plant, (axis, offset), k in itertools.product(
            ('fin_only', 'second_only', 'combined_tau'), AXES, (8, 24)):
        add('mismatch_dev', 'dev', 'single', plant=plant, axis=axis, k=k,
            phase_rad=.31, depth_phase_offset=offset)
    for split, phases, amplitudes, speeds in [
            ('dev', (.31, 1.88), (.006,), (15.,)),
            ('train', (.1, .7, 1.3, 1.9), (.003, .006), (15.,)),
            ('test', (.93, 2.51), (.003, .006), (12., 15., 18.))]:
        for axis, order, phase, amplitude, speed in itertools.product(
                ('east', 'depth', 'coupled'), ('low_high_low', 'high_low_high'),
                phases, amplitudes, speeds):
            add(f'switch_{split}', split, 'switch', axis=axis, order=order,
                phase_rad=phase, amplitude=amplitude, speed_mps=speed)
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sets = build_sets()
    for group, tasks in sets.items():
        (args.out/f'{group}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    write_json(args.out/'manifest.json', dict(task_counts={k: len(v) for k, v in sets.items()},
                                            provenance=provenance()))
    print({k: len(v) for k, v in sets.items()})


if __name__ == '__main__':
    main()
