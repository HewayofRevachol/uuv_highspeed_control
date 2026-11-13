"""Create the reduced, frozen B-F task set; does not run simulations."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .common import provenance, write_json


def build_sets():
    sets = {}

    def add(group, split, kind, **fields):
        rows = sets.setdefault(group, [])
        task = dict(task_id=f'{group}_{len(rows):03d}', split=split, kind=kind,
                    speed_mps=15., duration_s=64., plant='combined', amplitude=.006)
        task.update(fields)
        rows.append(task)

    # Sixteen short single-frequency cases: two plants, four axes, two bands.
    for plant, axis, k in itertools.product(
            ('matched', 'combined'), ('east', 'depth', 'speed', 'coupled'), (8, 24)):
        add('beta_dev', 'dev', 'single', plant=plant, axis=axis, k=k,
            phase_rad=.31, depth_phase_offset=float(np.pi/2) if axis == 'coupled' else 0.)

    # Check selected beta at held-out speeds on the two most informative axes.
    for speed, axis in itertools.product((12., 18.), ('depth', 'coupled')):
        add('speed_check', 'dev', 'single', speed_mps=speed, axis=axis, k=24,
            phase_rad=.93, depth_phase_offset=float(np.pi/2) if axis == 'coupled' else 0.)

    # Four 144-second switch cases retain both frequency-order directions.
    for axis, order in itertools.product(('east', 'depth'), ('low_high_low', 'high_low_high')):
        add('switch_dev', 'dev', 'switch', axis=axis, order=order,
            phase_rad=.93, duration_s=144.)

    # Eight distinct training references; PPO samples these for 16 episodes.
    for axis, order, phase in itertools.product(
            ('east', 'depth'), ('low_high_low', 'high_low_high'), (.1, .7)):
        add('switch_train', 'train', 'switch', axis=axis, order=order,
            phase_rad=phase, duration_s=144.)

    # Eight frozen holdout cases: new phase, lower amplitude and two speeds.
    for axis, order, speed in itertools.product(
            ('east', 'depth'), ('low_high_low', 'high_low_high'), (12., 18.)):
        add('switch_test', 'test', 'switch', axis=axis, order=order, speed_mps=speed,
            phase_rad=2.51, amplitude=.003, duration_s=144.)
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sets = build_sets()
    for group, tasks in sets.items():
        (args.out/f'{group}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    counts = {key: len(value) for key, value in sets.items()}
    write_json(args.out/'manifest.json', dict(task_counts=counts, provenance=provenance()))
    print(counts)


if __name__ == '__main__':
    main()
