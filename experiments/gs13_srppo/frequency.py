"""Paired empirical frequency probes using the GS-13 closed-loop adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.gs12_gate.data_io import file_hash, immutable_json, write_npz
from experiments.gs12_gate.frequency import _case, _analysis, FREQUENCY_BINS
from .episode import run_episode
from .run import BASELINES
from .storage import artifacts, load_run
from .train import load_checkpoint


def frequency(run, checkpoint, mode='pilot', methods='fixed_best,rule_best,srppo'):
    manifest, _ = load_run(run)
    names = methods.split(',')
    if not names or set(names) - (set(BASELINES) | {'srppo'}):
        raise ValueError('unknown frequency method')
    policy = load_checkpoint(checkpoint, run) if 'srppo' in names else None
    tag = file_hash(checkpoint)[:12] if checkpoint else 'baselines'
    root = Path(run) / 'frequency' / mode / tag
    bins = (6, 24) if mode == 'pilot' else FREQUENCY_BINS
    amplitudes = (.006,) if mode == 'pilot' else (.006, .003)
    duration = 80. if mode == 'pilot' else 144.
    records = []
    for name in names:
        method = 'srppo_mean' if name == 'srppo' else BASELINES[name]
        agent = policy if name == 'srppo' else None
        zero_case = _case(f'{mode}_zero', 6, np.array([1., 0., 0.]), 0., duration)
        zero = root / name / 'zero'
        zero_result = run_episode(zero_case, method, zero, policy=agent, artifacts=artifacts(manifest))
        for amplitude in amplitudes:
            for axis, direction in enumerate(np.eye(3)):
                case = _case(f'{mode}_{amplitude}_axis_{axis}', bins[0], direction, amplitude, duration)
                case['multi_sine_bins'] = list(bins)
                folder = root / name / f'axis_{axis}_{amplitude}'
                summary = run_episode(case, method, folder, policy=agent, artifacts=artifacts(manifest))
                for bin_number in bins:
                    for start in ((16.,) if mode == 'pilot' else (16., 80.)):
                        row = {'method': name, 'axis': axis, 'amplitude': amplitude,
                               'frequency_bin': bin_number, 'window_start_s': start,
                               'valid': False, 'reason': 'physical_failure_or_incomplete',
                               'description': 'separate-axis empirical response, not an LTI MIMO FRF'}
                        if zero_result['complete'] and summary['complete']:
                            metric = _analysis(folder, zero, bin_number * 2 * np.pi / 64., direction,
                                               amplitude, start, excited_bins=bins)
                            row.update(valid=metric['valid'], reason='ok' if metric['valid'] else 'saturation_qp_or_amplitude')
                            for key in ('direction_gain', 'total_error_rms', 'nonexcited_energy', 'amplitude_ratio'):
                                row[key] = float(metric[key])
                            write_npz(folder / f'spectrum_{bin_number}_{int(start)}.npz',
                                      input_complex=metric['input_complex'], error_complex=metric['error_complex'],
                                      nonexcited_spectrum=metric['nonexcited_spectrum'],
                                      spectrum_omega=metric['spectrum_omega'])
                        records.append(row)
    immutable_json(root / 'summary.json', {'mode': mode, 'checkpoint_hash': tag, 'records': records})
    return {'probes': len(records), 'valid': sum(r['valid'] for r in records), 'output': str(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--mode', choices=('pilot', 'confirm'), default='pilot')
    parser.add_argument('--methods', default='fixed_best,rule_best,srppo')
    print(json.dumps(frequency(**vars(parser.parse_args())), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
