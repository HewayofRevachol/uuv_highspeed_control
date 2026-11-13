"""Original seven-frequency 3x3 complex FRF/SVD with paired zero removal."""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np

from .analyze import load_rows
from .common import CONFIG, ROOT, write_json
from .metrics import TRACK_SCALES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in load_rows(args.runs) if r['task']['kind'] == 'bridge']
    args.out.mkdir(parents=True, exist_ok=False)
    omega = 2*np.pi*np.array(CONFIG['bins'])/64.
    summary = []
    for method, plant in sorted({(r['method'], r['task']['plant']) for r in rows}):
        group = {r['task']['axis']: r for r in rows if r['method'] == method and r['task']['plant'] == plant}
        required = ['zero', 'east', 'depth', 'speed']
        if not all(axis in group for axis in required) or not all(
                group[a]['R2'] is not None for a in required):
            summary.append(dict(method=method, plant=plant, status='missing_or_incomplete_axes'))
            continue
        data = {a: np.load(group[a]['_directory']/group[a]['trace']) for a in required}
        baseline = data['zero']['error'][4::5, [1, 2, 5]]/TRACK_SCALES
        H = np.empty((2, len(omega), 3, 3), complex)
        for axis, name in enumerate(required[1:]):
            d = data[name]
            t = d['time'][4::5]
            e = d['error'][4::5, [1, 2, 5]]/TRACK_SCALES-baseline
            ref = d['reference'][4::5]
            inputs = np.column_stack((ref[:, 1], ref[:, 2]-20, ref[:, 5]-15))/TRACK_SCALES
            for window, (lo, hi) in enumerate(((16., 80.), (80., 144.))):
                mask = (t > lo+1e-9)&(t <= hi+1e-9)
                assert mask.sum() == 640
                projection = 2/640*np.exp(-1j*omega[:, None]*t[mask])
                output = projection@e[mask]
                input_coefficient = projection@inputs[mask, axis]
                H[window, :, :, axis] = output/input_coefficient[:, None]
        u, singular, vh = np.linalg.svd(H[1])
        v = vh.conj().transpose(0, 2, 1)
        name = f'{method.replace(":", "_")}__{plant}'
        np.savez_compressed(args.out/f'{name}.npz', H=H[1], H_period1=H[0],
                            omega_radps=omega, singular_values=singular,
                            left_vectors=u, right_vectors=v)
        item = dict(method=method, plant=plant, status='complete',
            sigma_max=singular[:, 0].tolist(),
            relative_top_gap=((singular[:, 0]-singular[:, 1])/singular[:, 0]).tolist(),
            right_max_abs=np.abs(v[:, :, 0]).tolist(),
            right_max_phase_rad=np.angle(v[:, :, 0]).tolist(),
            period_relative_change=(np.linalg.norm(H[1]-H[0], axis=(1, 2))/
                                    np.linalg.norm(H[1], axis=(1, 2))).tolist(),
            constrained=any(any(group[a]['command_saturation']+group[a]['actual_saturation']) for a in required))
        if method == 'first:0':
            archive = ROOT/'experiments/gs12_analaysis/GS-12_frequency_data.zip'
            member = f'gs12_followup/frequency_data/frf_v15_{plant}_a0.006.npz'
            with zipfile.ZipFile(archive) as z:
                old = np.load(io.BytesIO(z.read(member)))['H']
            delta = np.linalg.norm(H[1]-old, axis=(1, 2))/np.linalg.norm(old, axis=(1, 2))
            item.update(legacy_relative_matrix_difference=delta.tolist(),
                        legacy_bridge_pass=bool(np.all(delta < 1e-3)))
        summary.append(item)
    write_json(args.out/'summary.json', summary)
    print('Saved empirical preview-reference FRFs; these are not a stability certificate.')


if __name__ == '__main__':
    main()
