"""Plot saved beta/frequency RMSE tables; never runs a closed loop."""
import argparse
import csv
from pathlib import Path

import numpy as np

from .common import CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--axis', choices=['east', 'depth', 'speed', 'coupled'], default='depth')
    parser.add_argument('--speed', type=float, default=15.)
    parser.add_argument('--amplitude', type=float, default=.006)
    parser.add_argument('--depth-phase-offset', type=float, default=0.)
    parser.add_argument('--metric', choices=['E_m', 'D_m', 'VN_mps', 'R'], default='D_m')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(args.table.open()))
    rows = [r for r in rows if r['kind'] == 'single' and r['axis'] == args.axis
            and float(r['speed_mps']) == args.speed and float(r['amplitude']) == args.amplitude
            and float(r['depth_phase_offset']) == args.depth_phase_offset and r['eligible'] == 'True']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, structure in zip(axes, ('first', 'hybrid')):
        matrix = np.full((len(CONFIG['betas']), len(CONFIG['bins'])), np.nan)
        for i, beta in enumerate(CONFIG['betas']):
            for j, k in enumerate(CONFIG['bins']):
                values = [float(r[args.metric]) for r in rows
                          if r['method'] == f'{structure}:{beta:g}' and int(r['k']) == k]
                if values:
                    matrix[i, j] = np.mean(values)
        mesh = ax.imshow(matrix, aspect='auto', origin='lower')
        ax.set_xticks(range(len(CONFIG['bins'])), [f'{2*np.pi*k/64:.3f}' for k in CONFIG['bins']])
        ax.set_yticks(range(len(CONFIG['betas'])), CONFIG['betas'])
        ax.set(title=f'{structure}: {args.axis}, {args.speed:g} m/s', xlabel='Temporal frequency [rad/s]', ylabel='Beta')
        fig.colorbar(mesh, ax=ax, label=args.metric)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)


if __name__ == '__main__':
    main()
