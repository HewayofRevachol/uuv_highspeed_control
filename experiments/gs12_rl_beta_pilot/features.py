"""Frozen 19-coordinate observation built only from measured state and plan."""
from __future__ import annotations

import numpy as np

from experiments.gs12_weight_route.runner_core import SCALE

PREVIEW_SAMPLES = 160
PREVIEW_STRIDE = 5  # 0.1 s samples in the runner's 0.02 s reference grid
OMEGA_CUT = 2.2


def spectrum(reference_window):
    ref = np.asarray(reference_window, dtype=float)
    if ref.shape != (PREVIEW_SAMPLES, 8):
        raise ValueError(f"expected (160, 8) reference preview, got {ref.shape}")
    z = np.column_stack((ref[:, 1]/8., (ref[:, 2]-20.)/5.,
                         (ref[:, 5]-15.)/2.))
    t = np.arange(PREVIEW_SAMPLES, dtype=float)
    design = np.column_stack((np.ones(PREVIEW_SAMPLES), t))
    detrended = z - design @ np.linalg.lstsq(design, z, rcond=None)[0]
    spec = np.fft.rfft(detrended*np.hanning(PREVIEW_SAMPLES)[:, None], axis=0)
    power = np.abs(spec)**2/PREVIEW_SAMPLES**2
    omega = 2*np.pi*np.fft.rfftfreq(PREVIEW_SAMPLES, d=.1)
    energy = power[omega > 0].sum(axis=0)
    high = power[omega >= OMEGA_CUT].sum(axis=0)
    rho = np.divide(high, energy, out=np.zeros(3), where=energy >= 1e-8)
    total = float(energy.sum())
    return energy, rho, float(high.sum()/total) if total >= 1e-8 else 0., total < 1e-8


def observation(session, beta_actual, beta_prev_target, previous_j=0.,
                previous_slew=0.):
    """Build observation at a high-level boundary before the next MPC solve."""
    k = session.k
    start = 5*k
    window = session.ref_grid[start:start+PREVIEW_SAMPLES*PREVIEW_STRIDE:PREVIEW_STRIDE]
    energy, rho, total_rho, low_energy = spectrum(window)
    ref = session.ref_grid[start]
    measured = session.plant.output()
    error = np.array(((ref[1]-measured[1])/8.,
                      (ref[2]-measured[2])/5.,
                      (ref[5]-measured[5])/2.))
    values = np.concatenate((
        np.clip((np.log10(np.maximum(energy, 1e-12))+12.)/12., 0., 1.),
        rho, [total_rho, float(low_energy)],
        np.clip(error, -2., 2.),
        session.plant.actual/SCALE,
        [beta_actual, beta_prev_target, k*.1/64.,
         np.clip(np.log1p(previous_j/1e-9), 0., 20.),
         np.clip(previous_slew, 0., 1.)],
    )).astype(np.float32)
    if values.shape != (19,) or not np.isfinite(values).all():
        raise ValueError("nonfinite or wrong-shaped GS-12 observation")
    return values
