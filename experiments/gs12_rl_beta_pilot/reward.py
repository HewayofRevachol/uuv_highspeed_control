"""Segment score and frozen pilot reward; external J_track stays unchanged."""
from __future__ import annotations

import numpy as np


def tracking_error_squared(output, reference):
    output = np.asarray(output)
    reference = np.asarray(reference)
    diff = (reference[..., [1, 2, 5]] - output[..., [1, 2, 5]]) / [8., 5., 2.]
    return np.sum(diff**2, axis=-1)


def segment_reward(j_segment, j_f75, beta_target, previous_target,
                   segment_index, failed=False):
    if not np.isfinite(j_f75) or j_f75 < 0:
        raise ValueError("F75 reward baseline must be finite and nonnegative")
    if not np.isfinite(j_segment):
        raise ValueError("segment tracking error must be finite")
    reward = -min(float(j_segment)/max(float(j_f75), 1e-9), 20.)
    reward -= .02*(float(beta_target)-float(previous_target))**2
    if failed:
        reward -= 20.*(9-int(segment_index))
    return float(reward)


def external_j_track(arrays, window=(32., 64.)):
    times = np.asarray(arrays["time_substep_s"])
    mask = (times >= window[0]) & (times <= window[1])
    if not np.any(mask):
        raise ValueError("trajectory does not cover the external scoring window")
    return float(np.mean(tracking_error_squared(
        arrays["output_substep"][mask], arrays["reference_substep"][mask])))
