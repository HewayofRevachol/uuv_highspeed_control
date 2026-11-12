"""Frozen fixed-beta and Rule-cap75 action choices."""
from __future__ import annotations

from .episode_beta import ACTION_BETAS
from .features import spectrum


def fixed_action(beta):
    try:
        return ACTION_BETAS.index(float(beta))
    except ValueError as exc:
        raise ValueError("fixed beta must be one of 0, .25, .5, .75") from exc


def rule_cap75_action(reference_window, previous_target=.75):
    """Choose from the 16 s planned reference; low energy holds prior target."""
    _, _, rho_high, low_energy = spectrum(reference_window)
    if low_energy:
        return fixed_action(previous_target)
    return 0 if rho_high < .2 else (2 if rho_high < .8 else 3)
