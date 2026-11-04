"""H15 evaluation rules shared by future mainline runners.

Historical success is evaluated at control boundaries. Mainline success uses
every integration substep and explicitly accounts for controller interventions.
Unknown intervention status is never silently interpreted as success.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping


LIMITS = {"lateral_error_m": 8.0, "depth_error_m": 5.0,
          "pitch_deg": 40.0, "speed_error_mps": 2.0}


def first_physical_violation(samples: Iterable[Mapping]):
    """Return the first (time, kind), including nonfinite measurements."""
    for sample in samples:
        for field, bound in LIMITS.items():
            value = float(sample[field])
            if not math.isfinite(value):
                return (float(sample["time_s"]), "nonfinite_" + field)
            if abs(value) > bound:
                return (float(sample["time_s"]), field)
    return None


def evaluate(*, completed: bool, period_samples: Iterable[Mapping],
             substep_samples: Iterable[Mapping] | None,
             mpc_fallback: bool, candidate_empty: bool | None,
             backup_used: bool | None, deadline_missed: bool | None,
             fallback_defined: bool | None):
    """Evaluate one full episode; missing mainline evidence yields ``None``.

    ``completed_with_backup`` is reported separately even when strict success
    fails. The caller must supply all controller event flags for mainline use.
    """
    period = list(period_samples)
    substeps = None if substep_samples is None else list(substep_samples)
    if not period:
        raise ValueError("at least one control period is required")
    sampled_first = first_physical_violation(period)
    strict_hist = bool(completed and sampled_first is None and not mpc_fallback)
    if substeps is None or any(flag is None for flag in
                               (candidate_empty, backup_used, deadline_missed,
                                fallback_defined)):
        strict_mainline = None
        substep_first = None if substeps is None else first_physical_violation(substeps)
    else:
        substep_first = first_physical_violation(substeps)
        strict_mainline = bool(completed and sampled_first is None and
                               substep_first is None and not mpc_fallback and
                               not candidate_empty and not backup_used and
                               not deadline_missed and fallback_defined)
    backup_completion = (bool(completed and backup_used and sampled_first is None and
                              substep_first is None and fallback_defined)
                         if backup_used is not None and substeps is not None and
                         fallback_defined is not None else None)
    return {"strict_hist": strict_hist,
            "strict_mainline": strict_mainline,
            "completed_with_backup": backup_completion,
            "first_sampled_violation": sampled_first,
            "first_substep_violation": substep_first}
