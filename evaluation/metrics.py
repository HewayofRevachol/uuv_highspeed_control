"""Shared path-level evaluation metrics."""

import numpy as np


def summarize_paths(rows):
    return {"mean_return": float(np.mean([row["return"] for row in rows])),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_position_rmse_m": float(np.mean([row["position_rmse_m"] for row in rows])),
            "mpc_fallback_fraction": float(np.mean([row["mpc_fallback_fraction"] for row in rows]))}
