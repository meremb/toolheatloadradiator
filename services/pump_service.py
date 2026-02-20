"""
services/pump_service.py
========================
Pump-curve interpolation and operating-point finding.

Pure functions with no Dash dependencies.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def interpolate_curve(
    points: List[Tuple[float, float]],
    x_vals: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation over a pump or system curve.

    Returns NaN for x values outside the range of points.
    """
    if not points:
        return np.full_like(x_vals, np.nan, dtype=float)
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    return np.interp(x_vals, xs, ys, left=np.nan, right=np.nan)


def derive_system_curve_k(Q_total_kgph: float, branch_kpa: float) -> float:
    """
    Estimate K in ΔP_sys = K · Q² from the current operating point.

    Parameters
    ----------
    Q_total_kgph : Total system flow          [kg/h]
    branch_kpa   : Maximum branch pressure    [kPa]
    """
    if Q_total_kgph <= 0:
        return 0.0
    return max(branch_kpa, 0.0) / max(Q_total_kgph, 1e-6) ** 2


def find_operating_point(
    Q_grid: np.ndarray,
    pump_kpa: np.ndarray,
    K_sys: float,
) -> Tuple[Optional[float], Optional[float], List[str]]:
    """
    Find the intersection of the pump curve and the system curve (K · Q²).

    Returns
    -------
    (Q_star, ΔP_star, warnings)
        Q_star   : Operating flow  [kg/h]  or None if not found
        ΔP_star  : Operating head  [kPa]   or None if not found
        warnings : List of diagnostic messages
    """
    warnings: List[str] = []
    sys_kpa = K_sys * (Q_grid ** 2)
    diff    = pump_kpa - sys_kpa
    mask    = ~np.isnan(pump_kpa) & ~np.isnan(sys_kpa)

    if mask.sum() < 2:
        warnings.append("Insufficient data to find pump operating point.")
        return None, None, warnings

    sign            = np.sign(diff[mask])
    sign_change_idx = np.where(np.diff(sign) != 0)[0]

    if len(sign_change_idx) == 0:
        if np.all(diff[mask] > 0):
            warnings.append(
                "Pump curve stays above system curve – operating point may be "
                "at a higher flow than the provided curve covers."
            )
        else:
            warnings.append(
                "System curve stays above pump curve – pump likely insufficient."
            )
        return None, None, warnings

    Qm, Dm = Q_grid[mask], diff[mask]
    i       = sign_change_idx[0]
    x0, x1 = Qm[i], Qm[i + 1]
    y0, y1 = Dm[i], Dm[i + 1]
    q_star  = x0 - y0 * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x0
    dp_star = float(np.interp(q_star, Qm, pump_kpa[mask]))

    return float(q_star), dp_star, warnings
