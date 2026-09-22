"""Small numerical helpers shared across the dilatometry module."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def _smoothing_window(
    n_total: int,
    smooth_window_fraction: float,
    min_points_smooth: int,
) -> int:
    """Odd Savitzky-Golay window length for ``n_total`` points, or 0 if too short."""
    window_length = max(min_points_smooth, int(n_total * smooth_window_fraction))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, n_total - 2)
    return window_length if window_length >= 3 else 0


def _mad_scale(values: NDArray[np.float64]) -> float:
    """Median absolute deviation, scaled to be comparable to a std."""
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def _longest_run(flags: NDArray[np.bool_]) -> Tuple[int, int]:
    """First and last index of the longest run of True in ``flags``.

    Falls back to the middle 70 % of the array if nothing is flagged.
    """
    n_total = len(flags)
    best_start, best_end, best_length = -1, -1, 0
    run_start = -1
    for i in range(n_total):
        if flags[i]:
            if run_start < 0:
                run_start = i
        elif run_start >= 0:
            if i - run_start > best_length:
                best_start, best_end, best_length = run_start, i - 1, i - run_start
            run_start = -1
    if run_start >= 0 and n_total - run_start > best_length:
        best_start, best_end = run_start, n_total - 1

    if best_start < 0:
        return int(n_total * 0.15), int(n_total * 0.85)
    return best_start, best_end


def calculate_r2(
    x: NDArray[np.float64], y: NDArray[np.float64], p: NDArray[np.float64]
) -> float:
    """Calculate R² (coefficient of determination) value for a polynomial fit."""
    if len(x) < 2:
        # Cannot calculate R² with less than 2 points
        return np.nan
    if np.all(y == y[0]):
        # If y is constant, R² is ill-defined or arguably 0 if fit is also constant, 1 if fit matches.
        # Let's return NaN as SS_tot would be zero.
        return np.nan

    y_pred = np.polyval(p, x)
    ss_res: float = float(np.sum((y - y_pred) ** 2))
    ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))

    if ss_tot < 1e-15:  # Avoid division by zero if y is effectively constant
        return 1.0 if ss_res < 1e-15 else 0.0  # Perfect fit if residuals are also zero

    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2)
