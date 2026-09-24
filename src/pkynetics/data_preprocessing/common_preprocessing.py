from typing import Dict

import numpy as np
from numpy.typing import NDArray

from .smoothing import smooth_data as smooth_data


def calculate_derivatives(
    x: np.ndarray, y: np.ndarray, smooth: bool = True
) -> Dict[str, np.ndarray]:
    """
    Calculate first and second derivatives of y with respect to x.

    Args:
        x: Independent variable values
        y: Dependent variable values
        smooth: Whether to smooth the derivatives

    Returns:
        Dict containing first and second derivatives
    """
    if smooth:
        y = smooth_data(y)

    dx = np.gradient(x)
    dy = np.gradient(y)
    d2y = np.gradient(dy)

    return {"first": dy / dx, "second": d2y / dx**2}


def baseline_correct(
    data: NDArray[np.float64], reference_indices: slice
) -> NDArray[np.float64]:
    """
    Perform baseline correction using reference region.

    Args:
        data: Input data to be corrected
        reference_indices: Slice indicating reference region

    Returns:
        Baseline corrected data
    """
    baseline = np.mean(data[reference_indices])
    return np.array(data - baseline, dtype=np.float64)
