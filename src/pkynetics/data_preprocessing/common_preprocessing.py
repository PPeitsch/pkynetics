from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter


def smooth_data(
    data: NDArray[np.float64], window_length: Optional[int] = None, polyorder: int = 3
) -> NDArray[np.float64]:
    """
    Smooth data using Savitzky-Golay filter.

    Args:
        data: Input data to be smoothed
        window_length: Length of the filter window (must be odd). If None, automatically calculated
        polyorder: Order of the polynomial used to fit the samples

    Returns:
        Smoothed data array

    Raises:
        ValueError: If window_length is even or too large for the data
    """
    if window_length is None:
        # Calculate appropriate window length (odd number, ~5% of data length)
        window_length = min(len(data) - 2, int(len(data) * 0.05) // 2 * 2 + 1)

    if window_length % 2 == 0:
        window_length += 1

    if window_length < polyorder + 2:
        window_length = polyorder + 2 + (1 - (polyorder + 2) % 2)

    if window_length >= len(data):
        raise ValueError("Window length must be less than data length")

    return np.array(savgol_filter(data, window_length, polyorder), dtype=np.float64)


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
