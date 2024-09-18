"""Avrami method for isothermal crystallization kinetics."""

import numpy as np
from scipy.optimize import curve_fit
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def avrami_equation(t: np.ndarray, k: float, n: float) -> np.ndarray:
    """
    Avrami equation for isothermal crystallization.

    Args:
        t (np.ndarray): Time array.
        k (float): Crystallization rate constant.
        n (float): Avrami exponent.

    Returns:
        np.ndarray: Relative crystallinity.
    """
    return 1 - np.exp(-(k * t) ** n)


def avrami_method(time: np.ndarray, relative_crystallinity: np.ndarray,
                  t_range: Optional[Tuple[float, float]] = None) -> Tuple[float, float, float]:
    """
    Perform Avrami analysis for isothermal crystallization kinetics.

    Args:
        time (np.ndarray): Time data.
        relative_crystallinity (np.ndarray): Relative crystallinity data.
        t_range (Optional[Tuple[float, float]]): Time range for fitting. If None, use full range.

    Returns:
        Tuple[float, float, float]:
            Avrami exponent (n), rate constant (k), coefficient of determination (R^2)

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values.
    """
    logger.info("Performing Avrami analysis")

    # Input validation
    time, relative_crystallinity = np.asarray(time), np.asarray(relative_crystallinity)
    if time.shape != relative_crystallinity.shape:
        raise ValueError("Time and relative crystallinity arrays must have the same shape")
    if np.any(relative_crystallinity < 0) or np.any(relative_crystallinity > 1):
        raise ValueError("Relative crystallinity values must be between 0 and 1")
    if np.any(time < 0):
        raise ValueError("Time values must be non-negative")

    # Remove zero time values and corresponding crystallinity values
    mask = time > 0
    time, relative_crystallinity = time[mask], relative_crystallinity[mask]

    # Apply time range if specified
    if t_range is not None:
        mask = (time >= t_range[0]) & (time <= t_range[1])
        time, relative_crystallinity = time[mask], relative_crystallinity[mask]

    try:
        # Initial parameter estimates
        k_init = 1 / np.mean(time)
        n_init = 2.0  # A typical initial guess for Avrami exponent

        # Non-linear fit
        popt, pcov = curve_fit(avrami_equation, time, relative_crystallinity,
                               p0=[k_init, n_init], bounds=([0, 0], [np.inf, 10]))
        k, n = popt

        # Calculate R^2
        predicted = avrami_equation(time, k, n)
        ss_res = np.sum((relative_crystallinity - predicted) ** 2)
        ss_tot = np.sum((relative_crystallinity - np.mean(relative_crystallinity)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        logger.info(f"Avrami analysis completed. n = {n:.3f}, k = {k:.3e}, R^2 = {r_squared:.3f}")
        return n, k, r_squared

    except Exception as e:
        logger.error(f"Error in Avrami analysis: {str(e)}")
        raise


def calculate_half_time(k: float, n: float) -> float:
    """
    Calculate the half-time of crystallization.

    Args:
        k (float): Crystallization rate constant.
        n (float): Avrami exponent.

    Returns:
        float: Half-time of crystallization.
    """
    return (np.log(2) / k) ** (1 / n)
