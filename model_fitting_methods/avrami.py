"""Avrami method for isothermal crystallization kinetics."""

import numpy as np
from scipy.optimize import curve_fit
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def avrami_equation(t: np.ndarray, k: float, n: float) -> np.ndarray:
    """
    Avrami equation for isothermal crystallization.

    Args:
        t (np.ndarray): Time.
        k (float): Crystallization rate constant.
        n (float): Avrami exponent.

    Returns:
        np.ndarray: Relative crystallinity.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 1 - np.exp(-(k * t) ** n)
    return np.where(t > 0, result, 0.0)


def avrami_method(time: np.ndarray, relative_crystallinity: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform Avrami analysis for isothermal crystallization kinetics.

    Args:
        time (np.ndarray): Time data.
        relative_crystallinity (np.ndarray): Relative crystallinity data.

    Returns:
        Tuple[float, float, float]: Avrami exponent (n), rate constant (k), and coefficient of determination (R^2).

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values.

    Example:
        import numpy as np
        from pkynetics.model_fitting import avrami_method

        # Generate sample data
        time = np.linspace(0, 100, 100)
        true_n, true_k = 2.5, 0.01
        relative_crystallinity = 1 - np.exp(-(true_k * time) ** true_n)

        # Perform Avrami analysis
        n, k, r_squared = avrami_method(time, relative_crystallinity)

        print(f"Fitted values: n = {n:.3f}, k = {k:.3e}")
        print(f"R^2 = {r_squared:.3f}")
    """
    logger.info("Performing Avrami analysis")

    time = np.asarray(time)
    relative_crystallinity = np.asarray(relative_crystallinity)

    if time.shape != relative_crystallinity.shape:
        raise ValueError("Time and relative crystallinity arrays must have the same shape")

    if np.any(relative_crystallinity < 0) or np.any(relative_crystallinity > 1):
        raise ValueError("Relative crystallinity values must be between 0 and 1")

    if np.any(time < 0):
        raise ValueError("Time values must be non-negative")

    try:
        # Fit the Avrami equation to the data
        popt, pcov = curve_fit(avrami_equation, time, relative_crystallinity, p0=[1e-3, 2],
                               bounds=([0, 0], [np.inf, 10]))
        k, n = popt

        # Calculate R^2
        predicted = avrami_equation(time, k, n)
        residuals = relative_crystallinity - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((relative_crystallinity - np.mean(relative_crystallinity)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        logger.info(f"Avrami analysis completed. n = {n:.3f}, k = {k:.3e}, R^2 = {r_squared:.3f}")
        return n, k, r_squared

    except Exception as e:
        logger.error(f"Error in Avrami analysis: {str(e)}")
        raise
