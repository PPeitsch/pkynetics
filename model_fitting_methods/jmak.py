"""JMAK (Johnson-Mehl-Avrami-Kolmogorov) method for phase transformation kinetics."""

import numpy as np
from scipy.optimize import curve_fit
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def jmak_equation(t: np.ndarray, k: float, n: float) -> np.ndarray:
    """
    Calculate the transformed fraction using the JMAK (Johnson-Mehl-Avrami-Kolmogorov) equation.

    This function implements the JMAK equation for phase transformation kinetics:
    f(t) = 1 - exp(-(k*t)^n)

    Args:
        t (np.ndarray): Array of time values.
        k (float): Rate constant, related to the overall transformation rate.
        n (float): JMAK exponent, related to nucleation and growth mechanisms.

    Returns:
        np.ndarray: Array of transformed fraction values corresponding to input times.
    """
    return 1 - np.exp(-(k * t) ** n)


def jmak_method(time: np.ndarray, transformed_fraction: np.ndarray,
                t_range: Optional[Tuple[float, float]] = None,
                k_init: Optional[float] = None,
                n_init: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Fit the JMAK (Johnson-Mehl-Avrami-Kolmogorov) model to transformation data.

    This function performs non-linear regression to fit the JMAK equation to
    experimental phase transformation data, determining the JMAK exponent (n)
    and rate constant (k).

    Args:
        time: Array of time values.
        transformed_fraction: Array of corresponding transformed fraction values.
        t_range: Optional time range for fitting. If None, uses the full range.
        k_init: Initial guess for rate constant k. If None, estimated from data.
        n_init: Initial guess for JMAK exponent n. If None, defaults to 2.0.

    Returns:
        JMAK exponent (n), rate constant (k), and coefficient of determination (R^2).

    Raises:
        ValueError: If input data is invalid or inconsistent.
    """
    logger.info("Performing JMAK analysis")

    # Input validation
    time, transformed_fraction = np.asarray(time), np.asarray(transformed_fraction)
    if time.shape != transformed_fraction.shape:
        raise ValueError("Time and transformed fraction arrays must have the same shape")
    if np.any(transformed_fraction < 0) or np.any(transformed_fraction > 1):
        raise ValueError("Transformed fraction values must be between 0 and 1")
    if np.any(time < 0):
        raise ValueError("Time values must be non-negative")

    # Remove zero time values and corresponding transformed fraction values
    mask = time > 0
    time, transformed_fraction = time[mask], transformed_fraction[mask]

    # Apply time range if specified
    if t_range is not None:
        mask = (time >= t_range[0]) & (time <= t_range[1])
        time, transformed_fraction = time[mask], transformed_fraction[mask]

    try:
        # Initial parameter estimates
        k_init = k_init if k_init is not None else 1 / np.mean(time)
        n_init = n_init if n_init is not None else 2.0

        # Non-linear fit
        popt, pcov = curve_fit(jmak_equation, time, transformed_fraction,
                               p0=[k_init, n_init], bounds=([0, 0], [np.inf, 10]))
        k, n = popt

        # Calculate R^2
        predicted = jmak_equation(time, k, n)
        ss_res = np.sum((transformed_fraction - predicted) ** 2)
        ss_tot = np.sum((transformed_fraction - np.mean(transformed_fraction)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        logger.info(f"JMAK analysis completed. n = {n:.3f}, k = {k:.3e}, R^2 = {r_squared:.3f}")
        return n, k, r_squared

    except Exception as e:
        logger.error(f"Error in JMAK analysis: {str(e)}")
        raise


def jmak_half_time(k: float, n: float) -> float:
    """
    Calculate the half-time of transformation using the JMAK (Johnson-Mehl-Avrami-Kolmogorov) model.

    This function computes the time at which the transformed fraction reaches 0.5 (50%)
    according to the JMAK equation. It is derived from solving the equation:
    0.5 = 1 - exp(-(k * t)^n) for t.

    Args:
        k (float): JMAK rate constant, related to the overall transformation rate.
        n (float): JMAK exponent, which can provide information about the nucleation and growth mechanisms.

    Returns:
        float: The half-time of transformation (t_0.5), i.e., the time at which
               the transformed fraction is 0.5 according to the JMAK model.

    Note:
        The units of the returned half-time will be consistent with the units used
        for the rate constant k. Ensure that k and n are obtained from the same
        JMAK analysis for meaningful results.
    """
    return (np.log(2) / k) ** (1 / n)
