"""JMAK (Johnson-Mehl-Avrami-Kolmogorov) method for phase transformation kinetics."""

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def jmak_equation(t: NDArray[np.float64], k: float, n: float) -> NDArray[np.float64]:
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
    return np.array(1 - np.exp(-((k * t) ** n)), dtype=np.float64)


def jmak_method(
    time: np.ndarray,
    transformed_fraction: np.ndarray,
    t_range: Optional[Tuple[float, float]] = None,
    k_init: Optional[float] = None,
    n_init: Optional[float] = None,
) -> Tuple[float, float, float]:
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
        raise ValueError(
            "Time and transformed fraction arrays must have the same shape"
        )
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
        popt, pcov = curve_fit(
            jmak_equation,
            time,
            transformed_fraction,
            p0=[k_init, n_init],
            bounds=([0, 0], [np.inf, 10]),
        )
        k, n = popt

        # Calculate R^2
        predicted = jmak_equation(time, k, n)
        ss_res = np.sum((transformed_fraction - predicted) ** 2)
        ss_tot = np.sum((transformed_fraction - np.mean(transformed_fraction)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        logger.info(
            f"JMAK analysis completed. n = {n:.3f}, k = {k:.3e}, R^2 = {r_squared:.3f}"
        )
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
    result: float = (np.log(2) / k) ** (1 / n)
    return result


def modified_jmak_equation(
    T: NDArray[np.float64], k0: float, n: float, E: float, T0: float, phi: float
) -> NDArray[np.float64]:
    """
    Modified JMAK equation for non-isothermal processes.

    Args:
        T (np.ndarray): Temperature array.
        k0 (float): Pre-exponential factor.
        n (float): Avrami exponent.
        E (float): Activation energy.
        T0 (float): Onset temperature.
        phi (float): Heating rate.

    Returns:
        np.ndarray: Transformed fraction.
    """
    R = 8.314  # Gas constant in J/(mol·K)
    return np.array(
        1 - np.exp(-((k0 / phi * (np.exp(-E / (R * T)) * (T - T0))) ** n)),
        dtype=np.float64,
    )


def fit_modified_jmak(
    T: NDArray[np.float64],
    transformed_fraction: NDArray[np.float64],
    T0: float,
    phi: float,
    E: float,
) -> Tuple[float, float, float]:
    """
    Fit the modified JMAK equation to experimental data.

    Args:
        T (np.ndarray): Temperature array.
        transformed_fraction (np.ndarray): Experimental transformed fraction.
        T0 (float): Onset temperature.
        phi (float): Heating rate.
        E (float): Activation energy (from Kissinger method).

    Returns:
        Tuple[float, float, float]: k0, n, and R-squared value.
    """

    def objective(T: NDArray[np.float64], k0: float, n: float) -> NDArray[np.float64]:
        return modified_jmak_equation(T, k0, n, E, T0, phi)

    popt, _ = curve_fit(objective, T, transformed_fraction, p0=[1e5, 1])
    k0, n = popt

    predicted = objective(T, k0, n)
    ss_res = np.sum((transformed_fraction - predicted) ** 2)
    ss_tot = np.sum((transformed_fraction - np.mean(transformed_fraction)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return k0, n, r_squared
