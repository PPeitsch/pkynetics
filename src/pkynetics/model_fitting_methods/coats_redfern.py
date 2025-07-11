"""Implementation of the Coats-Redfern method for kinetic analysis."""

import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def coats_redfern_equation(
    t: np.ndarray, e_a: float, ln_a: float, n: float, r: float = 8.314
) -> np.ndarray:
    """
    Coats-Redfern equation for kinetic analysis.

    Args:
        t (np.ndarray): Temperature data in Kelvin.
        e_a (float): Activation energy in J/mol.
        ln_a (float): Natural logarithm of pre-exponential factor.
        n (float): Reaction order.
        r (float): Gas constant in J/(mol·K). Default is 8.314.

    Returns:
        np.ndarray: y values for the Coats-Redfern plot.
    """
    return ln_a - e_a / (r * t)


def coats_redfern_method(
    temperature: NDArray[np.float64],
    alpha: NDArray[np.float64],
    heating_rate: float,
    n: float = 1,
) -> Tuple[
    float,
    float,
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Perform Coats-Redfern analysis to determine kinetic parameters.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        heating_rate (float): Heating rate in K/min.
        n (float): Reaction order. Default is 1.

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Activation energy (J/mol)
            - Pre-exponential factor (min^-1)
            - R-squared value
            - x values for plotting (1000/T)
            - y values for plotting
            - x values used for fitting
            - y values used for fitting

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values.
    """
    if len(temperature) != len(alpha):
        raise ValueError("Temperature and alpha arrays must have the same length")

    alpha = np.clip(alpha, 0, 1)

    x = 1000 / temperature
    y = _prepare_y_data(temperature, alpha, n)

    # Focus on the most linear part (typically 20% to 80% conversion)
    mask = (alpha >= 0.2) & (alpha <= 0.8)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Remove any invalid points (NaN or inf)
    valid_mask = np.isfinite(x_filtered) & np.isfinite(y_filtered)
    x_filtered = x_filtered[valid_mask]
    y_filtered = y_filtered[valid_mask]

    # Perform robust linear regression
    slope, intercept, r_value, _, _ = linregress(x_filtered, y_filtered)

    # Calculate kinetic parameters
    r = 8.314  # Gas constant in J/(mol·K)
    e_a = -slope * r  # Activation energy in J/mol

    # Check if activation energy is physically plausible
    if e_a <= 0:
        logger.warning(
            "Calculated activation energy is negative or zero. "
            "This is physically implausible and indicates potential issues with the data or fitting."
        )
        # Use absolute value to proceed with calculation while avoiding math errors
        e_a_for_calc = abs(e_a)
    else:
        e_a_for_calc = e_a

    # Safely calculate pre-exponential factor
    try:
        ln_a = intercept + np.log(heating_rate / e_a_for_calc)
        a = np.exp(ln_a)
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.error(f"Error calculating pre-exponential factor: {str(e)}")
        a = float("nan")

    return e_a, a, r_value**2, x, y, x_filtered, y_filtered


def _prepare_y_data(
    temperature: NDArray[np.float64], alpha: NDArray[np.float64], n: float
) -> NDArray[np.float64]:
    """Prepare y data for Coats-Redfern analysis."""
    eps = 1e-10
    alpha_term = np.clip(1 - alpha, eps, 1 - eps)

    if n == 1:
        return np.array(np.log(-np.log(alpha_term) / temperature**2), dtype=np.float64)
    else:
        return np.array(
            np.log((1 - alpha_term ** (1 - n)) / ((1 - n) * temperature**2)),
            dtype=np.float64,
        )
