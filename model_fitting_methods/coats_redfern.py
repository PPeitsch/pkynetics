"""Implementation of the Coats-Redfern method for kinetic analysis."""

import numpy as np
from numpy import ndarray, dtype, floating, void
from scipy.stats import linregress
from typing import Tuple, Union, Any


def coats_redfern_equation(t: np.ndarray, e_a: float, ln_a: float, n: float, r: float = 8.314) -> np.ndarray:
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


def coats_redfern_method(temperature: np.ndarray, alpha: np.ndarray,
                         heating_rate: float, n: float = 1) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Perform Coats-Redfern analysis to determine kinetic parameters.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        heating_rate (float): Heating rate in K/min.
        n (float): Reaction order. Default is 1.

    Returns:
        Tuple[float, float, float]: Activation energy (J/mol), pre-exponential factor (min^-1), and R-squared value.

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values.
    """
    if len(temperature) != len(alpha):
        raise ValueError("Temperature and alpha arrays must have the same length")

    alpha = np.clip(alpha, 0, 1)

    x = 1000 / temperature
    y = _prepare_y_data(alpha, temperature, n)

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
    a = np.exp(intercept + np.log(heating_rate / e_a))  # Pre-exponential factor in min^-1

    return e_a, a, r_value ** 2, x, y, x_filtered, y_filtered


def _prepare_y_data(alpha: np.ndarray, temperature: np.ndarray, n: float) -> np.ndarray:
    """
    Prepare y data for Coats-Redfern analysis based on reaction order.
    """
    eps = 1e-10  # Small value to avoid log(0)
    alpha_term = np.clip(1 - alpha, eps, 1 - eps)  # Ensure we don't take log of 0 or negative values

    if n == 1:
        y = np.log(-np.log(alpha_term) / temperature ** 2)
    else:
        y = np.log((1 - alpha_term ** (1 - n)) / ((1 - n) * temperature ** 2))

    return y