"""Implementation of the Coats-Redfern method for kinetic analysis."""

import numpy as np
from scipy.stats import linregress
from typing import Tuple


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
                         heating_rate: float, n: float = 1) -> Tuple[float, float, float]:
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

    if np.any(alpha < 0) or np.any(alpha >= 1):
        raise ValueError("Alpha values must be between 0 and 1 (exclusive)")

    if np.any(temperature <= 0):
        raise ValueError("Temperature values must be positive")

    # Prepare data for fitting
    x = 1 / temperature
    y = _prepare_y_data(alpha, temperature, n)

    # Remove any invalid points (NaN or inf) and potential outliers
    valid_mask = np.isfinite(y) & (y > -35)  # Adjust this threshold as needed
    x = x[valid_mask]
    y = y[valid_mask]

    # Perform robust linear regression
    slope, intercept, r_value, _, _ = linregress(x, y)

    # Calculate kinetic parameters
    r = 8.314  # Gas constant in J/(mol·K)
    e_a = -slope * r  # Activation energy in J/mol
    a = np.exp(intercept + np.log(heating_rate / e_a))  # Pre-exponential factor in min^-1

    return e_a, a, r_value ** 2


def coats_redfern_plot(temperature: np.ndarray, alpha: np.ndarray,
                       heating_rate: float, n: float = 1) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Generate data for Coats-Redfern plot and perform analysis.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        heating_rate (float): Heating rate in K/min.
        n (float): Reaction order. Default is 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float, float]: 
            x values (1000/T), y values, activation energy (J/mol), 
            pre-exponential factor (min^-1), and R-squared value.
    """
    e_a, a, r_squared = coats_redfern_method(temperature, alpha, heating_rate, n)

    x = 1000 / temperature  # Convert to 1000/T for better scale
    y = _prepare_y_data(alpha, temperature, n)

    # Remove any invalid points (NaN or inf) and potential outliers
    valid_mask = np.isfinite(y) & (y > -35)  # Adjust this threshold as needed
    x = x[valid_mask]
    y = y[valid_mask]

    return x, y, e_a, a, r_squared


def _prepare_y_data(alpha: np.ndarray, temperature: np.ndarray, n: float) -> np.ndarray:
    """
    Prepare y data for Coats-Redfern analysis based on reaction order.

    Args:
        alpha (np.ndarray): Conversion data.
        temperature (np.ndarray): Temperature data in Kelvin.
        n (float): Reaction order.

    Returns:
        np.ndarray: Prepared y data for Coats-Redfern plot.
    """
    eps = 1e-10  # Small value to avoid log(0)
    if n == 1:
        return np.log(np.maximum(-np.log(1 - alpha + eps), eps) / temperature ** 2)
    else:
        return np.log(np.maximum((1 - (1 - alpha + eps) ** (1 - n)) / ((1 - n) * temperature ** 2), eps))
