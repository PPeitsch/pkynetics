"""Implementation of the Horowitz-Metzger method for kinetic analysis."""

import logging
from typing import Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def horowitz_metzger_equation(
    theta: np.ndarray, e_a: float, r: float, t_s: float
) -> np.ndarray:
    """
    Horowitz-Metzger equation for kinetic analysis.

    Args:
        theta (np.ndarray): Theta values (T - T_s).
        e_a (float): Activation energy in J/mol.
        r (float): Gas constant in J/(mol·K).
        t_s (float): Temperature of maximum decomposition rate in Kelvin.

    Returns:
        np.ndarray: y values for the Horowitz-Metzger plot.
    """
    return e_a * theta / (r * t_s**2)


def horowitz_metzger_method(
    temperature: np.ndarray, alpha: np.ndarray, n: float = 1
) -> Tuple[float, float, float, float]:
    """
    Perform Horowitz-Metzger analysis to determine kinetic parameters.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        n (float): Reaction order. Default is 1.

    Returns:
        Tuple[float, float, float, float]: Activation energy (J/mol), pre-exponential factor (min^-1),
        temperature of maximum decomposition rate (K), and R-squared value.

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values.
    """
    logger.info("Performing Horowitz-Metzger analysis")

    if len(temperature) != len(alpha):
        raise ValueError("Temperature and alpha arrays must have the same length")

    if np.any(alpha <= 0) or np.any(alpha >= 1):
        raise ValueError("Alpha values must be between 0 and 1 (exclusive)")

    if np.any(temperature <= 0):
        raise ValueError("Temperature values must be positive")

    try:
        # Find temperature of maximum decomposition rate
        SAVGOL_WINDOW = 21
        SAVGOL_POLY_ORDER = 3
        d_alpha = savgol_filter(
            np.gradient(alpha, temperature), SAVGOL_WINDOW, SAVGOL_POLY_ORDER
        )  # Smooth the derivative
        t_s = temperature[np.argmax(d_alpha)]

        # Calculate theta
        theta = temperature - t_s

        # Prepare data for fitting
        if n == 1:
            y = np.log(-np.log(1 - alpha))
        else:
            y = np.log((1 - (1 - alpha) ** (1 - n)) / (1 - n))

        # Select the most linear region
        theta_selected, y_selected = select_linear_region(theta, y, alpha)

        # Perform robust linear regression on selected data
        slope, intercept, r_value, _, _ = linregress(theta_selected, y_selected)

        # Calculate kinetic parameters
        r = 8.314  # Gas constant in J/(mol·K)
        e_a = slope * r * t_s**2  # Activation energy in J/mol
        a = np.exp(intercept + e_a / (r * t_s))  # Pre-exponential factor in min^-1

        logger.info(
            f"Horowitz-Metzger analysis completed. E_a = {e_a / 1000:.2f} kJ/mol, A = {a:.2e} min^-1, T_s = {t_s:.2f} K, R^2 = {r_value ** 2:.4f}"
        )
        return e_a, a, t_s, r_value**2

    except (ValueError, RuntimeError) as e:  # Specify expected exception types
        logger.error(f"Error in Horowitz-Metzger analysis: {str(e)}")
        raise


def select_linear_region(
    theta: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    min_conversion: float = 0.2,
    max_conversion: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the region of the data between 20-80% conversion.

    Args:
        theta (np.ndarray): Theta values (T - T_s).
        y (np.ndarray): ln(-ln(1-α)) values.
        alpha (np.ndarray): Conversion values.
        min_conversion (float): Minimum conversion value to consider (default: 0.2).
        max_conversion (float): Maximum conversion value to consider (default: 0.8).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Selected theta and y values.
    """
    # Selection based on conversion range
    mask = (alpha >= min_conversion) & (alpha <= max_conversion)
    theta_selected = theta[mask]
    y_selected = y[mask]

    # Ensure we have enough points for meaningful analysis
    if len(theta_selected) < 20:
        logger.warning(
            "Not enough points in the 20-80% conversion range. Expanding range."
        )
        min_conversion, max_conversion = 0.1, 0.9
        mask = (alpha >= min_conversion) & (alpha <= max_conversion)
        theta_selected = theta[mask]
        y_selected = y[mask]

    return theta_selected, y_selected
