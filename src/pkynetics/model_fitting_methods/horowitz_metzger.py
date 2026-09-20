"""Implementation of the Horowitz-Metzger method for kinetic analysis."""

import logging
from typing import Tuple

import numpy as np
from scipy.constants import R
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
    result: np.ndarray = np.array(e_a * theta / (r * t_s**2), dtype=np.float64)
    return result


def _horowitz_metzger_fit(
    temperature: np.ndarray, alpha: np.ndarray, heating_rate: float, n: float
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Shared Horowitz-Metzger fit behind the public method and plot helpers.

    Returns theta, y, E_a, A, T_s, R^2 and the selected points, so the two
    entry points cannot drift apart.
    """
    if len(temperature) != len(alpha):
        raise ValueError("Temperature and alpha arrays must have the same length")

    if np.any(alpha <= 0) or np.any(alpha >= 1):
        raise ValueError("Alpha values must be between 0 and 1 (exclusive)")

    if np.any(temperature <= 0):
        raise ValueError("Temperature values must be positive")

    if heating_rate <= 0:
        raise ValueError("Heating rate must be positive")

    # Find temperature of maximum decomposition rate
    SAVGOL_WINDOW = 21
    SAVGOL_POLY_ORDER = 3
    d_alpha = savgol_filter(
        np.gradient(alpha, temperature), SAVGOL_WINDOW, SAVGOL_POLY_ORDER
    )  # Smooth the derivative
    t_s = float(temperature[np.argmax(d_alpha)])

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
    slope, _, r_value, _, _ = linregress(theta_selected, y_selected)

    # Calculate kinetic parameters
    e_a = float(slope * R * t_s**2)  # Activation energy in J/mol

    # At T_s the reaction rate is at its maximum, so the Kissinger condition
    # holds there: beta * E_a / (R * T_s**2) = A * exp(-E_a / (R * T_s)).
    # Solving for A is what fixes the pre-exponential factor; the intercept
    # of the Horowitz-Metzger plot carries no information about it (for a
    # first-order reaction it is ln(-ln(1 - alpha_s)) ~ 0 by construction).
    a = float(heating_rate * e_a / (R * t_s**2) * np.exp(e_a / (R * t_s)))

    return theta, y, e_a, a, t_s, float(r_value**2), theta_selected, y_selected


def horowitz_metzger_method(
    temperature: np.ndarray, alpha: np.ndarray, heating_rate: float, n: float = 1
) -> Tuple[float, float, float, float]:
    """
    Perform Horowitz-Metzger analysis to determine kinetic parameters.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        heating_rate (float): Heating rate in K/min. Required for the
            pre-exponential factor, which is set by the rate maximum at T_s
            and not by the intercept of the plot.
        n (float): Reaction order. Default is 1.

    Returns:
        Tuple[float, float, float, float]: Activation energy (J/mol), pre-exponential factor (min^-1),
        temperature of maximum decomposition rate (K), and R-squared value.

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values.

    Notes:
        Horowitz-Metzger approximates the temperature integral by expanding
        1/T around T_s, and the approximation biases E_a high: on exact
        first-order non-isothermal data it recovers 133.6 kJ/mol for a true
        120 kJ/mol (+11 %). A is then exponential in that error, so it comes
        out over an order of magnitude high even though the formula itself
        is exact (fed the true E_a it reproduces A to 0.2 %). Use the method
        for a quick single-run estimate and a model-free method
        (Friedman, KAS, OFW) when the absolute value matters.
    """
    logger.info("Performing Horowitz-Metzger analysis")

    try:
        _, _, e_a, a, t_s, r_squared, _, _ = _horowitz_metzger_fit(
            temperature, alpha, heating_rate, n
        )

        logger.info(
            f"Horowitz-Metzger analysis completed. E_a = {e_a / 1000:.2f} kJ/mol, A = {a:.2e} min^-1, T_s = {t_s:.2f} K, R^2 = {r_squared:.4f}"
        )
        return e_a, a, t_s, r_squared

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


def horowitz_metzger_plot(
    temperature: np.ndarray, alpha: np.ndarray, heating_rate: float, n: float = 1
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Prepare data for Horowitz-Metzger plot.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        heating_rate (float): Heating rate in K/min.
        n (float): Reaction order. Default is 1.

    Returns:
        Tuple containing:
        - theta: Theta values (T - T_s)
        - y: Transformed y values
        - e_a: Activation energy in J/mol
        - a: Pre-exponential factor in min^-1
        - t_s: Temperature of maximum decomposition rate in K
        - r_squared: R-squared value
        - theta_selected: Selected theta values used for fitting
        - y_selected: Selected y values used for fitting
    """
    return _horowitz_metzger_fit(temperature, alpha, heating_rate, n)
