"""Implementation of the Horowitz-Metzger method for kinetic analysis."""

import numpy as np
from scipy.stats import linregress
from scipy.signal import savgol_filter
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def horowitz_metzger_equation(theta: np.ndarray, e_a: float, r: float, t_s: float) -> np.ndarray:
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

def horowitz_metzger_method(temperature: np.ndarray, alpha: np.ndarray, n: float = 1) -> Tuple[float, float, float, float]:
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
        d_alpha = savgol_filter(np.gradient(alpha, temperature), 21, 3)  # Smooth the derivative
        t_s = temperature[np.argmax(d_alpha)]

        # Calculate theta
        theta = temperature - t_s

        # Prepare data for fitting
        if n == 1:
            y = np.log(np.log(1 / (1 - alpha)))
        else:
            y = np.log((1 - (1 - alpha)**(1-n)) / (1 - n))

        # Remove any invalid points (NaN or inf) and potential outliers
        valid_mask = np.isfinite(y) & (theta != 0)
        theta = theta[valid_mask]
        y = y[valid_mask]

        # Perform robust linear regression
        slope, intercept, r_value, _, _ = linregress(theta, y)

        # Calculate kinetic parameters
        r = 8.314  # Gas constant in J/(mol·K)
        e_a = slope * r * t_s**2  # Activation energy in J/mol
        a = np.exp(intercept + e_a / (r * t_s))  # Pre-exponential factor in min^-1

        logger.info(f"Horowitz-Metzger analysis completed. E_a = {e_a/1000:.2f} kJ/mol, A = {a:.2e} min^-1, T_s = {t_s:.2f} K, R^2 = {r_value**2:.4f}")
        return e_a, a, t_s, r_value**2

    except Exception as e:
        logger.error(f"Error in Horowitz-Metzger analysis: {str(e)}")
        raise

def horowitz_metzger_plot(temperature: np.ndarray, alpha: np.ndarray, n: float = 1) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Generate data for Horowitz-Metzger plot and perform analysis.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        n (float): Reaction order. Default is 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float, float, float]: 
            x values (theta), y values, activation energy (J/mol), 
            pre-exponential factor (min^-1), temperature of maximum decomposition rate (K),
            and R-squared value.
    """
    e_a, a, t_s, r_squared = horowitz_metzger_method(temperature, alpha, n)
    
    theta = temperature - t_s
    if n == 1:
        y = np.log(np.log(1 / (1 - alpha)))
    else:
        y = np.log((1 - (1 - alpha)**(1-n)) / (1 - n))

    # Remove any invalid points (NaN or inf) and potential outliers
    valid_mask = np.isfinite(y) & (theta != 0)
    theta = theta[valid_mask]
    y = y[valid_mask]

    return theta, y, e_a, a, t_s, r_squared
