"""Implementation of the Freeman-Carroll method for kinetic analysis."""

import numpy as np
from scipy.stats import linregress
from typing import Tuple

def freeman_carroll_equation(x: np.ndarray, e_a: float, n: float) -> np.ndarray:
    """
    Freeman-Carroll equation for kinetic analysis.

    Args:
        x (np.ndarray): 1/T values.
        e_a (float): Activation energy in J/mol.
        n (float): Reaction order.

    Returns:
        np.ndarray: y values for the Freeman-Carroll plot.
    """
    r = 8.314  # Gas constant in J/(mol·K)
    return -e_a / r * x + n

def freeman_carroll_method(temperature: np.ndarray, alpha: np.ndarray, 
                           time: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform Freeman-Carroll analysis to determine kinetic parameters.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        time (np.ndarray): Time data in minutes.

    Returns:
        Tuple[float, float, float]: Activation energy (J/mol), reaction order, and R-squared value.

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values.
    """
    if len(temperature) != len(alpha) or len(temperature) != len(time):
        raise ValueError("Temperature, alpha, and time arrays must have the same length")
    
    if np.any(alpha <= 0) or np.any(alpha >= 1):
        raise ValueError("Alpha values must be between 0 and 1 (exclusive)")

    if np.any(temperature <= 0):
        raise ValueError("Temperature values must be positive")

    if np.any(time < 0):
        raise ValueError("Time values must be non-negative")

    # Calculate differentials
    d_alpha = np.gradient(alpha, time)
    d_temp_inv = np.gradient(1/temperature, time)

    # Calculate terms for Freeman-Carroll plot
    eps = 1e-10  # Small value to avoid divide by zero
    y = np.log(np.maximum(d_alpha, eps)) / np.log(np.maximum(1 - alpha, eps))
    x = d_temp_inv / np.log(np.maximum(1 - alpha, eps))

    # Remove any invalid points (NaN or inf) and potential outliers
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]

    # Remove outliers
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    outlier_mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
    x = x[outlier_mask]
    y = y[outlier_mask]

    # Perform robust linear regression
    slope, intercept, r_value, _, _ = linregress(x, y)

    # Calculate kinetic parameters
    r = 8.314  # Gas constant in J/(mol·K)
    e_a = max(-slope * r, 0)  # Activation energy in J/mol, ensure it's non-negative
    n = intercept  # Reaction order

    return e_a, n, r_value**2

def freeman_carroll_plot(temperature: np.ndarray, alpha: np.ndarray, 
                         time: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Generate data for Freeman-Carroll plot and perform analysis.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        time (np.ndarray): Time data in minutes.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float, float]: 
            x values, y values, activation energy (J/mol), 
            reaction order, and R-squared value.
    """
    e_a, n, r_squared = freeman_carroll_method(temperature, alpha, time)
    
    d_alpha = np.gradient(alpha, time)
    d_temp_inv = np.gradient(1/temperature, time)

    eps = 1e-10  # Small value to avoid divide by zero
    x = d_temp_inv / np.log(np.maximum(1 - alpha, eps))
    y = np.log(np.maximum(d_alpha, eps)) / np.log(np.maximum(1 - alpha, eps))

    # Remove any invalid points (NaN or inf) and potential outliers
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]

    # Remove outliers
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    outlier_mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
    x = x[outlier_mask]
    y = y[outlier_mask]

    return x, y, e_a, n, r_squared
