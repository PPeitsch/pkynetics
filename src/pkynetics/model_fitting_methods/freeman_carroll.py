from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from scipy.stats import linregress


def freeman_carroll_equation(
    x: NDArray[np.float64], e_a: float, n: float
) -> NDArray[np.float64]:
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
    return np.array(-e_a / r * x + n, dtype=np.float64)


def smooth_data(
    data: NDArray[np.float64], window_length: int = 21, polyorder: int = 3
) -> NDArray[np.float64]:
    """Apply Savitzky-Golay filter to smooth data."""
    return np.array(savgol_filter(data, window_length, polyorder), dtype=np.float64)


def safe_log(x: NDArray[np.float64], min_value: float = 1e-10) -> NDArray[np.float64]:
    """Safely compute logarithm, avoiding log(0) and negative values."""
    return np.array(np.log(np.maximum(x, min_value)), dtype=np.float64)


def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Safely divide two arrays, handling division by zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = fill_value
    return result


def freeman_carroll_method(
    temperature: np.ndarray, alpha: np.ndarray, time: np.ndarray
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Freeman-Carroll analysis to determine kinetic parameters.

    Args:
        temperature (np.ndarray): Temperature data in K
        alpha (np.ndarray): Conversion data
        time (np.ndarray): Time data

    Returns:
        Tuple containing:
        - Activation energy (J/mol)
        - Reaction order
        - R-squared value
        - x values for plotting
        - y values for plotting
        - filtered x values
        - filtered y values

    Raises:
        ValueError: If inputs have different lengths or contain invalid values
    """
    # Input validation
    if not (len(temperature) == len(alpha) == len(time)):
        raise ValueError("All input arrays must have the same length")

    if np.any(temperature <= 0):
        raise ValueError("Temperature values must be positive")

    if np.any(time < 0):
        raise ValueError("Time values must be non-negative")

    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("Conversion values must be between 0 and 1")

    if np.all(alpha == 0) or np.all(alpha == 1):
        raise ValueError("Conversion values cannot be all zeros or ones")

    # Smooth the data
    alpha_smooth = smooth_data(alpha)
    temp_smooth = smooth_data(temperature)

    # Calculate differentials
    d_alpha_dt = np.gradient(alpha_smooth, time)
    d_1_T = np.gradient(1 / temp_smooth, time)

    # Calculate terms for Freeman-Carroll plot
    ln_1_minus_alpha = safe_log(1 - alpha_smooth)
    d_ln_1_minus_alpha = np.gradient(ln_1_minus_alpha, time)

    # Compute y and x values safely
    y = safe_divide(np.gradient(safe_log(d_alpha_dt), time), d_ln_1_minus_alpha)
    x = safe_divide(d_1_T, d_ln_1_minus_alpha)

    # Focus on the most relevant part of the data
    reaction_mask = (
        (alpha_smooth >= 0.2) & (alpha_smooth <= 0.8) & np.isfinite(x) & np.isfinite(y)
    )

    x_filtered = x[reaction_mask]
    y_filtered = y[reaction_mask]

    if len(x_filtered) < 2:
        raise ValueError("Insufficient valid data points for analysis")

    # Remove outliers using IQR method
    q1, q3 = np.percentile(y_filtered, [25, 75])
    iqr = q3 - q1
    outlier_mask = (y_filtered >= q1 - 1.5 * iqr) & (y_filtered <= q3 + 1.5 * iqr)
    x_filtered = x_filtered[outlier_mask]
    y_filtered = y_filtered[outlier_mask]

    # Perform linear regression
    slope, intercept, r_value, _, _ = linregress(x_filtered, y_filtered)

    # Calculate kinetic parameters
    r = 8.314  # Gas constant in J/(mol·K)
    e_a = -slope * r  # Activation energy in J/mol
    n = intercept  # Reaction order

    return e_a, n, r_value**2, x, y, x_filtered, y_filtered


def plot_diagnostic(
    time: np.ndarray, alpha: np.ndarray, temperature: np.ndarray
) -> None:
    d_alpha_dt = np.gradient(alpha, time)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(time, alpha, label="α")
    ax1.plot(time, d_alpha_dt, label="dα/dt")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Conversion (α) and Rate (dα/dt)")
    ax1.legend()

    ax2.plot(1 / temperature, safe_log(d_alpha_dt), "o")
    ax2.set_xlabel("1/T")
    ax2.set_ylabel("ln(dα/dt)")
    ax2.set_title("Arrhenius Plot")

    plt.tight_layout()
    plt.show()
