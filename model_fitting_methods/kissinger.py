"""Kissinger method for non-isothermal kinetics analysis."""

import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_t_p(e_a: float, a: float, beta: np.ndarray) -> np.ndarray:
    """
    Calculate peak temperatures using the Kissinger equation.

    Args:
        e_a (float): Activation energy in J/mol.
        a (float): Pre-exponential factor in min^-1.
        beta (np.ndarray): Heating rates in K/min.

    Returns:
        np.ndarray: Calculated peak temperatures in K.
    """
    r = 8.314  # Gas constant in J/(mol·K)
    t_p = np.full_like(beta, 500.0)  # Initial guess
    for _ in range(100):  # Max 100 iterations
        t_p_new = e_a / (r * np.log(a * r * t_p ** 2 / (e_a * beta)))
        if np.allclose(t_p, t_p_new, rtol=1e-6):
            return t_p_new
        t_p = t_p_new
    return t_p


def kissinger_equation(t_p: np.ndarray, e_a: float, ln_ar_ea: float) -> np.ndarray:
    """
    Kissinger equation for non-isothermal kinetics.

    Args:
        t_p (np.ndarray): Peak temperatures.
        e_a (float): Activation energy.
        ln_ar_ea (float): ln(AR/E_a), where A is the pre-exponential factor and R is the gas constant.

    Returns:
        np.ndarray: ln(β/T_p^2) values.
    """
    r = 8.314  # Gas constant in J/(mol·K)
    return ln_ar_ea - e_a / (r * t_p)


def kissinger_method(t_p: np.ndarray, beta: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform Kissinger analysis for non-isothermal kinetics.

    Args:
        t_p (np.ndarray): Peak temperatures for different heating rates.
        beta (np.ndarray): Heating rates corresponding to the peak temperatures.

    Returns:
        Tuple[float, float, float]: Activation energy (e_a), pre-exponential factor (a), 
                                    and coefficient of determination (r_squared).

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values.
    """
    logger.info("Performing Kissinger analysis")

    t_p = np.asarray(t_p)
    beta = np.asarray(beta)

    if t_p.shape != beta.shape:
        raise ValueError("Peak temperature and heating rate arrays must have the same shape")

    if np.any(t_p <= 0) or np.any(beta <= 0):
        raise ValueError("Peak temperatures and heating rates must be positive")

    try:
        # Prepare data for Kissinger plot
        y = np.log(beta / t_p ** 2)
        x = 1 / t_p

        # Perform linear regression
        slope, intercept = np.polyfit(x, y, 1)

        # Calculate activation energy and pre-exponential factor
        r = 8.314  # Gas constant in J/(mol·K)
        e_a = -slope * r
        a = np.exp(intercept) * e_a / r

        # Calculate R^2
        y_fit = slope * x + intercept
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        logger.info(
            f"Kissinger analysis completed. e_a = {e_a / 1000:.2f} kJ/mol, a = {a:.2e} min^-1, R^2 = {r_squared:.4f}")
        return e_a, a, r_squared

    except Exception as e:
        logger.error(f"Error in Kissinger analysis: {str(e)}")
        raise
