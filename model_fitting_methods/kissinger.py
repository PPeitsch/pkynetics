"""Kissinger method for non-isothermal kinetics analysis."""

import numpy as np
from scipy import stats
from scipy.optimize import fsolve
from typing import Tuple
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

R = 8.314  # Gas constant in J/(mol·K)


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

    def kissinger_equation(t, b):
        return (e_a * b) / (R * t ** 2) - a * np.exp(-e_a / (R * t))

    t_p = np.zeros_like(beta)
    for i, b in enumerate(beta):
        t_p[i] = fsolve(kissinger_equation, x0=e_a / (R * 20), args=(b,))[0]

    return t_p


def kissinger_equation(t_p: np.ndarray, e_a: float, a: float, beta: np.ndarray) -> np.ndarray:
    """
    Kissinger equation for non-isothermal kinetics.

    Args:
        t_p (np.ndarray): Peak temperatures in K.
        e_a (float): Activation energy in J/mol.
        a (float): Pre-exponential factor in min^-1.
        beta (np.ndarray): Heating rates in K/min.

    Returns:
        np.ndarray: ln(β/T_p^2) values.
    """
    return np.log(beta / t_p ** 2)


def kissinger_method(t_p: np.ndarray, beta: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Perform Kissinger analysis for non-isothermal kinetics.

    Args:
        t_p (np.ndarray): Peak temperatures for different heating rates in K.
        beta (np.ndarray): Heating rates corresponding to the peak temperatures in K/min.

    Returns:
        Tuple[float, float, float, float, float]:
            Activation energy (e_a) in J/mol,
            Pre-exponential factor (a) in min^-1,
            Standard error of E_a in J/mol,
            Standard error of ln(A),
            Coefficient of determination (r_squared).
    """
    logger.info("Performing Kissinger analysis")

    if np.any(t_p <= 0) or np.any(beta <= 0):
        raise ValueError("Peak temperatures and heating rates must be positive")

    if len(t_p) < 2 or len(beta) < 2:
        warnings.warn("Kissinger analysis requires at least two data points. Results may not be reliable.", UserWarning)
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x = 1 / t_p
    y = np.log(beta / t_p ** 2)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)

    if np.isnan(slope) or np.isnan(intercept):
        warnings.warn("Linear regression failed. Check your input data.", UserWarning)
        return np.nan, np.nan, np.nan, np.nan, np.nan

    e_a = -R * slope
    a = np.exp(intercept + np.log(e_a / R))

    se_e_a = R * stderr
    se_ln_a = np.sqrt((stderr / slope) ** 2 + (se_e_a / e_a) ** 2)

    r_squared = r_value ** 2

    logger.info(f"Kissinger analysis completed. E_a = {e_a / 1000:.2f} ± {se_e_a / 1000:.2f} kJ/mol, "
                f"A = {a:.2e} min^-1, R^2 = {r_squared:.4f}")

    return e_a, a, se_e_a, se_ln_a, r_squared
