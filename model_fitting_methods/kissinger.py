"""Kissinger method for non-isothermal kinetics analysis."""

import numpy as np
from scipy import stats
from scipy.optimize import fsolve
from typing import Tuple
import logging

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

    def kissinger_equation(t, b):
        return (e_a * b) / (r * t ** 2) - a * np.exp(-e_a / (r * t))

    t_p = np.zeros_like(beta)
    for i, b in enumerate(beta):
        t_p[i] = fsolve(kissinger_equation, x0=e_a / (r * 20), args=(b,))[
            0]  # Initial guess based on typical T_p values

    return t_p


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

    x = 1 / t_p
    y = np.log(beta / t_p**2)

    slope, intercept, r_value, _, stderr = stats.linregress(x, y)

    r = 8.314  # Gas constant in J/(mol·K)
    e_a = -r * slope
    ln_a = intercept + np.log(e_a / r)
    a = np.exp(ln_a)

    se_e_a = r * stderr
    se_ln_a = np.sqrt((stderr / slope)**2 + (se_e_a / e_a)**2)

    r_squared = r_value**2

    logger.info(f"Kissinger analysis completed. E_a = {e_a/1000:.2f} ± {se_e_a/1000:.2f} kJ/mol, "
                f"A = {a:.2e} min^-1, R^2 = {r_squared:.4f}")

    return e_a, a, se_e_a, se_ln_a, r_squared
