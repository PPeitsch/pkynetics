"""Kissinger method for non-isothermal kinetics analysis."""

import logging
from typing import Tuple

import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy.optimize import fsolve

logger = logging.getLogger(__name__)

R = 8.314  # Gas constant in J/(mol·K)


def kissinger_nonlinear_eq(t: float, e_a: float, a: float, b: float) -> float:
    """
    Kissinger nonlinear equation for peak temperature calculation.

    Args:
        t (float): Temperature in K.
        e_a (float): Activation energy in J/mol.
        a (float): Pre-exponential factor in min^-1.
        b (float): Heating rate in K/min.

    Returns:
        float: Value of the Kissinger equation at the given temperature.
    """
    result = (e_a * b) / (R * t**2) - a * np.exp(-e_a / (R * t))
    return float(result)


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
    t_p = np.zeros_like(beta)
    for i, b in enumerate(beta):
        try:
            t_p[i] = fsolve(
                kissinger_nonlinear_eq, x0=e_a / (R * 20), args=(e_a, a, b)
            )[0]
        except (RuntimeError, ValueError) as e:
            logger.warning(
                f"Failed to converge for heating rate {b}: {e}. Using initial guess."
            )
            t_p[i] = e_a / (R * 20)
    return t_p


def kissinger_equation(
    t_p: NDArray[np.float64], beta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Kissinger equation for non-isothermal kinetics.

    Args:
        t_p (np.ndarray): Peak temperatures in K.
        beta (np.ndarray): Heating rates in K/min.

    Returns:
        np.ndarray: ln(β/T_p^2) values.
    """
    return np.log(beta / t_p**2)


def kissinger_method(
    t_p: np.ndarray, beta: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Perform Kissinger analysis for non-isothermal kinetics.

    Args:
        t_p (np.ndarray): Peak temperatures for different heating rates in °C.
        beta (np.ndarray): Heating rates in K/min.

    Returns:
        Tuple[float, float, float, float, float]:
            Activation energy (e_a) in J/mol,
            Pre-exponential factor (a) in min^-1,
            Standard error of E_a in J/mol,
            Standard error of ln(A),
            Coefficient of determination (r_squared).

    Raises:
        ValueError: If input arrays have different lengths, negative or invalid values.
    """
    # Input validation
    if len(t_p) != len(beta):
        raise ValueError(
            "Temperature and heating rate arrays must have the same length"
        )

    if len(t_p) < 2:
        raise ValueError("At least two data points are required for Kissinger analysis")

    if np.any(t_p < -273.15):  # Check for temperatures below absolute zero
        raise ValueError("Temperature values cannot be below absolute zero")

    if np.any(beta <= 0):
        raise ValueError("Heating rates must be positive")

    # Convert temperatures to Kelvin
    t_p_k = t_p + 273.15

    x = 1 / t_p_k
    y = kissinger_equation(t_p=t_p_k, beta=beta)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    slope = model.params[1]
    intercept = model.params[0]
    se_slope = model.bse[1]
    se_intercept = model.bse[0]

    e_a = -R * slope
    se_e_a = R * se_slope
    a = np.exp(intercept + np.log(e_a / R))
    se_ln_a = np.sqrt(se_intercept**2 + (se_e_a / e_a) ** 2)

    r_squared = model.rsquared

    logger.info(
        f"Kissinger analysis completed. E_a = {e_a:.2f} ± {se_e_a:.2f} J/mol, "
        f"A = {a:.2e} min^-1, R^2 = {r_squared:.4f}"
    )

    return e_a, a, se_e_a, se_ln_a, r_squared
