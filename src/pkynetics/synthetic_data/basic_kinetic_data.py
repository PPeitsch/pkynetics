import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.constants import R
from scipy.special import expn

logger = logging.getLogger(__name__)


def generate_basic_kinetic_data(
    e_a: float,
    a: float,
    heating_rates: NDArray[np.float64],
    t_range: Tuple[float, float],
    reaction_model: str = "first_order",
    noise_level: float = 0,
    num_points: int = 1000,
    n: float = 1.5,
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
    """
    Generate non-isothermal kinetic data at constant heating rates.

    Integrates the rate equation over temperature: at heating rate beta the
    integral of the reaction model is

        g(alpha) = (A / beta) * integral from T0 to T of exp(-Ea / (R T')) dT'

    evaluated exactly with the exponential integral,
    integral of exp(-Ea/(R T)) dT = T * E2(Ea / (R T)). The heating rate is
    converted to K/s so that it matches A in 1/s.

    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rates (List[float]): List of heating rates in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        reaction_model (str): Type of reaction model ('first_order', 'nth_order', etc.)
        noise_level (float): Standard deviation of Gaussian noise to add
        num_points (int): Number of points to generate for each heating rate
        n (float): Reaction order for 'nth_order' model

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Lists of temperature data and conversion data for each heating rate
    """
    temperature_data = []
    conversion_data = []

    for beta in heating_rates:
        t = np.linspace(*t_range, num_points)
        temperature_integral = _temperature_integral(e_a, t) - _temperature_integral(
            e_a, t[:1]
        )
        # Integral of the reaction model, g(alpha); beta from K/min to K/s
        g_alpha = a / (beta / 60.0) * temperature_integral

        # Special case: if n is exactly 1 and reaction_model is nth_order, use first_order formula
        if reaction_model == "nth_order" and abs(n - 1.0) < 1e-10:
            logger.info("Using first_order model when n=1 to avoid division by zero")
            alpha = 1 - np.exp(-g_alpha)
        elif reaction_model == "first_order":
            alpha = 1 - np.exp(-g_alpha)
        elif reaction_model == "nth_order":
            alpha = 1 - (1 + (n - 1) * g_alpha) ** (1 / (1 - n))
        else:
            logger.warning(f"Unsupported reaction model: {reaction_model}")
            raise ValueError(f"Unsupported reaction model: {reaction_model}")

        # Add noise
        if noise_level > 0:
            alpha = np.clip(alpha + np.random.normal(0, noise_level, alpha.shape), 0, 1)

        temperature_data.append(t)
        conversion_data.append(alpha)

    return temperature_data, conversion_data


def _temperature_integral(e_a: float, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Antiderivative of exp(-Ea / (R T)) with respect to T: T * E2(Ea / (R T))."""
    return np.asarray(t * expn(2, e_a / (R * t)), dtype=np.float64)
