import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

# Constants
R = 8.314  # Gas constant in J/(mol·K)

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
    Generate basic kinetic data for testing various models.

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
        time = (t - t[0]) / beta
        k = a * np.exp(-e_a / (R * t))

        if reaction_model == "first_order":
            alpha = 1 - np.exp(-k * time)
        elif reaction_model == "nth_order":
            alpha = 1 - (1 + (n - 1) * k * time) ** (1 / (1 - n))
        else:
            logger.warning(f"Unsupported reaction model: {reaction_model}")
            raise ValueError(f"Unsupported reaction model: {reaction_model}")

        # Add noise
        if noise_level > 0:
            alpha = np.clip(alpha + np.random.normal(0, noise_level, alpha.shape), 0, 1)

        temperature_data.append(t)
        conversion_data.append(alpha)

    return temperature_data, conversion_data
