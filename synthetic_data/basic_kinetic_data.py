"""Module for generating basic kinetic data for testing various models."""

import numpy as np
from typing import List, Tuple


def generate_basic_kinetic_data(e_a: float, a: float, heating_rates: List[float],
                                t_range: Tuple[float, float], reaction_model: str = 'first_order',
                                noise_level: float = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate basic kinetic data for testing various models.
    
    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rates (List[float]): List of heating rates in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        reaction_model (str): Type of reaction model ('first_order', 'nth_order', etc.)
        noise_level (float): Standard deviation of Gaussian noise to add
    
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Lists of temperature data and conversion data for each heating rate
    """
    r = 8.314  # Gas constant in J/(mol·K)

    temperature_data = []
    conversion_data = []

    for beta in heating_rates:
        t = np.linspace(*t_range, 1000)
        time = (t - t[0]) / beta
        k = a * np.exp(-e_a / (r * t))

        if reaction_model == 'first_order':
            alpha = 1 - np.exp(-k * time)
        elif reaction_model == 'nth_order':
            n = 1.5  # Example value, could be parameterized
            alpha = 1 - (1 + (n - 1) * k * time) ** (1 / (1 - n))
        else:
            raise ValueError(f"Unsupported reaction model: {reaction_model}")

        # Add noise
        if noise_level > 0:
            alpha = np.clip(alpha + np.random.normal(0, noise_level, alpha.shape), 0, 1)

        temperature_data.append(t)
        conversion_data.append(alpha)

    return temperature_data, conversion_data
