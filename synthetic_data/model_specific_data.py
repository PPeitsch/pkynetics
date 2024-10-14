"""Module for generating model-specific kinetic data."""

import numpy as np
from typing import Tuple
from .basic_kinetic_data import generate_basic_kinetic_data
from .noise_generators import add_gaussian_noise
from model_fitting_methods import modified_jmak_equation


def generate_coats_redfern_data(e_a: float, a: float, heating_rate: float,
                                t_range: Tuple[float, float],
                                noise_level: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data specific to Coats-Redfern analysis.
    
    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rate (float): Heating rate in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        noise_level (float): Standard deviation of Gaussian noise to add
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Temperature and conversion data
    """
    temp_data, conv_data = generate_basic_kinetic_data(e_a, a, [heating_rate], t_range,
                                                       reaction_model='nth_order', noise_level=noise_level)
    return temp_data[0], conv_data[0]


def generate_freeman_carroll_data(e_a: float, a: float, heating_rate: float,
                                  t_range: Tuple[float, float],
                                  noise_level: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data specific to Freeman-Carroll analysis.
    
    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rate (float): Heating rate in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        noise_level (float): Standard deviation of Gaussian noise to add
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Temperature, conversion, and time data
    """
    temp_data, conv_data = generate_basic_kinetic_data(e_a, a, [heating_rate], t_range,
                                                       reaction_model='nth_order', noise_level=noise_level)
    time_data = (temp_data[0] - temp_data[0][0]) / heating_rate
    return temp_data[0], conv_data[0], time_data


def generate_jmak_data(time: np.ndarray, n: float, k: float, noise_level: float = 0.01) -> np.ndarray:
    """
    Generate JMAK (Johnson-Mehl-Avrami-Kolmogorov) data with optional noise.

    Args:
        time (np.ndarray): Time array
        n (float): JMAK exponent
        k (float): Rate constant
        noise_level (float): Standard deviation of Gaussian noise to add

    Returns:
        np.ndarray: Transformed fraction data
    """
    transformed_fraction = 1 - np.exp(-(k * time) ** n)
    if noise_level > 0:
        transformed_fraction = add_gaussian_noise(transformed_fraction, noise_level)
    return transformed_fraction


def generate_modified_jmak_data(T: np.ndarray, k0: float, n: float, E: float, T0: float, phi: float,
                                noise_level: float = 0.01) -> np.ndarray:
    """
    Generate synthetic data based on the modified JMAK model.

    Args:
        T (np.ndarray): Temperature array.
        k0 (float): Pre-exponential factor.
        n (float): Avrami exponent.
        E (float): Activation energy.
        T0 (float): Onset temperature.
        phi (float): Heating rate.
        noise_level (float): Standard deviation of Gaussian noise to add.

    Returns:
        np.ndarray: Synthetic transformed fraction data.
    """
    transformed_fraction = modified_jmak_equation(T, k0, n, E, T0, phi)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, transformed_fraction.shape)
        transformed_fraction = np.clip(transformed_fraction + noise, 0, 1)
    return transformed_fraction
