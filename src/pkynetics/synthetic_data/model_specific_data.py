"""Module for generating model-specific kinetic data."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from pkynetics.model_fitting_methods import modified_jmak_equation

from .basic_kinetic_data import generate_basic_kinetic_data
from .noise_generators import add_gaussian_noise


def generate_coats_redfern_data(
    e_a: float,
    a: float,
    heating_rate: float,
    t_range: Tuple[float, float],
    n: float = 1.5,
    noise_level: float = 0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate data specific to Coats-Redfern analysis.

    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rate (float): Heating rate in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        n (float): Reaction order. Default is 1.5
        noise_level (float): Standard deviation of Gaussian noise to add

    Returns:
        Tuple[np.ndarray, np.ndarray]: Temperature and conversion data
    """
    temp_data, conv_data = generate_basic_kinetic_data(
        e_a,
        a,
        np.array([heating_rate], dtype=np.float64),
        t_range,
        reaction_model="nth_order",
        noise_level=noise_level,
        n=n,
    )
    return temp_data[0], conv_data[0]


def generate_freeman_carroll_data(
    e_a: float,
    a: float,
    heating_rate: float,
    t_range: Tuple[float, float],
    n: float = 1.5,
    noise_level: float = 0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate data specific to Freeman-Carroll analysis.

    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rate (float): Heating rate in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        n (float): Reaction order. Default is 1.5
        noise_level (float): Standard deviation of Gaussian noise to add

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: Temperature, conversion, and time data
    """
    temp_data, conv_data = generate_basic_kinetic_data(
        e_a,
        a,
        np.array([heating_rate], dtype=np.float64),
        t_range,
        reaction_model="nth_order",
        noise_level=noise_level,
        n=n,
    )
    time_data = np.array(
        (temp_data[0] - temp_data[0][0]) / heating_rate, dtype=np.float64
    )
    return temp_data[0], conv_data[0], time_data


def generate_jmak_data(
    time: NDArray[np.float64], n: float, k: float, noise_level: float = 0.01
) -> NDArray[np.float64]:
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
    transformed_fraction = 1 - np.exp(-((k * time) ** n))
    if noise_level > 0:
        transformed_fraction = add_gaussian_noise(transformed_fraction, noise_level)
    return np.array(transformed_fraction, dtype=np.float64)


def generate_modified_jmak_data(
    T: np.ndarray,
    k0: float,
    n: float,
    E: float,
    T0: float,
    phi: float,
    noise_level: float = 0.01,
) -> NDArray[np.float64]:
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
    return np.array(transformed_fraction, dtype=np.float64)
