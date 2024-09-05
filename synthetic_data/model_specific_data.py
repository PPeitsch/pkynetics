"""Module for generating model-specific kinetic data."""

import numpy as np
from typing import Tuple, List
from .basic_kinetic_data import generate_basic_kinetic_data


def generate_coats_redfern_data(e_a: float, a: float, heating_rate: float,
                                t_range: Tuple[float, float], n: float = 1,
                                noise_level: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data specific to Coats-Redfern analysis.
    
    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rate (float): Heating rate in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        n (float): Reaction order
        noise_level (float): Standard deviation of Gaussian noise to add
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Temperature and conversion data
    """
    temp_data, conv_data = generate_basic_kinetic_data(e_a, a, [heating_rate], t_range,
                                                       reaction_model='nth_order', noise_level=noise_level)
    return temp_data[0], conv_data[0]


def generate_freeman_carroll_data(e_a: float, a: float, heating_rate: float,
                                  t_range: Tuple[float, float], n: float = 1.5,
                                  noise_level: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data specific to Freeman-Carroll analysis.
    
    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rate (float): Heating rate in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        n (float): Reaction order
        noise_level (float): Standard deviation of Gaussian noise to add
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Temperature, conversion, and time data
    """
    temp_data, conv_data = generate_basic_kinetic_data(e_a, a, [heating_rate], t_range,
                                                       reaction_model='nth_order', noise_level=noise_level)
    time_data = (temp_data[0] - temp_data[0][0]) / heating_rate
    return temp_data[0], conv_data[0], time_data
