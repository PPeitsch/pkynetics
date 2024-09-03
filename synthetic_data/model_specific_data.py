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


def generate_kissinger_data(e_a: float, a: float, heating_rates: List[float],
                            t_range: Tuple[float, float], noise_level: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data specific to Kissinger analysis.
    
    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rates (List[float]): List of heating rates in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        noise_level (float): Standard deviation of Gaussian noise to add
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Peak temperatures and heating rates
    """
    temp_data, conv_data = generate_basic_kinetic_data(e_a, a, heating_rates, t_range,
                                                       reaction_model='first_order', noise_level=noise_level)
    peak_temps = np.array([temp[np.argmax(np.gradient(conv))] for temp, conv in zip(temp_data, conv_data)])
    return peak_temps, np.array(heating_rates)


def generate_ofw_data(e_a: float, a: float, heating_rates: List[float],
                      t_range: Tuple[float, float], noise_level: float = 0) -> Tuple[
    List[np.ndarray], List[np.ndarray]]:
    """
    Generate data specific to Ozawa-Flynn-Wall (OFW) analysis.
    
    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rates (List[float]): List of heating rates in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        noise_level (float): Standard deviation of Gaussian noise to add
    
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Temperature data and conversion data for each heating rate
    """
    return generate_basic_kinetic_data(e_a, a, heating_rates, t_range,
                                       reaction_model='first_order', noise_level=noise_level)


def generate_friedman_data(e_a: float, a: float, heating_rates: List[float],
                           t_range: Tuple[float, float], noise_level: float = 0) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate data specific to Friedman analysis.
    
    Args:
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in 1/s
        heating_rates (List[float]): List of heating rates in K/min
        t_range (Tuple[float, float]): Temperature range (start, end) in K
        noise_level (float): Standard deviation of Gaussian noise to add
    
    Returns: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: Temperature data, conversion data,
    and conversion rate data for each heating rate
    """
    temp_data, conv_data = generate_basic_kinetic_data(e_a, a, heating_rates, t_range,
                                                       reaction_model='first_order', noise_level=noise_level)
    conv_rate_data = [np.gradient(conv, temp) * beta for conv, temp, beta in zip(conv_data, temp_data, heating_rates)]
    return temp_data, conv_data, conv_rate_data
