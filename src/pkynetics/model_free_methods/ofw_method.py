"""Ozawa-Flynn-Wall method for model-free kinetic analysis."""

import logging
from typing import List, Tuple

import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def ofw_method(
    temperature: List[np.ndarray],
    conversion: List[np.ndarray],
    heating_rate: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Ozawa-Flynn-Wall analysis for model-free kinetics.

    Args:
        temperature (List[np.ndarray]): List of temperature data arrays for each heating rate.
        conversion (List[np.ndarray]): List of conversion data arrays for each heating rate.
        heating_rate (List[float]): List of heating rates.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - activation_energy for each conversion level
            - pre_exp_factor for each conversion level
            - conv_levels
            - r_squared values for each linear regression

    Raises:
        ValueError: If input data is inconsistent or invalid.

    Example:
        temperature = [temp_data1, temp_data2, ...]
        conversion = [conv_data1, conv_data2, ...]
        heating_rate = [5, 10, 15, ...]
        activation_energy, pre_exp_factor, conv_levels, r_squared = ofw_method(temperature, conversion, heating_rate)
    """
    logger.info("Performing Ozawa-Flynn-Wall analysis")

    if not (len(temperature) == len(conversion) == len(heating_rate)):
        raise ValueError("Inconsistent number of datasets")

    # Validate input data
    for temp, conv in zip(temperature, conversion):
        if len(temp) != len(conv):
            raise ValueError(
                "Temperature and conversion arrays must have the same length"
            )
        if np.any(temp <= 0):
            raise ValueError("Temperature values must be positive")
        if np.any((conv < 0) | (conv > 1)):
            raise ValueError("Conversion values must be between 0 and 1")

    # Find common conversion levels
    conv_levels = np.linspace(0.1, 0.9, 50)

    activation_energy = np.zeros_like(conv_levels)
    pre_exp_factor = np.zeros_like(conv_levels)
    r_squared = np.zeros_like(conv_levels)

    log_beta = np.log(heating_rate)

    for i, alpha in enumerate(conv_levels):
        temperatures = []

        for temp, conv in zip(temperature, conversion):
            # Find the index closest to the current conversion level
            idx = np.argmin(np.abs(conv - alpha))
            temperatures.append(temp[idx])

        if len(temperatures) > 2:  # Ensure enough data points for regression
            y = log_beta
            x = 1 / (8.314 * np.array(temperatures))  # 1/RT

            slope, intercept, r_value, _, _ = linregress(x, y)

            activation_energy[i] = -slope * 1.052  # Correction factor for OFW method
            pre_exp_factor[i] = np.exp(intercept)
            r_squared[i] = r_value**2
        else:
            activation_energy[i] = pre_exp_factor[i] = r_squared[i] = np.nan

    logger.info("Ozawa-Flynn-Wall analysis completed")
    return activation_energy, pre_exp_factor, conv_levels, r_squared
