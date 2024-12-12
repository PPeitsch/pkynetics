"""Friedman method for model-free kinetic analysis."""

import logging
from typing import List, Tuple

import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def friedman_method(
    temperature: List[np.ndarray],
    conversion: List[np.ndarray],
    heating_rate: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Friedman analysis for model-free kinetics.

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
        activation_energy, pre_exp_factor, conv_levels, r_squared = friedman_method(temperature, conversion, heating_rate)
    """
    logger.info("Performing Friedman analysis")

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

    for i, alpha in enumerate(conv_levels):
        y_data = []
        x_data = []

        for temp, conv, beta in zip(temperature, conversion, heating_rate):
            # Find the index closest to the current conversion level
            idx = np.argmin(np.abs(conv - alpha))

            # Calculate the reaction rate
            if 0 < idx < len(conv) - 1:
                da_dt = (
                    (conv[idx + 1] - conv[idx - 1])
                    / (temp[idx + 1] - temp[idx - 1])
                    * beta
                )
                if da_dt > 0:  # Only include positive reaction rates
                    y_data.append(np.log(da_dt))
                    x_data.append(1 / (8.314 * temp[idx]))  # 1/RT

        # Perform linear regression
        if len(x_data) > 2:  # Ensure enough data points for regression
            slope, intercept, r_value, _, _ = linregress(x_data, y_data)
            activation_energy[i] = -slope
            pre_exp_factor[i] = np.exp(intercept)
            r_squared[i] = r_value**2
        else:
            activation_energy[i] = pre_exp_factor[i] = r_squared[i] = np.nan

    logger.info("Friedman analysis completed")
    return activation_energy, pre_exp_factor, conv_levels, r_squared
