"""Kissinger-Akahira-Sunose (KAS) method for model-free kinetic analysis."""

import logging
from typing import List, Tuple

import numpy as np
from scipy.constants import R
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def kas_method(
    temperature_data: List[np.ndarray],
    conversion_data: List[np.ndarray],
    heating_rates: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Kissinger-Akahira-Sunose (KAS) analysis for model-free kinetics.

    Args:
        temperature_data (List[np.ndarray]): List of temperature data arrays for each heating rate.
        conversion_data (List[np.ndarray]): List of conversion data arrays for each heating rate.
        heating_rates (List[float]): List of heating rates corresponding to the data arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Activation energy for each conversion degree (E_a, J/mol)
            - Apparent pre-exponential factor A/g(alpha) for each conversion
              degree, in the time unit of the heating rates (1/min for K/min).
              Model-free methods cannot separate A from the integral reaction
              model g(alpha): multiply by g(alpha) to get A.
            - Conversion levels used for the analysis
            - R-squared values for each linear regression

    Raises:
        ValueError: If input data is inconsistent or invalid.

    Note:
        Linearizes ln(beta/T^2) = ln(A R / (E_a g(alpha))) - E_a / (R T).
        This method assumes that the Arrhenius equation and the integral approximation used in
        the KAS method are valid for the reaction being studied.
    """
    if len(temperature_data) != len(conversion_data) or len(temperature_data) != len(
        heating_rates
    ):
        raise ValueError("Inconsistent number of datasets")

    for temp, conv in zip(temperature_data, conversion_data):
        if len(temp) != len(conv):
            raise ValueError(
                "Temperature and conversion data arrays must have the same length"
            )
        if np.any(temp <= 0):
            raise ValueError("Temperature values must be positive")
        if np.any((conv < 0) | (conv > 1)):
            raise ValueError("Conversion values must be between 0 and 1")

    if np.any(np.array(heating_rates) <= 0):
        raise ValueError("Heating rates must be positive")

    # Determine the range of conversion to analyze
    conversion_range = np.linspace(0.1, 0.9, 100)

    activation_energy = []
    pre_exp_factor = []
    r_squared_values = []

    for alpha in conversion_range:
        y_data = []
        x_data = []

        for temp, conv, beta in zip(temperature_data, conversion_data, heating_rates):
            # Find the temperature at the closest conversion value
            idx = np.argmin(np.abs(conv - alpha))
            T = temp[idx]

            y_data.append(np.log(beta / T**2))
            x_data.append(1 / (R * T))

        # Perform linear regression
        slope, intercept, r_value, _, _ = linregress(x_data, y_data)

        # x is already 1/(R T), so the slope is -E_a
        E_a = -slope
        A = np.exp(intercept) * E_a / R  # A / g(alpha)

        activation_energy.append(E_a)
        pre_exp_factor.append(A)
        r_squared_values.append(r_value**2)

    return (
        np.array(activation_energy),
        np.array(pre_exp_factor),
        conversion_range,
        np.array(r_squared_values),
    )
