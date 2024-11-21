import numpy as np
from typing import Dict, Optional, Tuple
from data_preprocessing.common_preprocessing import smooth_data


def calculate_specific_heat_two_step(
        sample_heat_flow: np.ndarray,
        baseline_heat_flow: np.ndarray,
        temperature: np.ndarray,
        time: np.ndarray,
        sample_mass: float,
        heating_rate: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Calculate specific heat using the two-step method.

    Args:
        sample_heat_flow: Heat flow data from sample measurement
        baseline_heat_flow: Heat flow data from empty pan (baseline)
        temperature: Temperature data in Kelvin
        time: Time data
        sample_mass: Mass of the sample in mg
        heating_rate: Heating rate in K/min. If None, calculated from data.

    Returns:
        Tuple containing:
        - Array of specific heat values
        - Dictionary with calculation metadata
    """
    # Calculate heating rate if not provided
    if heating_rate is None:
        heating_rate = np.mean(np.gradient(temperature, time)) * 60  # Convert to K/min

    # Smooth heat flow signals
    sample_heat_flow_smooth = smooth_data(sample_heat_flow)
    baseline_heat_flow_smooth = smooth_data(baseline_heat_flow)

    # Calculate net heat flow
    net_heat_flow = sample_heat_flow_smooth - baseline_heat_flow_smooth

    # Convert units and calculate Cp
    # Heat flow in mW, mass in mg, heating rate in K/min
    # Result in J/(g·K)
    specific_heat = (net_heat_flow * 60) / (sample_mass * heating_rate)

    metadata = {
        'heating_rate': heating_rate,
        'mean_cp': np.mean(specific_heat),
        'std_cp': np.std(specific_heat)
    }

    return specific_heat, metadata


def calculate_specific_heat_three_step(
        sample_heat_flow: np.ndarray,
        reference_heat_flow: np.ndarray,
        baseline_heat_flow: np.ndarray,
        temperature: np.ndarray,
        time: np.ndarray,
        sample_mass: float,
        reference_mass: float,
        reference_cp: np.ndarray,
        heating_rate: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Calculate specific heat using the three-step method with a reference material.

    Args:
        sample_heat_flow: Heat flow data from sample measurement
        reference_heat_flow: Heat flow data from reference material (e.g., sapphire)
        baseline_heat_flow: Heat flow data from empty pan (baseline)
        temperature: Temperature data in Kelvin
        time: Time data
        sample_mass: Mass of the sample in mg
        reference_mass: Mass of the reference material in mg
        reference_cp: Known specific heat values of reference material
        heating_rate: Heating rate in K/min. If None, calculated from data.

    Returns:
        Tuple containing:
        - Array of specific heat values
        - Dictionary with calculation metadata
    """
    # Calculate heating rate if not provided
    if heating_rate is None:
        heating_rate = np.mean(np.gradient(temperature, time)) * 60  # Convert to K/min

    # Smooth heat flow signals
    sample_heat_flow_smooth = smooth_data(sample_heat_flow)
    reference_heat_flow_smooth = smooth_data(reference_heat_flow)
    baseline_heat_flow_smooth = smooth_data(baseline_heat_flow)

    # Calculate net heat flows
    net_sample = sample_heat_flow_smooth - baseline_heat_flow_smooth
    net_reference = reference_heat_flow_smooth - baseline_heat_flow_smooth

    # Calculate specific heat using reference material method
    # (HFsample / msample) = (HFreference / mreference) * (Cpsample / Cpreference)
    specific_heat = (net_sample * reference_mass * reference_cp) / (net_reference * sample_mass)

    metadata = {
        'heating_rate': heating_rate,
        'mean_cp': np.mean(specific_heat),
        'std_cp': np.std(specific_heat),
        'reference_ratio': np.mean(net_sample / net_reference)
    }

    return specific_heat, metadata


def get_sapphire_cp(temperature: np.ndarray) -> np.ndarray:
    """
    Calculate the specific heat capacity of sapphire (α-Al2O3) reference material.
    Valid for temperature range 273.15 K to 1000 K.

    Args:
        temperature: Temperature data in Kelvin

    Returns:
        Array of specific heat values in J/(g·K)
    """
    # Coefficients for Cp calculation (valid 273.15 K to 1000 K)
    # From NIST Standard Reference Material 720
    a = 1.0289
    b = 2.3506e-4
    c = -1.6818e-7

    # Ensure temperature is within valid range
    if np.any(temperature < 273.15) or np.any(temperature > 1000):
        raise ValueError("Temperature must be between 273.15 K and 1000 K")

    # Calculate Cp using polynomial equation
    cp = a + b * temperature + c * temperature ** 2

    return cp