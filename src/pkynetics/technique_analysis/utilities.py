from typing import Dict, Optional, Tuple

import numpy as np


def analyze_range(
    temperature: np.ndarray, strain: np.ndarray, start_temp: float, end_temp: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and return data within specified temperature range.

    Args:
        temperature: Full temperature array
        strain: Full strain array
        start_temp: Start temperature for analysis
        end_temp: End temperature for analysis

    Returns:
        Tuple containing:
            - Temperature array within specified range
            - Strain array within specified range

    Raises:
        ValueError: If start_temp >= end_temp or if temperatures are outside data range
    """
    if start_temp >= end_temp:
        raise ValueError("Start temperature must be less than end temperature")

    if start_temp < temperature.min() or end_temp > temperature.max():
        raise ValueError("Analysis temperatures must be within data range")

    mask = (temperature >= start_temp) & (temperature <= end_temp)
    return temperature[mask], strain[mask]


def validate_temperature_range(
    temperature: np.ndarray,
    start_temp: Optional[float] = None,
    end_temp: Optional[float] = None,
) -> bool:
    """
    Validate if temperature range is valid for analysis.

    Args:
        temperature: Temperature data array
        start_temp: Start temperature for analysis (optional)
        end_temp: End temperature for analysis (optional)

    Returns:
        bool: True if range is valid, False otherwise
    """
    if start_temp is None and end_temp is None:
        return True

    if start_temp is not None and end_temp is not None:
        if start_temp >= end_temp:
            return False

        if start_temp < temperature.min() or end_temp > temperature.max():
            return False

    return True


def get_analysis_summary(results: Dict) -> str:
    """
    Generate a formatted summary of analysis results.

    Args:
        results: Dictionary containing analysis results

    Returns:
        str: Formatted summary string
    """
    summary = []
    summary.append("Analysis Results:")
    summary.append(f"Start temperature: {results['start_temperature']:.2f}°C")
    summary.append(f"End temperature: {results['end_temperature']:.2f}°C")
    summary.append(f"Mid temperature: {results['mid_temperature']:.2f}°C")

    if "fit_quality" in results:
        fit_quality = results["fit_quality"]
        summary.append("\nFit Quality Metrics:")
        summary.append(f"R² (start): {fit_quality['r2_start']:.4f}")
        summary.append(f"R² (end): {fit_quality['r2_end']:.4f}")
        summary.append(f"Margin used: {fit_quality['margin_used']:.2%}")

    return "\n".join(summary)


def estimate_heating_rate(
    temperature: np.ndarray, time: Optional[np.ndarray] = None
) -> float:
    """
    Estimate heating rate from temperature data.

    Args:
        temperature: Temperature data array
        time: Time data array (optional)

    Returns:
        float: Estimated heating rate in °C/min
    """
    if time is None:
        # Assume constant time steps if time not provided
        time = np.arange(len(temperature))

    # Calculate average heating rate
    temp_diff = float(temperature[-1] - temperature[0])
    time_diff = float(time[-1] - time[0])
    return temp_diff / time_diff


def get_transformation_metrics(results: Dict) -> Dict:
    """
    Calculate additional transformation metrics.

    Args:
        results: Dictionary containing analysis results

    Returns:
        Dict: Additional transformation metrics
    """
    metrics = {}

    # Temperature range
    metrics["temperature_range"] = (
        results["end_temperature"] - results["start_temperature"]
    )

    # Mid-point position (normalized)
    temp_range = results["end_temperature"] - results["start_temperature"]
    mid_position = (
        results["mid_temperature"] - results["start_temperature"]
    ) / temp_range
    metrics["normalized_mid_position"] = mid_position

    # Transformation rate estimation
    if "transformed_fraction" in results and len(results["transformed_fraction"]) > 1:
        max_rate = np.max(np.gradient(results["transformed_fraction"]))
        metrics["max_transformation_rate"] = max_rate

    return metrics
