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
        start_temp: Start temperature for analysis (higher for cooling, lower for heating)
        end_temp: End temperature for analysis (lower for cooling, higher for heating)

    Returns:
        Tuple containing:
            - Temperature array within specified range
            - Strain array within specified range

    Raises:
        ValueError: If temperatures are outside data range or incorrectly ordered for direction
    """
    temp_min, temp_max = min(temperature), max(temperature)
    if not (temp_min <= start_temp <= temp_max and temp_min <= end_temp <= temp_max):
        raise ValueError(
            f"Analysis temperatures ({start_temp:.1f}°C, {end_temp:.1f}°C) must be within data range [{temp_min:.1f}°C, {temp_max:.1f}°C]"
        )

    is_cooling = detect_segment_direction(temperature, strain)

    # Validate order based on direction BEFORE creating mask
    if is_cooling:
        if start_temp <= end_temp:
            raise ValueError(
                "For cooling segments, start temperature must be greater than end temperature."
            )
        mask = (temperature <= start_temp) & (temperature >= end_temp)
    else:
        if start_temp >= end_temp:
            raise ValueError(
                "For heating segments, start temperature must be less than end temperature."
            )
        mask = (temperature >= start_temp) & (temperature <= end_temp)

    if not np.any(mask):
        raise ValueError(
            "No data points found within the specified temperature range and direction."
        )

    return temperature[mask], strain[mask]


def validate_temperature_range(
    temperature: np.ndarray,
    start_temp: Optional[float] = None,
    end_temp: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Validate if temperature range is valid for analysis, considering direction.

    Args:
        temperature: Temperature data array
        start_temp: Start temperature for analysis (optional)
        end_temp: End temperature for analysis (optional)

    Returns:
        Tuple[bool, str]: (True if range is valid, explanation message)
    """
    if start_temp is None or end_temp is None:
        return True, "No range specified."

    temp_min, temp_max = min(temperature), max(temperature)

    # Check temperatures are within data range
    start_in_range = temp_min <= start_temp <= temp_max
    end_in_range = temp_min <= end_temp <= temp_max

    if not start_in_range or not end_in_range:
        msg = f"Temperatures must be within the data range [{temp_min:.1f}°C, {temp_max:.1f}°C]."
        if not start_in_range:
            msg += f" Start temp {start_temp:.1f}°C is outside."
        if not end_in_range:
            msg += f" End temp {end_temp:.1f}°C is outside."
        return False, msg

    # Determine if cooling or heating segment
    # Pass dummy strain if None, as direction depends only on temperature trend vs index
    dummy_strain = np.arange(len(temperature))
    is_cooling = detect_segment_direction(temperature, dummy_strain)

    # Validate direction-appropriate order
    if is_cooling:
        if start_temp > end_temp:
            return True, "Valid range for cooling segment."
        else:
            return (
                False,
                "Invalid range: For cooling segments, start temperature must be greater than end temperature.",
            )
    else:
        if start_temp < end_temp:
            return True, "Valid range for heating segment."
        else:
            return (
                False,
                "Invalid range: For heating segments, start temperature must be less than end temperature.",
            )


def detect_segment_direction(
    temperature: np.ndarray, strain: Optional[np.ndarray] = None
) -> bool:
    """
    Detect if the data segment represents cooling or heating based on temperature trend.

    Args:
        temperature: Temperature data array
        strain: Strain data array (optional, not used for direction detection)

    Returns:
        bool: True if cooling segment (decreasing temperature trend), False if heating
    """
    if len(temperature) < 2:
        return False  # Cannot determine direction with less than 2 points

    # Use linear regression of temperature vs. index to robustly determine overall trend
    time_points = np.arange(len(temperature))

    # Handle potential NaN values if any
    valid_mask = ~np.isnan(temperature)
    if np.sum(valid_mask) < 2:
        return False  # Not enough valid points

    # Fit only valid points
    slope = np.polyfit(time_points[valid_mask], temperature[valid_mask], 1)[0]

    # Negative slope indicates cooling
    return slope < 0


def get_analysis_summary(results: Dict) -> str:
    """
    Generate a formatted summary of analysis results.

    Args:
        results: Dictionary containing analysis results

    Returns:
        str: Formatted summary string
    """
    summary = []
    direction = "Cooling" if results.get("is_cooling", False) else "Heating"
    summary.append(f"Analysis Results ({direction}):")
    summary.append(f"Start temperature (T_start): {results['start_temperature']:.2f}°C")
    summary.append(f"End temperature (T_end):   {results['end_temperature']:.2f}°C")
    summary.append(f"Mid temperature (T_50%):   {results['mid_temperature']:.2f}°C")

    if "fit_quality" in results:
        fit_quality = results["fit_quality"]
        summary.append("\nTangent Method Fit Quality Metrics:")
        summary.append(
            f"  R² (start fit): {fit_quality.get('r2_start', float('nan')):.4f}"
        )
        summary.append(
            f"  R² (end fit):   {fit_quality.get('r2_end', float('nan')):.4f}"
        )
        summary.append(
            f"  Margin used:    {fit_quality.get('margin_used', float('nan')):.2%}"
        )
        summary.append(
            f"  Deviation thr:  {fit_quality.get('deviation_threshold', float('nan')):.2e}"
        )
        if "warnings" in fit_quality and fit_quality["warnings"]:
            summary.append("  Warnings:")
            for warning in fit_quality["warnings"]:
                summary.append(f"    - {warning}")

    return "\n".join(summary)


def estimate_heating_rate(
    temperature: np.ndarray, time: Optional[np.ndarray] = None
) -> float:
    """
    Estimate heating/cooling rate from temperature data.

    Args:
        temperature: Temperature data array
        time: Time data array (optional, assumed in seconds if provided)

    Returns:
        float: Estimated rate in °C/min. Positive for heating, negative for cooling.
               Returns NaN if calculation is not possible.
    """
    if len(temperature) < 2:
        return float("nan")

    if time is None:
        # Assume constant time steps if time not provided, rate cannot be determined in °C/min
        # Fit temperature vs index to get slope per point
        indices = np.arange(len(temperature))
        slope = np.polyfit(indices, temperature, 1)[0]
        # Cannot return meaningful °C/min without time, maybe return slope per index?
        # For now, return NaN as time unit is unknown.
        return float("nan")  # Or consider raising an error/warning
    else:
        if len(time) != len(temperature):
            raise ValueError("Time and temperature arrays must have the same length.")
        # Calculate average rate using robust linear fit (slope)
        # Ensure time is in minutes for the output unit
        time_minutes = time / 60.0
        slope, _ = np.polyfit(time_minutes, temperature, 1)
        return float(slope)  # °C/min


def get_transformation_metrics(results: Dict) -> Dict:
    """
    Calculate additional transformation metrics.

    Args:
        results: Dictionary containing analysis results

    Returns:
        Dict: Additional transformation metrics
    """
    metrics = {}
    start_temp = results["start_temperature"]
    end_temp = results["end_temperature"]
    mid_temp = results["mid_temperature"]
    is_cooling = results.get("is_cooling", False)

    # Temperature range (absolute value)
    metrics["temperature_span"] = abs(end_temp - start_temp)

    # Mid-point position (normalized relative to transformation span)
    temp_span = end_temp - start_temp  # Can be negative for cooling
    if abs(temp_span) > 1e-6:  # Avoid division by zero
        # Normalize relative to the start of the transformation process
        mid_position = (mid_temp - start_temp) / temp_span
        # For cooling, Tstart > Tend, span is negative. If Tmid is between them, ratio is positive.
        # Clamp between 0 and 1, although ideally it should be ~0.5
        metrics["normalized_mid_position"] = np.clip(mid_position, 0, 1)
    else:
        metrics["normalized_mid_position"] = float("nan")

    # Transformation rate estimation (requires temperature and transformed_fraction)
    if "transformed_fraction" in results and "temperature" in results:
        fraction = results["transformed_fraction"]
        temperature = results[
            "temperature"
        ]  # Need temperature associated with fraction
        if len(fraction) > 1 and len(temperature) == len(fraction):
            # Calculate derivative of fraction w.r.t temperature
            df_dT = np.gradient(fraction, temperature)
            # Max rate is the peak of the absolute derivative
            max_rate_idx = np.argmax(np.abs(df_dT))
            metrics["max_transformation_rate_per_degree"] = df_dT[max_rate_idx]
            metrics["temperature_at_max_rate"] = temperature[max_rate_idx]

    return metrics
