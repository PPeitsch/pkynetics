import numpy as np
from typing import Dict, Tuple
from data_preprocessing.common_preprocessing import smooth_data


def find_inflection_points(temperature: np.ndarray, strain: np.ndarray) -> Tuple[float, float]:
    """
    Find the inflection points that mark the start and end of the transformation.

    Args:
        temperature (np.ndarray): Array of temperature values.
        strain (np.ndarray): Array of strain values.

    Returns:
        Tuple[float, float]: Start and end temperatures of the transformation.

    This function uses the second derivative of the smoothed strain data to identify
    the two most prominent inflection points, which are assumed to be the start and
    end of the transformation.
    """
    smooth_strain = smooth_data(strain)
    second_derivative = np.gradient(np.gradient(smooth_strain))
    peaks = np.argsort(np.abs(second_derivative))[-2:]
    start_temp, end_temp = temperature[min(peaks)], temperature[max(peaks)]
    return start_temp, end_temp


def extrapolate_linear_segments(temperature: np.ndarray, strain: np.ndarray,
                                start_temp: float, end_temp: float) -> Tuple[
        np.ndarray, np.ndarray, np.poly1d, np.poly1d]:
    """
    Extrapolate linear segments before and after the transformation.

    Args:
        temperature (np.ndarray): Array of temperature values.
        strain (np.ndarray): Array of strain values.
        start_temp (float): Start temperature of the transformation.
        end_temp (float): End temperature of the transformation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.poly1d, np.poly1d]:
            - Extrapolated strain values before transformation
            - Extrapolated strain values after transformation
            - Polynomial function for before extrapolation
            - Polynomial function for after extrapolation
    """
    before_mask = temperature < start_temp
    before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
    before_extrapolation = np.poly1d(before_fit)

    after_mask = temperature > end_temp
    after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)
    after_extrapolation = np.poly1d(after_fit)

    return (before_extrapolation(temperature), after_extrapolation(temperature),
            before_extrapolation, after_extrapolation)


def calculate_transformed_fraction_lever(temperature: np.ndarray, strain: np.ndarray,
                                         start_temp: float, end_temp: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the transformed fraction using the lever rule method.

    Args:
        temperature (np.ndarray): Array of temperature values.
        strain (np.ndarray): Array of strain values.
        start_temp (float): Start temperature of the transformation.
        end_temp (float): End temperature of the transformation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Transformed fraction
            - Extrapolated strain values before transformation
            - Extrapolated strain values after transformation
    """
    before_extrap, after_extrap, _, _ = extrapolate_linear_segments(
        temperature, strain, start_temp, end_temp)
    transformed_fraction = (strain - before_extrap) / (after_extrap - before_extrap)
    return np.clip(transformed_fraction, 0, 1), before_extrap, after_extrap


def analyze_dilatometry_curve(temperature: np.ndarray, strain: np.ndarray,
                              method: str = 'lever') -> Dict[str, np.ndarray]:
    """
    Analyze the dilatometry curve to extract key parameters.

    Args:
        temperature (np.ndarray): Array of temperature values.
        strain (np.ndarray): Array of strain values.
        method (str): Analysis method to use ('lever' or 'tangent')

    Returns:
        Dict[str, np.ndarray]: Analysis results including temperatures and transformed fraction

    Raises:
        ValueError: If an unsupported method is specified
    """
    if method == 'lever':
        start_temp, end_temp = find_inflection_points(temperature, strain)
        transformed_fraction, before_extrap, after_extrap = calculate_transformed_fraction_lever(
            temperature, strain, start_temp, end_temp)

        mid_temp_idx = np.argmin(np.abs(transformed_fraction - 0.5))
        mid_temp = temperature[mid_temp_idx]

        return {
            'start_temperature': start_temp,
            'end_temperature': end_temp,
            'mid_temperature': mid_temp,
            'transformed_fraction': transformed_fraction,
            'before_extrapolation': before_extrap,
            'after_extrapolation': after_extrap
        }
    elif method == 'tangent':
        return tangent_method(temperature, strain)
    else:
        raise ValueError(f"Unsupported method: {method}")


def tangent_method(temperature: np.ndarray, strain: np.ndarray,
                   window_size: int = 20, deviation_threshold: float = 0.001) -> Dict:
    """
    Analyze dilatometry curve using the tangent method to find transformation points.

    Args:
        temperature: Temperature data in °C
        strain: Relative length change data
        window_size: Size of window for calculating local tangents
        deviation_threshold: Threshold for identifying significant deviations

    Returns:
        Dict containing analysis results including transformation temperatures and fractions
    """
    temperature = np.asarray(temperature)
    strain = np.asarray(strain)

    # Calculate initial tangent line (before transformation)
    start_region = slice(0, window_size * 2)
    p_start = np.polyfit(temperature[start_region], strain[start_region], 1)

    # Calculate final tangent line (after transformation)
    end_region = slice(-window_size * 2, None)
    p_end = np.polyfit(temperature[end_region], strain[end_region], 1)

    # Calculate predicted values
    pred_start = np.polyval(p_start, temperature)
    pred_end = np.polyval(p_end, temperature)

    # Find deviations
    dev_start = np.abs(strain - pred_start)
    dev_end = np.abs(strain - pred_end)

    # Identify transformation points
    start_devs = dev_start > deviation_threshold
    end_devs = dev_end > deviation_threshold

    start_temp_idx = _find_start_point(start_devs, window_size)
    end_temp_idx = _find_end_point(end_devs, window_size)

    transformed_fraction = _calculate_transformed_fraction_tangent(
        strain, pred_start, pred_end, start_temp_idx, end_temp_idx)

    mid_temp_idx = np.argmin(np.abs(transformed_fraction - 0.5))

    return {
        'start_temperature': temperature[start_temp_idx],
        'end_temperature': temperature[end_temp_idx],
        'mid_temperature': temperature[mid_temp_idx],
        'transformed_fraction': transformed_fraction,
        'before_extrapolation': pred_start,
        'after_extrapolation': pred_end
    }


# Helper functions for tangent method
def _find_start_point(deviations: np.ndarray, window_size: int) -> int:
    """Find the transformation start point."""
    for i in range(window_size, len(deviations) - window_size):
        if np.all(deviations[i:i + window_size]):
            return i
    raise ValueError("Could not find transformation start point")


def _find_end_point(deviations: np.ndarray, window_size: int) -> int:
    """Find the transformation end point."""
    for i in range(len(deviations) - window_size - 1, window_size, -1):
        if np.all(deviations[i - window_size:i]):
            return i
    raise ValueError("Could not find transformation end point")


def _calculate_transformed_fraction_tangent(strain: np.ndarray, pred_start: np.ndarray,
                                            pred_end: np.ndarray, start_idx: int,
                                            end_idx: int) -> np.ndarray:
    """Calculate transformed fraction for tangent method."""
    transformed_fraction = np.zeros_like(strain)
    transformation_region = slice(start_idx, end_idx + 1)

    height_total = pred_end[transformation_region] - pred_start[transformation_region]
    height_current = strain[transformation_region] - pred_start[transformation_region]
    transformed_fraction[transformation_region] = height_current / height_total
    transformed_fraction[end_idx + 1:] = 1.0

    return transformed_fraction
