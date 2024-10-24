import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Dict
from .common_preprocessing import smooth_data


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

    This function fits linear functions to the segments before and after the
    transformation, and uses these to extrapolate over the entire temperature range.
    """
    before_mask = temperature < start_temp
    before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
    before_extrapolation = np.poly1d(before_fit)

    after_mask = temperature > end_temp
    after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)
    after_extrapolation = np.poly1d(after_fit)

    return before_extrapolation(temperature), after_extrapolation(
        temperature), before_extrapolation, after_extrapolation


def calculate_dilatometry_transformed_fraction(temperature: np.ndarray, strain: np.ndarray,
                                               start_temp: float, end_temp: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the transformed fraction for dilatometry data using the lever rule.

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

    This function applies the lever rule to calculate the transformed fraction
    based on the extrapolated linear segments before and after the transformation.
    """
    before_extrap, after_extrap, _, _ = extrapolate_linear_segments(temperature, strain, start_temp, end_temp)
    transformed_fraction = (strain - before_extrap) / (after_extrap - before_extrap)
    return np.clip(transformed_fraction, 0, 1), before_extrap, after_extrap


def analyze_dilatometry_curve(temperature: np.ndarray, strain: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analyze the dilatometry curve to extract key parameters.

    Args:
        temperature (np.ndarray): Array of temperature values.
        strain (np.ndarray): Array of strain values.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'start_temperature': Start temperature of the transformation
            - 'end_temperature': End temperature of the transformation
            - 'mid_temperature': Mid-point temperature of the transformation
            - 'transformed_fraction': Array of transformed fraction values
            - 'before_extrapolation': Extrapolated strain values before transformation
            - 'after_extrapolation': Extrapolated strain values after transformation

    This function performs a complete analysis of the dilatometry curve, including
    finding the transformation points, calculating the transformed fraction, and
    extrapolating the linear segments before and after the transformation.
    """
    start_temp, end_temp = find_inflection_points(temperature, strain)
    transformed_fraction, before_extrap, after_extrap = calculate_dilatometry_transformed_fraction(temperature, strain,
                                                                                                   start_temp, end_temp)

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