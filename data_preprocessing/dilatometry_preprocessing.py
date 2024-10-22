from typing import Tuple, Dict
import numpy as np
from .common_preprocessing import smooth_data


def calculate_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the curvature of a function.

    Args:
        x (np.ndarray): x-coordinates
        y (np.ndarray): y-coordinates

    Returns:
        np.ndarray: Curvature values
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2y = np.gradient(dy)
    curvature = np.abs(d2y) / (1 + dy ** 2) ** 1.5
    return curvature


def find_inflection_points(temperature: np.ndarray, strain: np.ndarray) -> Tuple[float, float]:
    """
    Find the inflection points that mark the start and end of the transformation.

    Args:
        temperature (np.ndarray): Array of temperature values.
        strain (np.ndarray): Array of strain values.

    Returns:
        Tuple[float, float]: Start and end temperatures of the transformation.
    """
    smooth_strain = smooth_data(strain)
    curvature = calculate_curvature(temperature, smooth_strain)

    # Find peaks in curvature
    peak_indices = np.argpartition(curvature, -5)[-5:]  # Get indices of top 5 peaks
    peak_indices = peak_indices[np.argsort(curvature[peak_indices])][::-1]  # Sort by curvature value

    # Filter out peaks that are too close to each other
    filtered_peaks = [peak_indices[0]]
    for peak in peak_indices[1:]:
        if np.min(np.abs(temperature[peak] - temperature[filtered_peaks])) > 50:  # 50°C minimum separation
            filtered_peaks.append(peak)
        if len(filtered_peaks) == 2:
            break

    start_temp, end_temp = temperature[filtered_peaks[0]], temperature[filtered_peaks[1]]
    return min(start_temp, end_temp), max(start_temp, end_temp)


def find_separation_point(x: np.ndarray, y: np.ndarray, fit_func: np.poly1d, threshold: float = 0.001) -> float:
    """
    Find the point where the curve separates from the linear fit.

    Args:
        x (np.ndarray): x-coordinates
        y (np.ndarray): y-coordinates
        fit_func (np.poly1d): Linear fit function
        threshold (float): Threshold for separation

    Returns:
        float: x-coordinate of separation point
    """
    differences = np.abs(y - fit_func(x))
    separation_index = np.argmax(differences > threshold)
    return x[separation_index]


def extrapolate_linear_segments(temperature: np.ndarray, strain: np.ndarray,
                                start_temp: float, end_temp: float) -> Tuple[
    np.ndarray, np.ndarray, np.poly1d, np.poly1d, float, float]:
    """
    Extrapolate linear segments before and after the transformation.

    Args:
        temperature (np.ndarray): Array of temperature values.
        strain (np.ndarray): Array of strain values.
        start_temp (float): Initial estimate of start temperature.
        end_temp (float): Initial estimate of end temperature.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.poly1d, np.poly1d, float, float]:
            - Extrapolated strain values before transformation
            - Extrapolated strain values after transformation
            - Polynomial function for before extrapolation
            - Polynomial function for after extrapolation
            - Adjusted start temperature
            - Adjusted end temperature
    """
    before_mask = temperature < start_temp
    before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
    before_extrapolation = np.poly1d(before_fit)

    after_mask = temperature > end_temp
    after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)
    after_extrapolation = np.poly1d(after_fit)

    # Adjust start and end temperatures
    adjusted_start = find_separation_point(temperature, strain, before_extrapolation)
    adjusted_end = find_separation_point(temperature[::-1], strain[::-1], after_extrapolation)
    adjusted_end = temperature[-1] - adjusted_end  # Convert back to original scale

    return (before_extrapolation(temperature), after_extrapolation(temperature),
            before_extrapolation, after_extrapolation, adjusted_start, adjusted_end)


def calculate_dilatometry_transformed_fraction(temperature: np.ndarray, strain: np.ndarray,
                                               start_temp: float, end_temp: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Calculate the transformed fraction for dilatometry data using the lever rule.

    Args:
        temperature (np.ndarray): Array of temperature values.
        strain (np.ndarray): Array of strain values.
        start_temp (float): Start temperature of the transformation.
        end_temp (float): End temperature of the transformation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
            - Transformed fraction
            - Extrapolated strain values before transformation
            - Extrapolated strain values after transformation
            - Adjusted start temperature
            - Adjusted end temperature
    """
    before_extrap, after_extrap, _, _, adjusted_start, adjusted_end = extrapolate_linear_segments(temperature, strain,
                                                                                                  start_temp, end_temp)
    transformed_fraction = (strain - before_extrap) / (after_extrap - before_extrap)
    return np.clip(transformed_fraction, 0, 1), before_extrap, after_extrap, adjusted_start, adjusted_end


def analyze_dilatometry_curve(temperature: np.ndarray, strain: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analyze the dilatometry curve to extract key parameters.

    Args:
        temperature (np.ndarray): Array of temperature values.
        strain (np.ndarray): Array of strain values.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing analysis results.
    """
    initial_start, initial_end = find_inflection_points(temperature, strain)
    transformed_fraction, before_extrap, after_extrap, start_temp, end_temp = calculate_dilatometry_transformed_fraction(
        temperature, strain, initial_start, initial_end)

    mid_temp_idx = np.argmin(np.abs(transformed_fraction - 0.5))
    mid_temp = temperature[mid_temp_idx]

    return {
        'start_temperature': start_temp,
        'end_temperature': end_temp,
        'mid_temperature': mid_temp,
        'transformed_fraction': transformed_fraction,
        'before_extrapolation': before_extrap,
        'after_extrapolation': after_extrap,
        'inflection_points': [initial_start, initial_end]
    }
