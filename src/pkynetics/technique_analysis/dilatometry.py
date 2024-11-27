import numpy as np
from typing import Dict, Tuple, Optional
from src.pkynetics.data_preprocessing.common_preprocessing import smooth_data


def extrapolate_linear_segments(temperature: np.ndarray, strain: np.ndarray,
                              start_temp: float, end_temp: float) -> Tuple[
                                  np.ndarray, np.ndarray, np.poly1d, np.poly1d]:
    """
    Extrapolate linear segments before and after the transformation.

    Args:
        temperature: Array of temperature values
        strain: Array of strain values
        start_temp: Start temperature of the transformation
        end_temp: End temperature of the transformation

    Returns:
        Tuple containing:
        - Extrapolated strain values before transformation
        - Extrapolated strain values after transformation
        - Polynomial function for before extrapolation
        - Polynomial function for after extrapolation

    Raises:
        ValueError: If temperatures are invalid or if insufficient data for fitting
    """
    # Input validation
    if start_temp >= end_temp:
        raise ValueError("Start temperature must be less than end temperature")
    if not (temperature.min() <= start_temp <= temperature.max()):
        raise ValueError("Start temperature outside data range")
    if not (temperature.min() <= end_temp <= temperature.max()):
        raise ValueError("End temperature outside data range")

    # Create masks for linear regions
    before_mask = temperature < start_temp
    after_mask = temperature > end_temp

    # Ensure enough points for fitting
    min_points = 5
    if np.sum(before_mask) < min_points or np.sum(after_mask) < min_points:
        raise ValueError(f"Insufficient points for fitting. Need at least {min_points} points in each region.")

    try:
        # Fit linear functions to the segments
        before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
        after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)

        # Create polynomial functions
        before_extrapolation = np.poly1d(before_fit)
        after_extrapolation = np.poly1d(after_fit)

    except np.linalg.LinAlgError:
        raise ValueError("Unable to perform linear fit on the data segments")

    # Calculate extrapolated values for the entire temperature range
    before_values = before_extrapolation(temperature)
    after_values = after_extrapolation(temperature)

    return before_values, after_values, before_extrapolation, after_extrapolation


def find_optimal_margin(temperature: np.ndarray, strain: np.ndarray,
                        min_r2: float = 0.99, min_points: int = 10) -> float:
    """
    Determine the optimal margin percentage for linear segment fitting.

    Args:
        temperature: Temperature data array
        strain: Strain data array
        min_r2: Minimum R² value for acceptable linear fit (default: 0.99)
        min_points: Minimum number of points required for fitting (default: 10)

    Returns:
        float: Optimal margin percentage (between 0.1 and 0.4)

    Raises:
        ValueError: If no acceptable margin is found or if data is insufficient
    """
    # Input validation
    if len(temperature) < min_points * 2:
        raise ValueError(f"Insufficient data points. Need at least {min_points * 2} points.")

    # Test different margins
    margins = np.linspace(0.1, 0.4, 7)  # Test margins from 10% to 40%
    best_margin = None
    best_r2 = 0

    for margin in margins:
        n_points = int(len(temperature) * margin)
        if n_points < min_points:
            continue

        # Test start region
        start_mask = temperature <= (temperature.min() + (temperature.max() - temperature.min()) * margin)
        if np.sum(start_mask) < min_points:
            continue

        # Test end region
        end_mask = temperature >= (temperature.max() - (temperature.max() - temperature.min()) * margin)
        if np.sum(end_mask) < min_points:
            continue

        # Calculate R² for both regions
        try:
            # Start region fit
            p_start = np.polyfit(temperature[start_mask], strain[start_mask], 1)
            r2_start = calculate_r2(temperature[start_mask], strain[start_mask], p_start)

            # End region fit
            p_end = np.polyfit(temperature[end_mask], strain[end_mask], 1)
            r2_end = calculate_r2(temperature[end_mask], strain[end_mask], p_end)

            # Calculate average R²
            avg_r2 = (r2_start + r2_end) / 2

            # Update best margin if this one is better
            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_margin = margin

        except (np.linalg.LinAlgError, ValueError):
            continue

    # Check if we found an acceptable margin
    if best_margin is None or best_r2 < min_r2:
        # If no margin meets the min_r2 criteria, return the one with best R²
        if best_margin is not None:
            return best_margin
        # If no margin worked at all, use a default value
        return 0.2

    return best_margin


def calculate_transformed_fraction_lever(temperature: np.ndarray, strain: np.ndarray,
                                         start_temp: float, end_temp: float,
                                         margin_percent: float = 0.2) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the transformed fraction using the lever rule method.

    Args:
        temperature: Array of temperature values
        strain: Array of strain values
        start_temp: Start temperature of the transformation
        end_temp: End temperature of the transformation
        margin_percent: Percentage of data to use for linear fits

    Returns:
        Tuple containing:
        - Transformed fraction array
        - Extrapolated strain values before transformation
        - Extrapolated strain values after transformation

    Raises:
        ValueError: If temperatures are invalid or if insufficient data for fitting
    """
    # Input validation
    if start_temp >= end_temp:
        raise ValueError("Start temperature must be less than end temperature")
    if not (temperature.min() <= start_temp <= temperature.max()):
        raise ValueError("Start temperature outside data range")
    if not (temperature.min() <= end_temp <= temperature.max()):
        raise ValueError("End temperature outside data range")

    # Get linear segments for before and after transformation
    temp_range = temperature.max() - temperature.min()
    fit_range = temp_range * margin_percent

    # Create masks for fitting regions, using margin_percent for fit range
    before_mask = (temperature >= temperature.min()) & (temperature <= (temperature.min() + fit_range))
    after_mask = (temperature <= temperature.max()) & (temperature >= (temperature.max() - fit_range))

    # Ensure enough points for fitting
    min_points = 5
    if np.sum(before_mask) < min_points or np.sum(after_mask) < min_points:
        raise ValueError(f"Insufficient points for fitting. Need at least {min_points} points in each region.")

    # Fit linear functions to segments
    try:
        before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
        after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)
    except np.linalg.LinAlgError:
        raise ValueError("Unable to perform linear fit on the data segments")

    # Calculate extrapolated values
    before_extrap = np.polyval(before_fit, temperature)
    after_extrap = np.polyval(after_fit, temperature)

    # Calculate transformed fraction
    transformed_fraction = np.zeros_like(strain)

    # Apply lever rule in transformation region
    mask = (temperature >= start_temp) & (temperature <= end_temp)
    height_total = after_extrap[mask] - before_extrap[mask]
    height_current = strain[mask] - before_extrap[mask]

    # Avoid division by zero
    valid_total = height_total != 0
    transformed_fraction[mask] = np.where(valid_total,
                                          height_current / height_total,
                                          0)

    # Set values outside transformation region
    transformed_fraction[temperature > end_temp] = 1.0
    transformed_fraction[temperature < start_temp] = 0.0

    return np.clip(transformed_fraction, 0, 1), before_extrap, after_extrap


# Analysis methods
def analyze_dilatometry_curve(temperature: np.ndarray, strain: np.ndarray,
                              method: str = 'lever',
                              margin_percent: float = 0.2) -> Dict[str, np.ndarray]:
    """
    Analyze the dilatometry curve to extract key parameters.

    Args:
        temperature: Array of temperature values
        strain: Array of strain values
        method: Analysis method ('lever' or 'tangent')
        margin_percent: Percentage of data to use for fitting linear segments

    Returns:
        Dictionary containing analysis results
    """
    if method == 'lever':
        return lever_method(temperature, strain, margin_percent)
    elif method == 'tangent':
        return tangent_method(temperature, strain, margin_percent)
    else:
        raise ValueError(f"Unsupported method: {method}")


def lever_method(temperature: np.ndarray, strain: np.ndarray,
                 margin_percent: float = 0.2) -> Dict[str, np.ndarray]:
    """
    Analyze dilatometry curve using the lever rule method.
    """
    # Find transformation points
    start_temp, end_temp = find_inflection_points(temperature, strain)

    # Calculate transformed fraction and extrapolations
    transformed_fraction, before_extrap, after_extrap = calculate_transformed_fraction_lever(
        temperature, strain, start_temp, end_temp, margin_percent)

    # Find mid-point temperature
    mid_temp = find_midpoint_temperature(temperature, transformed_fraction, start_temp, end_temp)

    return {
        'start_temperature': start_temp,
        'end_temperature': end_temp,
        'mid_temperature': mid_temp,
        'transformed_fraction': transformed_fraction,
        'before_extrapolation': before_extrap,
        'after_extrapolation': after_extrap
    }


def tangent_method(temperature: np.ndarray, strain: np.ndarray,
                   margin_percent: Optional[float] = None,
                   deviation_threshold: Optional[float] = None) -> Dict:
    """
    Analyze dilatometry curve using the tangent method.
    """
    # Ensure arrays are numpy arrays
    temperature = np.asarray(temperature)
    strain = np.asarray(strain)

    # Determine optimal margin if not provided
    if margin_percent is None:
        margin_percent = find_optimal_margin(temperature, strain)

    # Get linear segments and their fits
    start_mask, end_mask = get_linear_segment_masks(temperature, margin_percent)
    p_start, p_end = fit_linear_segments(temperature, strain, start_mask, end_mask)
    pred_start, pred_end = get_extrapolated_values(temperature, p_start, p_end)

    # Find transformation points
    start_idx, end_idx = find_transformation_points(
        temperature, strain, pred_start, pred_end,
        deviation_threshold, start_mask, end_mask)

    # Calculate transformed fraction
    transformed_fraction = calculate_transformed_fraction(
        strain, pred_start, pred_end, start_idx, end_idx)

    # Find middle point
    mid_temp = find_midpoint_temperature(
        temperature, transformed_fraction,
        temperature[start_idx], temperature[end_idx])

    # Calculate fit quality metrics
    fit_quality = calculate_fit_quality(
        temperature, strain, p_start, p_end,
        start_mask, end_mask, margin_percent, deviation_threshold)

    return {
        'start_temperature': temperature[start_idx],
        'end_temperature': temperature[end_idx],
        'mid_temperature': mid_temp,
        'transformed_fraction': transformed_fraction,
        'before_extrapolation': pred_start,
        'after_extrapolation': pred_end,
        'fit_quality': fit_quality
    }


# Core analysis functions
def find_inflection_points(temperature: np.ndarray, strain: np.ndarray) -> Tuple[float, float]:
    """Find inflection points using second derivative."""
    smooth_strain = smooth_data(strain)
    second_derivative = np.gradient(np.gradient(smooth_strain))
    peaks = np.argsort(np.abs(second_derivative))[-2:]
    start_temp, end_temp = temperature[min(peaks)], temperature[max(peaks)]
    return start_temp, end_temp


def find_midpoint_temperature(temperature: np.ndarray, transformed_fraction: np.ndarray,
                              start_temp: float, end_temp: float) -> float:
    """Find temperature at 50% transformation."""
    mask = (temperature >= start_temp) & (temperature <= end_temp)
    valid_fraction = transformed_fraction[mask]
    valid_temp = temperature[mask]
    mid_idx = np.argmin(np.abs(valid_fraction - 0.5))
    return valid_temp[mid_idx]


# Linear segment analysis functions
def get_linear_segment_masks(temperature: np.ndarray, margin_percent: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get masks for linear segments at start and end."""
    temp_range = temperature.max() - temperature.min()
    margin = temp_range * margin_percent
    start_mask = temperature <= (temperature.min() + margin)
    end_mask = temperature >= (temperature.max() - margin)
    return start_mask, end_mask


def fit_linear_segments(temperature: np.ndarray, strain: np.ndarray,
                        start_mask: np.ndarray, end_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit linear functions to start and end segments."""
    p_start = np.polyfit(temperature[start_mask], strain[start_mask], 1)
    p_end = np.polyfit(temperature[end_mask], strain[end_mask], 1)
    return p_start, p_end


def get_extrapolated_values(temperature: np.ndarray,
                            p_start: np.ndarray, p_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate extrapolated values using linear fits."""
    pred_start = np.polyval(p_start, temperature)
    pred_end = np.polyval(p_end, temperature)
    return pred_start, pred_end


# Transformation analysis functions
def find_transformation_points(temperature: np.ndarray, strain: np.ndarray,
                               pred_start: np.ndarray, pred_end: np.ndarray,
                               deviation_threshold: Optional[float],
                               start_mask: np.ndarray, end_mask: np.ndarray) -> Tuple[int, int]:
    """Find transformation start and end points."""
    if deviation_threshold is None:
        deviation_threshold = calculate_deviation_threshold(
            strain, pred_start, pred_end, start_mask, end_mask)

    dev_start = np.abs(strain - pred_start)
    dev_end = np.abs(strain - pred_end)

    window = max(int(len(temperature) * 0.05), 3)  # At least 3 points
    start_idx = find_deviation_point(dev_start > deviation_threshold, window, forward=True)
    end_idx = find_deviation_point(dev_end > deviation_threshold, window, forward=False)

    return start_idx, end_idx


def calculate_deviation_threshold(strain: np.ndarray, pred_start: np.ndarray,
                                  pred_end: np.ndarray, start_mask: np.ndarray,
                                  end_mask: np.ndarray) -> float:
    """Calculate threshold for deviation detection."""
    start_residuals = np.abs(strain[start_mask] - pred_start[start_mask])
    end_residuals = np.abs(strain[end_mask] - pred_end[end_mask])
    return 3 * max(np.std(start_residuals), np.std(end_residuals))


def find_deviation_point(deviations: np.ndarray, window: int, forward: bool = True) -> int:
    """Find point where deviation becomes significant."""
    if forward:
        cum_dev = np.convolve(deviations, np.ones(window) / window, mode='valid')
        return np.argmax(cum_dev > 0.8) + window // 2
    else:
        cum_dev = np.convolve(deviations[::-1], np.ones(window) / window, mode='valid')
        return len(deviations) - np.argmax(cum_dev > 0.8) - window // 2


def calculate_transformed_fraction(strain: np.ndarray, pred_start: np.ndarray,
                                   pred_end: np.ndarray, start_idx: int,
                                   end_idx: int) -> np.ndarray:
    """Calculate transformed fraction."""
    transformed_fraction = np.zeros_like(strain)
    transformation_region = slice(start_idx, end_idx + 1)

    height_total = pred_end[transformation_region] - pred_start[transformation_region]
    height_current = strain[transformation_region] - pred_start[transformation_region]
    transformed_fraction[transformation_region] = height_current / height_total
    transformed_fraction[end_idx + 1:] = 1.0

    return np.clip(transformed_fraction, 0, 1)


# Quality assessment functions
def calculate_fit_quality(temperature: np.ndarray, strain: np.ndarray,
                          p_start: np.ndarray, p_end: np.ndarray,
                          start_mask: np.ndarray, end_mask: np.ndarray,
                          margin_percent: float,
                          deviation_threshold: float) -> Dict:
    """Calculate quality metrics for the analysis."""
    r2_start = calculate_r2(temperature[start_mask], strain[start_mask], p_start)
    r2_end = calculate_r2(temperature[end_mask], strain[end_mask], p_end)

    return {
        'r2_start': r2_start,
        'r2_end': r2_end,
        'margin_used': margin_percent,
        'deviation_threshold': deviation_threshold
    }


def calculate_r2(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Calculate R² value for a linear fit."""
    y_pred = np.polyval(p, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)
