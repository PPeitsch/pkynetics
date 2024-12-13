from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pkynetics.data_preprocessing.common_preprocessing import smooth_data

ReturnDict = Dict[str, Union[float, NDArray[np.float64], Dict[str, float]]]


def extrapolate_linear_segments(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    start_temp: float,
    end_temp: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], np.poly1d, np.poly1d]:
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
    if start_temp >= end_temp:
        raise ValueError("Start temperature must be less than end temperature")
    if not (temperature.min() <= start_temp <= temperature.max()):
        raise ValueError("Start temperature outside data range")
    if not (temperature.min() <= end_temp <= temperature.max()):
        raise ValueError("End temperature outside data range")

    before_mask = temperature < start_temp
    after_mask = temperature > end_temp

    min_points = 5
    if np.sum(before_mask) < min_points or np.sum(after_mask) < min_points:
        raise ValueError(
            f"Insufficient points for fitting. Need at least {min_points} points in each region."
        )

    try:
        before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
        after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)

        before_extrapolation = np.poly1d(before_fit)
        after_extrapolation = np.poly1d(after_fit)

    except np.linalg.LinAlgError:
        raise ValueError("Unable to perform linear fit on the data segments")

    before_values = before_extrapolation(temperature)
    after_values = after_extrapolation(temperature)

    return before_values, after_values, before_extrapolation, after_extrapolation


def find_optimal_margin(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    min_r2: float = 0.99,
    min_points: int = 10,
) -> float:
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
    if len(temperature) < min_points * 2:
        raise ValueError(
            f"Insufficient data points. Need at least {min_points * 2} points."
        )

    margins = np.linspace(0.1, 0.4, 7)  # Test margins from 10% to 40%
    best_margin: Optional[float] = None
    best_r2: float = 0.0

    for margin in margins:
        n_points = int(len(temperature) * margin)
        if n_points < min_points:
            continue

        start_mask = temperature <= (
            temperature.min() + (temperature.max() - temperature.min()) * margin
        )
        if np.sum(start_mask) < min_points:
            continue

        end_mask = temperature >= (
            temperature.max() - (temperature.max() - temperature.min()) * margin
        )
        if np.sum(end_mask) < min_points:
            continue

        try:
            p_start = np.polyfit(temperature[start_mask], strain[start_mask], 1)
            r2_start = calculate_r2(
                temperature[start_mask], strain[start_mask], p_start
            )

            p_end = np.polyfit(temperature[end_mask], strain[end_mask], 1)
            r2_end = calculate_r2(temperature[end_mask], strain[end_mask], p_end)

            avg_r2 = (r2_start + r2_end) / 2

            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_margin = margin

        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_margin is None or best_r2 < min_r2:
        if best_margin is not None:
            return float(best_margin)
        return 0.2

    return float(best_margin)


def calculate_transformed_fraction_lever(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    start_temp: float,
    end_temp: float,
    margin_percent: float = 0.2,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate transformed fraction using the lever rule method."""
    if start_temp >= end_temp:
        raise ValueError("Start temperature must be less than end temperature")
    if not (temperature.min() <= start_temp <= temperature.max()):
        raise ValueError("Start temperature outside data range")
    if not (temperature.min() <= end_temp <= temperature.max()):
        raise ValueError("End temperature outside data range")

    temp_range = temperature.max() - temperature.min()
    fit_range = temp_range * margin_percent

    before_mask = (temperature >= temperature.min()) & (
        temperature <= (temperature.min() + fit_range)
    )
    after_mask = (temperature <= temperature.max()) & (
        temperature >= (temperature.max() - fit_range)
    )

    min_points = 5
    if np.sum(before_mask) < min_points or np.sum(after_mask) < min_points:
        raise ValueError(
            f"Insufficient points for fitting. Need at least {min_points} points in each region."
        )

    try:
        before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
        after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)
    except np.linalg.LinAlgError:
        raise ValueError("Unable to perform linear fit on the data segments")

    before_extrap = np.polyval(before_fit, temperature)
    after_extrap = np.polyval(after_fit, temperature)

    transformed_fraction = np.zeros_like(strain)

    mask = (temperature >= start_temp) & (temperature <= end_temp)
    height_total = after_extrap[mask] - before_extrap[mask]
    height_current = strain[mask] - before_extrap[mask]

    valid_total = height_total != 0
    transformed_fraction[mask] = np.where(valid_total, height_current / height_total, 0)

    transformed_fraction[temperature > end_temp] = 1.0
    transformed_fraction[temperature < start_temp] = 0.0

    return np.clip(transformed_fraction, 0, 1), before_extrap, after_extrap


def analyze_dilatometry_curve(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    method: str = "lever",
    margin_percent: float = 0.2,
) -> ReturnDict:
    """Analyze the dilatometry curve to extract key parameters."""
    if method == "lever":
        return lever_method(temperature, strain, margin_percent)
    elif method == "tangent":
        return tangent_method(temperature, strain, margin_percent)
    else:
        raise ValueError(f"Unsupported method: {method}")


def lever_method(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    margin_percent: float = 0.2,
) -> ReturnDict:
    """Analyze dilatometry curve using the lever rule method."""
    start_temp, end_temp = find_inflection_points(temperature, strain)

    transformed_fraction, before_extrap, after_extrap = (
        calculate_transformed_fraction_lever(
            temperature, strain, start_temp, end_temp, margin_percent
        )
    )

    mid_temp = find_midpoint_temperature(
        temperature, transformed_fraction, start_temp, end_temp
    )

    return {
        "start_temperature": float(start_temp),
        "end_temperature": float(end_temp),
        "mid_temperature": float(mid_temp),
        "transformed_fraction": transformed_fraction,
        "before_extrapolation": before_extrap,
        "after_extrapolation": after_extrap,
    }


def tangent_method(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    margin_percent: Optional[float] = None,
    deviation_threshold: Optional[float] = None,
) -> ReturnDict:
    """Analyze dilatometry curve using the tangent method."""
    temperature = np.asarray(temperature)
    strain = np.asarray(strain)

    if margin_percent is None:
        margin_percent = find_optimal_margin(temperature, strain)

    start_mask, end_mask = get_linear_segment_masks(temperature, float(margin_percent))
    p_start, p_end = fit_linear_segments(temperature, strain, start_mask, end_mask)
    pred_start, pred_end = get_extrapolated_values(temperature, p_start, p_end)

    final_deviation_threshold = float(
        deviation_threshold
        if deviation_threshold is not None
        else calculate_deviation_threshold(
            strain, pred_start, pred_end, start_mask, end_mask
        )
    )

    start_idx, end_idx = find_transformation_points(
        temperature, strain, pred_start, pred_end, final_deviation_threshold
    )

    transformed_fraction = calculate_transformed_fraction(
        strain, pred_start, pred_end, start_idx, end_idx
    )

    mid_temp = find_midpoint_temperature(
        temperature, transformed_fraction, temperature[start_idx], temperature[end_idx]
    )

    fit_quality = calculate_fit_quality(
        temperature,
        strain,
        p_start,
        p_end,
        start_mask,
        end_mask,
        float(margin_percent),
        final_deviation_threshold,
    )

    return {
        "start_temperature": float(temperature[start_idx]),
        "end_temperature": float(temperature[end_idx]),
        "mid_temperature": float(mid_temp),
        "transformed_fraction": transformed_fraction,
        "before_extrapolation": pred_start,
        "after_extrapolation": pred_end,
        "fit_quality": fit_quality,
    }


def find_inflection_points(
    temperature: NDArray[np.float64], strain: NDArray[np.float64]
) -> Tuple[float, float]:
    """Find inflection points using second derivative."""
    smooth_strain = smooth_data(strain)
    second_derivative = np.gradient(np.gradient(smooth_strain))
    peaks = np.argsort(np.abs(second_derivative))[-2:]
    start_temp = float(temperature[min(peaks)])
    end_temp = float(temperature[max(peaks)])
    return start_temp, end_temp


def find_midpoint_temperature(
    temperature: NDArray[np.float64],
    transformed_fraction: NDArray[np.float64],
    start_temp: float,
    end_temp: float,
) -> float:
    """Find temperature at 50% transformation."""
    mask = (temperature >= start_temp) & (temperature <= end_temp)
    valid_fraction = transformed_fraction[mask]
    valid_temp = temperature[mask]
    mid_idx = np.argmin(np.abs(valid_fraction - 0.5))
    return float(valid_temp[mid_idx])


def get_linear_segment_masks(
    temperature: NDArray[np.float64], margin_percent: float
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Get masks for linear segments at start and end."""
    temp_range = temperature.max() - temperature.min()
    margin = temp_range * margin_percent
    start_mask = temperature <= (temperature.min() + margin)
    end_mask = temperature >= (temperature.max() - margin)
    return start_mask, end_mask


def fit_linear_segments(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    start_mask: NDArray[np.bool_],
    end_mask: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit linear functions to start and end segments."""
    p_start = np.polyfit(temperature[start_mask], strain[start_mask], 1)
    p_end = np.polyfit(temperature[end_mask], strain[end_mask], 1)
    return p_start, p_end


def get_extrapolated_values(
    temperature: NDArray[np.float64],
    p_start: NDArray[np.float64],
    p_end: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate extrapolated values using linear fits."""
    pred_start = np.polyval(p_start, temperature)
    pred_end = np.polyval(p_end, temperature)
    return pred_start, pred_end


def find_transformation_points(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    pred_start: NDArray[np.float64],
    pred_end: NDArray[np.float64],
    deviation_threshold: float,
) -> Tuple[int, int]:
    """Find transformation start and end points.

    Args:
        temperature: Array of temperature values
        strain: Array of strain values
        pred_start: Predicted values from start linear fit
        pred_end: Predicted values from end linear fit
        deviation_threshold: Threshold for detecting significant deviations

    Returns:
        Tuple containing start and end indices of the transformation region
    """
    dev_start = np.abs(strain - pred_start)
    dev_end = np.abs(strain - pred_end)

    window = max(int(len(temperature) * 0.05), 3)  # At least 3 points
    start_idx = find_deviation_point(
        dev_start > deviation_threshold, window, forward=True
    )
    end_idx = find_deviation_point(dev_end > deviation_threshold, window, forward=False)

    return start_idx, end_idx


def calculate_deviation_threshold(
    strain: NDArray[np.float64],
    pred_start: NDArray[np.float64],
    pred_end: NDArray[np.float64],
    start_mask: NDArray[np.bool_],
    end_mask: NDArray[np.bool_],
) -> float:
    """Calculate threshold for deviation detection."""
    start_residuals = np.abs(strain[start_mask] - pred_start[start_mask])
    end_residuals = np.abs(strain[end_mask] - pred_end[end_mask])
    return float(3 * max(float(np.std(start_residuals)), float(np.std(end_residuals))))


def find_deviation_point(
    deviations: NDArray[np.bool_], window: int, forward: bool = True
) -> int:
    """Find point where deviation becomes significant."""
    if forward:
        cum_dev = np.convolve(deviations, np.ones(window) / window, mode="valid")
        return int(np.argmax(cum_dev > 0.8) + window // 2)
    else:
        cum_dev = np.convolve(deviations[::-1], np.ones(window) / window, mode="valid")
        return int(len(deviations) - np.argmax(cum_dev > 0.8) - window // 2)


def calculate_transformed_fraction(
    strain: NDArray[np.float64],
    pred_start: NDArray[np.float64],
    pred_end: NDArray[np.float64],
    start_idx: int,
    end_idx: int,
) -> NDArray[np.float64]:
    """Calculate transformed fraction."""
    transformed_fraction = np.zeros_like(strain)
    transformation_region = slice(start_idx, end_idx + 1)

    height_total = pred_end[transformation_region] - pred_start[transformation_region]
    height_current = strain[transformation_region] - pred_start[transformation_region]
    transformed_fraction[transformation_region] = height_current / height_total
    transformed_fraction[end_idx + 1 :] = 1.0

    return np.clip(transformed_fraction, 0, 1)


def calculate_fit_quality(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    p_start: NDArray[np.float64],
    p_end: NDArray[np.float64],
    start_mask: NDArray[np.bool_],
    end_mask: NDArray[np.bool_],
    margin_percent: float,
    deviation_threshold: float,
) -> Dict[str, float]:
    """Calculate quality metrics for the analysis."""
    r2_start = float(calculate_r2(temperature[start_mask], strain[start_mask], p_start))
    r2_end = float(calculate_r2(temperature[end_mask], strain[end_mask], p_end))

    return {
        "r2_start": r2_start,
        "r2_end": r2_end,
        "margin_used": float(margin_percent),
        "deviation_threshold": float(deviation_threshold),
    }


def calculate_r2(
    x: NDArray[np.float64], y: NDArray[np.float64], p: NDArray[np.float64]
) -> float:
    """Calculate R² value for a linear fit."""
    y_pred = np.polyval(p, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(1 - (ss_res / ss_tot))
