"""The linear baselines on either side of a transformation: which points
they are fitted on, the fits themselves, and how well they hold."""

import warnings as py_warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pkynetics.technique_analysis.utilities import detect_segment_direction

from .types import FitQuality, ReturnDict
from .utilities import calculate_r2


def get_linear_segment_masks(
    temperature: NDArray[np.float64], margin_percent: float, is_cooling: bool = False
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Get boolean masks for linear segments at start and end based on margin percentage."""
    temp_min, temp_max = min(temperature), max(temperature)
    temp_range = temp_max - temp_min

    if temp_range < 1e-6:  # Avoid issues with isothermal data
        raise ValueError("Temperature range is too small to define margins.")

    margin_value = temp_range * margin_percent

    if is_cooling:
        # For cooling: initial segment at high temps, final at low temps
        # Use >= and <= to be inclusive of boundaries if margin is large
        start_mask = temperature >= (temp_max - margin_value)
        end_mask = temperature <= (temp_min + margin_value)
    else:
        # For heating: initial segment at low temps, final at high temps
        start_mask = temperature <= (temp_min + margin_value)
        end_mask = temperature >= (temp_max - margin_value)

    return start_mask, end_mask


def fit_linear_segments(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    start_mask: NDArray[np.bool_],
    end_mask: NDArray[np.bool_],
    min_points_fit: int = 5,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit linear functions (degree 1 polynomials) to start and end segments.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        start_mask: Boolean mask for the starting linear segment.
        end_mask: Boolean mask for the ending linear segment.
        min_points_fit: Minimum number of points required in each segment for fitting.

    Returns:
        Tuple containing polynomial coefficients for start and end fits.

    Raises:
        ValueError: If insufficient points are available in either segment.
        np.linalg.LinAlgError: If the linear fit fails mathematically.
    """
    n_start: int = int(np.sum(start_mask))
    n_end: int = int(np.sum(end_mask))

    if n_start < min_points_fit:
        raise ValueError(
            f"Insufficient points in start segment ({n_start}) for linear fit. Need at least {min_points_fit}."
        )
    if n_end < min_points_fit:
        raise ValueError(
            f"Insufficient points in end segment ({n_end}) for linear fit. Need at least {min_points_fit}."
        )

    try:
        p_start = np.polyfit(temperature[start_mask], strain[start_mask], 1)
        p_end = np.polyfit(temperature[end_mask], strain[end_mask], 1)
        return p_start, p_end
    except (np.linalg.LinAlgError, ValueError) as e:
        # Reraise with more context if needed, or let the original error propagate
        raise ValueError(f"Linear fitting failed: {e}")


def get_extrapolated_values(
    temperature: NDArray[np.float64],
    p_start: NDArray[np.float64],
    p_end: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate extrapolated strain values across the full temperature range using linear fit coefficients."""
    pred_start = np.asarray(np.polyval(p_start, temperature), dtype=np.float64)
    pred_end = np.asarray(np.polyval(p_end, temperature), dtype=np.float64)
    return pred_start, pred_end


def extrapolate_linear_segments(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    start_temp: float,
    end_temp: float,
    min_points_fit: int = 5,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], np.poly1d, np.poly1d]:
    """
    Extrapolate linear segments before and after the transformation range.
    Note: This function is less used now, as fitting happens within lever/tangent methods.

    Args:
        temperature: Array of temperature values
        strain: Array of strain values
        start_temp: Start temperature of the transformation (higher for cooling)
        end_temp: End temperature of the transformation (lower for cooling)
        min_points_fit: Minimum points required for linear fitting.

    Returns:
        Tuple containing:
        - Extrapolated strain values before transformation
        - Extrapolated strain values after transformation
        - Polynomial function for before extrapolation
        - Polynomial function for after extrapolation

    Raises:
        ValueError: If temperatures are invalid, incorrectly ordered, or if insufficient data for fitting
    """
    # Basic validation of temperature range relative to data
    temp_min_data, temp_max_data = temperature.min(), temperature.max()
    if not (temp_min_data <= start_temp <= temp_max_data):
        raise ValueError(
            f"Start temperature {start_temp} outside data range [{temp_min_data}, {temp_max_data}]"
        )
    if not (temp_min_data <= end_temp <= temp_max_data):
        raise ValueError(
            f"End temperature {end_temp} outside data range [{temp_min_data}, {temp_max_data}]"
        )

    is_cooling = detect_segment_direction(temperature, strain)

    # Define masks based on direction and transformation temps
    if is_cooling:
        if start_temp <= end_temp:
            raise ValueError("For cooling, start_temp must be > end_temp")
        before_mask = temperature > start_temp  # High temp region
        after_mask = temperature < end_temp  # Low temp region
    else:
        if start_temp >= end_temp:
            raise ValueError("For heating, start_temp must be < end_temp")
        before_mask = temperature < start_temp  # Low temp region
        after_mask = temperature > end_temp  # High temp region

    # Check for sufficient points
    if np.sum(before_mask) < min_points_fit:
        raise ValueError(
            f"Insufficient points ({np.sum(before_mask)}) for fitting 'before' segment. Need at least {min_points_fit}."
        )
    if np.sum(after_mask) < min_points_fit:
        raise ValueError(
            f"Insufficient points ({np.sum(after_mask)}) for fitting 'after' segment. Need at least {min_points_fit}."
        )

    # Perform linear fits
    try:
        before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
        after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)

        before_extrapolation = np.poly1d(before_fit)
        after_extrapolation = np.poly1d(after_fit)

    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f"Unable to perform linear fit on the data segments: {e}")

    before_values = before_extrapolation(temperature)
    after_values = after_extrapolation(temperature)

    return before_values, after_values, before_extrapolation, after_extrapolation


def find_optimal_margin(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    is_cooling: bool,
    min_r2: float = 0.99,
    min_points_fit: int = 10,
) -> float:
    """
    Determine the optimal margin percentage for linear segment fitting based on R².

    Returns the widest margin whose linear fits both reach ``min_r2``, so that
    the baselines use as much of the linear regions as they can without
    reaching into the transformation.

    Args:
        temperature: Temperature data array
        strain: Strain data array
        is_cooling: Boolean indicating if it's a cooling segment.
        min_r2: Minimum average R² value for acceptable linear fit (default: 0.99)
        min_points_fit: Minimum number of points required for fitting each segment (default: 10)

    Returns:
        float: Optimal margin percentage (between 0.1 and 0.4)

    Raises:
        ValueError: If no acceptable margin is found or if data is insufficient
                     even with the largest margin.
    """
    if len(temperature) < min_points_fit * 2:
        raise ValueError(
            f"Insufficient data points ({len(temperature)}). Need at least {min_points_fit * 2} points for optimal margin search."
        )

    margins = np.linspace(0.1, 0.4, 7)  # Test margins from 10% to 40%
    best_margin: Optional[float] = None
    highest_avg_r2: float = (
        -1.0
    )  # Keep track of the highest R2 found, even if below min_r2

    candidate_margins = []

    for margin in margins:
        try:
            start_mask, end_mask = get_linear_segment_masks(
                temperature, margin, is_cooling
            )

            # Check if enough points are selected by this margin
            n_start: int = int(np.sum(start_mask))
            n_end: int = int(np.sum(end_mask))

            if n_start < min_points_fit or n_end < min_points_fit:
                # py_warnings.warn(f"Margin {margin:.1%} yields insufficient points ({n_start}, {n_end} vs min {min_points_fit}). Skipping.", UserWarning)
                continue  # Skip this margin if it doesn't provide enough points

            # Attempt fitting
            p_start, p_end = fit_linear_segments(
                temperature, strain, start_mask, end_mask, min_points_fit
            )  # fit_linear_segments now checks points

            # Calculate R² for both fits
            r2_start = calculate_r2(
                temperature[start_mask], strain[start_mask], p_start
            )
            r2_end = calculate_r2(temperature[end_mask], strain[end_mask], p_end)

            avg_r2 = (r2_start + r2_end) / 2

            if avg_r2 > highest_avg_r2:
                highest_avg_r2 = avg_r2
                best_margin = margin

            # Store margins that meet the R² criteria
            if avg_r2 >= min_r2:
                candidate_margins.append({"margin": margin, "avg_r2": avg_r2})

        except (np.linalg.LinAlgError, ValueError) as e:
            # Ignore margins that cause fitting errors (e.g., due to singular matrix)
            # py_warnings.warn(f"Fitting failed for margin {margin:.1%}: {e}. Skipping.", UserWarning)
            continue

    if candidate_margins:
        # Among the margins that meet the R² criterion, take the widest: a
        # smaller window always fits a straight line better, so picking the
        # highest R² would systematically use the least baseline available,
        # which is the opposite of what makes the limits robust.
        best_candidate = max(candidate_margins, key=lambda x: x["margin"])
        return float(best_candidate["margin"])
    elif best_margin is not None:
        # If no margin met min_r2, but at least one fit was possible, return the one with highest R² found
        py_warnings.warn(
            f"No margin found meeting the minimum R² requirement ({min_r2:.3f}). "
            f"Using margin {best_margin:.1%} with the highest found average R² ({highest_avg_r2:.3f}). "
            f"Consider adjusting 'min_r2' or checking data quality.",
            UserWarning,
        )
        return float(best_margin)
    else:
        # If no margin allowed fitting (e.g., always insufficient points even at 40%)
        raise ValueError(
            f"Could not find a suitable margin ({margins.min():.1%} to {margins.max():.1%}) "
            f"providing at least {min_points_fit} points for linear fitting in both segments. "
            f"Check data length and quality."
        )


def calculate_fit_quality(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    p_start: NDArray[np.float64],  # Coefficients for start fit
    p_end: NDArray[np.float64],  # Coefficients for end fit
    start_mask: NDArray[np.bool_],  # Mask used for start fit
    end_mask: NDArray[np.bool_],  # Mask used for end fit
    margin_percent: float,
    deviation_fraction: float,
    existing_warnings: Optional[List[str]] = None,
    r2_warn_threshold: float = 0.98,
) -> FitQuality:
    """
    Calculate quality metrics for the tangent method analysis, including R² values
    and checks for potential issues.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        p_start: Coefficients of the linear fit for the start segment.
        p_end: Coefficients of the linear fit for the end segment.
        start_mask: Boolean mask for the start segment data points.
        end_mask: Boolean mask for the end segment data points.
        margin_percent: The margin percentage used for fitting.
        deviation_fraction: The fraction of the peak derivative excursion used
            to locate the transformation limits.
        existing_warnings: List of warnings generated earlier in the process.
        r2_warn_threshold: R² value below which a warning is generated.

    Returns:
        :class:`FitQuality` with the R² values, the margin, the deviation
        fraction and the warnings raised along the way.
    """
    warnings_list = list(existing_warnings) if existing_warnings is not None else []

    # Calculate R² for the start segment fit
    r2_start = np.nan
    if np.sum(start_mask) > 1:  # Need at least 2 points for R²
        try:
            r2_start = calculate_r2(
                temperature[start_mask], strain[start_mask], p_start
            )
            if r2_start < r2_warn_threshold:
                warnings_list.append(
                    f"R² for start segment fit ({r2_start:.3f}) is below threshold ({r2_warn_threshold}). Fit may be poor."
                )
        except (
            ValueError
        ):  # Handle potential issues in calculate_r2 (e.g., constant data)
            warnings_list.append("Could not calculate R² for the start segment fit.")

    # Calculate R² for the end segment fit
    r2_end = np.nan
    if np.sum(end_mask) > 1:  # Need at least 2 points for R²
        try:
            r2_end = calculate_r2(temperature[end_mask], strain[end_mask], p_end)
            if r2_end < r2_warn_threshold:
                warnings_list.append(
                    f"R² for end segment fit ({r2_end:.3f}) is below threshold ({r2_warn_threshold}). Fit may be poor."
                )
        except ValueError:  # Handle potential issues in calculate_r2
            warnings_list.append("Could not calculate R² for the end segment fit.")

    return FitQuality(
        r2_start=float(r2_start),
        r2_end=float(r2_end),
        margin_used=float(margin_percent),
        deviation_fraction=float(deviation_fraction),
        warnings=warnings_list,
    )
