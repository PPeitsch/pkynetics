import warnings as py_warnings  # Use standard warnings library
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pkynetics.data_preprocessing.common_preprocessing import smooth_data
from pkynetics.technique_analysis.utilities import detect_segment_direction

# Type hint for the dictionary returned by analysis functions
ReturnDict = Dict[
    str,
    Union[float, bool, str, NDArray[np.float64], Dict[str, Union[float, List[str]]]],
]


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


def calculate_transformed_fraction_lever(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    start_temp: float,
    end_temp: float,
    margin_percent: float = 0.2,
    min_points_fit: int = 5,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate transformed fraction using the lever rule method with extrapolated baselines.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        start_temp: Transformation start temperature (higher for cooling).
        end_temp: Transformation end temperature (lower for cooling).
        margin_percent: Fraction of temperature range used to define linear regions for fitting.
        min_points_fit: Minimum points required for linear fitting.

    Returns:
        Tuple containing:
        - Transformed fraction (0 to 1).
        - Extrapolated baseline from the 'before' transformation state.
        - Extrapolated baseline from the 'after' transformation state.

    Raises:
        ValueError: If temperature range is invalid or fitting fails.
    """
    # Detect direction
    is_cooling = detect_segment_direction(temperature, strain)

    # Validate temperature range relative to data and direction
    temp_min_data, temp_max_data = min(temperature), max(temperature)
    if not (
        temp_min_data <= start_temp <= temp_max_data
        and temp_min_data <= end_temp <= temp_max_data
    ):
        raise ValueError(
            "Transformation temperatures are outside the data's temperature range."
        )
    if is_cooling and start_temp <= end_temp:
        raise ValueError("For cooling, start_temp must be greater than end_temp.")
    if not is_cooling and start_temp >= end_temp:
        raise ValueError("For heating, start_temp must be less than end_temp.")

    # Determine regions for linear fitting using margin_percent
    start_mask_fit, end_mask_fit = get_linear_segment_masks(
        temperature, margin_percent, is_cooling
    )

    # Perform linear fitting (will raise ValueError if insufficient points)
    before_fit_coeffs, after_fit_coeffs = fit_linear_segments(
        temperature, strain, start_mask_fit, end_mask_fit, min_points_fit
    )

    # Calculate extrapolations across the entire temperature range
    before_extrap = np.polyval(before_fit_coeffs, temperature)
    after_extrap = np.polyval(after_fit_coeffs, temperature)

    # Initialize fraction array
    transformed_fraction = np.zeros_like(strain)

    # Define the actual transformation region mask based on start/end temps
    if is_cooling:
        # Start temp is higher, end temp is lower
        transform_mask = (temperature <= start_temp) & (temperature >= end_temp)
        # Before transformation region (higher temps)
        before_transform_mask = temperature > start_temp
        # After transformation region (lower temps)
        after_transform_mask = temperature < end_temp
    else:
        # Start temp is lower, end temp is higher
        transform_mask = (temperature >= start_temp) & (temperature <= end_temp)
        # Before transformation region (lower temps)
        before_transform_mask = temperature < start_temp
        # After transformation region (higher temps)
        after_transform_mask = temperature > end_temp

    # Calculate fraction within the transformation region using the lever rule
    height_total = after_extrap[transform_mask] - before_extrap[transform_mask]
    height_current = strain[transform_mask] - before_extrap[transform_mask]

    # Avoid division by zero if baselines coincide
    valid_total = np.abs(height_total) > 1e-9  # Use a small tolerance

    # Calculate raw fraction
    raw_fraction = np.zeros_like(height_current)
    np.divide(height_current, height_total, out=raw_fraction, where=valid_total)

    # Assign fraction based on direction
    if is_cooling:
        # For cooling, fraction goes from 1 (high temp state) to 0 (low temp state)
        # The lever rule calculation gives fraction of the 'after' state (low temp phase)
        # So, we need 1 - raw_fraction if we define fraction as the new phase forming.
        # Let's define transformed_fraction as the fraction of the low-temperature phase.
        # If raw_fraction is calculated as (strain - high_T_baseline) / (low_T_baseline - high_T_baseline)
        transformed_fraction[transform_mask] = raw_fraction
        transformed_fraction[before_transform_mask] = 0.0  # Fully high-T phase
        transformed_fraction[after_transform_mask] = 1.0  # Fully low-T phase
    else:
        # For heating, fraction goes from 0 (low temp state) to 1 (high temp state)
        # The lever rule calculation gives fraction of the 'after' state (high temp phase)
        transformed_fraction[transform_mask] = raw_fraction
        transformed_fraction[before_transform_mask] = 0.0  # Fully low-T phase
        transformed_fraction[after_transform_mask] = 1.0  # Fully high-T phase

    # Clip values to ensure they are strictly within [0, 1] due to potential noise/extrapolation issues
    return (
        np.clip(transformed_fraction, 0, 1),
        np.asarray(before_extrap, dtype=np.float64),
        np.asarray(after_extrap, dtype=np.float64),
    )


def analyze_dilatometry_curve(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    method: str = "lever",
    margin_percent: Optional[float] = None,
    find_inflection_margin: float = 0.2,
    min_points_fit: int = 10,
    min_r2_optimal_margin: float = 0.99,
    deviation_fraction: float = 0.05,
) -> ReturnDict:
    """
    Analyze the dilatometry curve to extract key transformation parameters.

    Args:
        temperature: Array of temperature values (°C).
        strain: Array of strain or relative length change values.
        method: Analysis method ('lever' or 'tangent'). Default is 'lever'.
        margin_percent: Margin percentage (0.0 to 1.0) for fitting linear segments
                        (used by both methods). If None for tangent, optimal margin is found.
                        Default for lever is often implicitly 0.2 or uses find_inflection_margin.
        find_inflection_margin: Margin percentage (0.1-0.4) used specifically by the
                                'lever' method's `find_inflection_points` function. Default is 0.2.
        min_points_fit: Minimum number of points required for reliable linear fitting
                        in tangent/lever methods. Default is 10.
        min_r2_optimal_margin: Minimum R² required when using `find_optimal_margin`
                               in the tangent method. Default is 0.99.
        deviation_fraction: Fraction of the peak excursion of ``dS/dT`` that still
            counts as transforming, for both methods. Default is 0.05.

    Returns:
        Dictionary containing analysis results: start, end, mid temperatures,
        transformed fraction, extrapolations, quality metrics (for tangent), etc.

    Raises:
        ValueError: If method is not supported, data is insufficient, or analysis fails.
    """
    if len(temperature) != len(strain):
        raise ValueError("Temperature and strain arrays must have the same length.")
    if len(temperature) < max(
        20, min_points_fit * 2
    ):  # Need a reasonable number of points overall
        raise ValueError(f"Insufficient data points ({len(temperature)}) for analysis.")

    # Ensure input arrays are numpy arrays
    temperature = np.asarray(temperature, dtype=np.float64)
    strain = np.asarray(strain, dtype=np.float64)

    # Detect direction early on
    is_cooling = detect_segment_direction(temperature, strain)

    # --- Method Dispatch ---
    if method.lower() == "lever":
        # Use find_inflection_margin for finding points, and a separate margin (or default) for fraction calc
        lever_margin = (
            margin_percent if margin_percent is not None else 0.2
        )  # Default margin for fraction calc if not given
        return lever_method(
            temperature,
            strain,
            is_cooling=is_cooling,
            margin_percent_fraction=lever_margin,
            find_inflection_margin=find_inflection_margin,
            min_points_fit=min_points_fit,
        )
    elif method.lower() == "tangent":
        return tangent_method(
            temperature,
            strain,
            is_cooling=is_cooling,
            margin_percent=margin_percent,  # Can be None to trigger optimal search
            deviation_fraction=deviation_fraction,
            min_points_fit=min_points_fit,
            min_r2_optimal_margin=min_r2_optimal_margin,
        )
    else:
        raise ValueError(
            f"Unsupported method: '{method}'. Choose 'lever' or 'tangent'."
        )


def lever_method(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    is_cooling: bool,
    margin_percent_fraction: float = 0.2,
    find_inflection_margin: float = 0.2,
    min_points_fit: int = 5,  # Min points for fraction calculation fit
) -> ReturnDict:
    """
    Analyze dilatometry curve using the lever rule method.
    Locates the transformation limits on the derivative of the strain, using
    'find_inflection_margin' to define the baseline segments.
    Calculates transformed fraction using tangents fitted using 'margin_percent_fraction'.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        is_cooling: Boolean indicating direction.
        margin_percent_fraction: Margin percentage for fitting baselines for fraction calculation (0.1-0.4).
        find_inflection_margin: Margin percentage for finding inflection points (0.1-0.4).
        min_points_fit: Minimum points for the linear fits used in fraction calculation.

    Returns:
        Dictionary containing analysis results.
    """
    # 1. Find transformation start and end points using the specified inflection margin
    start_temp, end_temp = find_inflection_points(
        temperature, strain, is_cooling, margin=find_inflection_margin
    )

    # 2. Calculate transformed fraction using baselines fitted with margin_percent_fraction
    transformed_fraction, before_extrap, after_extrap = (
        calculate_transformed_fraction_lever(
            temperature,
            strain,
            start_temp,
            end_temp,
            margin_percent=margin_percent_fraction,
            min_points_fit=min_points_fit,
        )
    )

    # 3. Find midpoint temperature (T50%)
    mid_temp = find_midpoint_temperature(
        temperature, transformed_fraction, start_temp, end_temp, is_cooling
    )

    return {
        "method": "lever",
        "start_temperature": float(start_temp),
        "end_temperature": float(end_temp),
        "mid_temperature": float(mid_temp),
        "transformed_fraction": transformed_fraction,
        "temperature": temperature,  # Include temperature for context
        "strain": strain,  # Include strain for context
        "before_extrapolation": before_extrap,
        "after_extrapolation": after_extrap,
        "is_cooling": is_cooling,
        "parameters": {  # Store parameters used
            "margin_percent_fraction": margin_percent_fraction,
            "find_inflection_margin": find_inflection_margin,
            "min_points_fit": min_points_fit,
        },
    }


def tangent_method(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    is_cooling: bool,
    margin_percent: Optional[float] = None,
    deviation_fraction: float = 0.05,
    limits_margin: float = 0.2,
    min_points_fit: int = 10,
    min_r2_optimal_margin: float = 0.99,
) -> ReturnDict:
    """
    Analyze dilatometry curve using the tangent intersection method.
    Fits tangents based on 'margin_percent' (or finds optimal), and locates the
    transformation limits on the derivative of the strain.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        is_cooling: Boolean indicating direction.
        margin_percent: Margin for fitting tangents. If None, finds optimal margin.
        deviation_fraction: Fraction of the peak excursion of ``dS/dT`` that still
            counts as transforming, passed to :func:`find_transformation_limits`.
        limits_margin: Fraction of the data at each end taken as baseline when
            locating the transformation limits.
        min_points_fit: Minimum points for tangent fitting.
        min_r2_optimal_margin: Minimum R² for optimal margin search.

    Returns:
        Dictionary containing analysis results including fit quality.
    """
    # 1. Determine margin for fitting tangents
    final_margin_percent: float
    if margin_percent is not None:
        if not (0.0 < margin_percent <= 0.5):
            raise ValueError(
                "margin_percent must be between 0 and 0.5 (exclusive of 0)"
            )
        final_margin_percent = margin_percent
    else:
        try:
            final_margin_percent = find_optimal_margin(
                temperature, strain, is_cooling, min_r2_optimal_margin, min_points_fit
            )
            # Warning if optimal margin search returned a value below min_r2 is handled inside find_optimal_margin
        except ValueError as e:
            raise ValueError(f"Failed to find optimal margin: {e}")

    # 2. Get masks and fit linear segments (tangents)
    start_mask, end_mask = get_linear_segment_masks(
        temperature, final_margin_percent, is_cooling
    )
    try:
        p_start, p_end = fit_linear_segments(
            temperature, strain, start_mask, end_mask, min_points_fit
        )
    except ValueError as e:
        raise ValueError(
            f"Failed to fit linear segments using margin {final_margin_percent:.1%}: {e}"
        )

    # 3. Get extrapolated values (full range)
    pred_start, pred_end = get_extrapolated_values(temperature, p_start, p_end)

    # 4. Locate the transformation on the derivative of the strain.
    # Deliberately not `final_margin_percent`: that margin is chosen to fit the
    # tangents used for the transformed fraction and can be wide enough to reach
    # into the transformation, which would bias the baseline slope.
    start_idx, end_idx = find_transformation_limits(
        temperature,
        strain,
        is_cooling=is_cooling,
        margin=limits_margin,
        deviation_fraction=deviation_fraction,
    )

    start_temp = temperature[start_idx]
    end_temp = temperature[end_idx]

    # Re-order start/end temperatures based on cooling/heating direction for reporting
    if is_cooling:
        report_start_temp, report_end_temp = max(start_temp, end_temp), min(
            start_temp, end_temp
        )
    else:
        report_start_temp, report_end_temp = min(start_temp, end_temp), max(
            start_temp, end_temp
        )

    # 5. Calculate transformed fraction using the identified points and tangents
    transformed_fraction = calculate_transformed_fraction(
        strain, pred_start, pred_end, start_idx, end_idx, is_cooling
    )

    # 6. Find midpoint temperature (T50%)
    mid_temp = find_midpoint_temperature(
        temperature,
        transformed_fraction,
        report_start_temp,  # Use direction-aware temps
        report_end_temp,
        is_cooling,
    )

    # 7. Calculate fit quality metrics
    fit_quality = calculate_fit_quality(
        temperature,
        strain,
        p_start,
        p_end,
        start_mask,
        end_mask,
        final_margin_percent,
        deviation_fraction,
    )

    return {
        "method": "tangent",
        "start_temperature": float(report_start_temp),
        "end_temperature": float(report_end_temp),
        "mid_temperature": float(mid_temp),
        "transformed_fraction": transformed_fraction,
        "temperature": temperature,  # Include temperature for context
        "strain": strain,  # Include strain for context
        "before_extrapolation": pred_start,
        "after_extrapolation": pred_end,
        "fit_quality": fit_quality,
        "is_cooling": is_cooling,
        "parameters": {  # Store parameters used
            "margin_percent": final_margin_percent,
            "deviation_fraction": deviation_fraction,
            "min_points_fit": min_points_fit,
            "min_r2_optimal_margin": min_r2_optimal_margin,
        },
    }


# --- Helper Functions ---


def _smoothing_window(
    n_total: int,
    smooth_window_fraction: float,
    min_points_smooth: int,
) -> int:
    """Odd Savitzky-Golay window length for ``n_total`` points, or 0 if too short."""
    window_length = max(min_points_smooth, int(n_total * smooth_window_fraction))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, n_total - 2)
    return window_length if window_length >= 3 else 0


def _mad_scale(values: NDArray[np.float64]) -> float:
    """Median absolute deviation, scaled to be comparable to a std."""
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def _longest_run(flags: NDArray[np.bool_]) -> Tuple[int, int]:
    """First and last index of the longest run of True in ``flags``.

    Falls back to the middle 70 % of the array if nothing is flagged.
    """
    n_total = len(flags)
    best_start, best_end, best_length = -1, -1, 0
    run_start = -1
    for i in range(n_total):
        if flags[i]:
            if run_start < 0:
                run_start = i
        elif run_start >= 0:
            if i - run_start > best_length:
                best_start, best_end, best_length = run_start, i - 1, i - run_start
            run_start = -1
    if run_start >= 0 and n_total - run_start > best_length:
        best_start, best_end = run_start, n_total - 1

    if best_start < 0:
        return int(n_total * 0.15), int(n_total * 0.85)
    return best_start, best_end


def find_transformation_limits(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    is_cooling: bool = False,
    margin: float = 0.2,
    deviation_fraction: float = 0.05,
    smooth_window_fraction: float = 0.05,
    polyorder: int = 2,
    min_points_smooth: int = 5,
    baseline_min_r2: float = 0.99,
) -> Tuple[int, int]:
    """
    Find the indices where the transformation starts and ends, from the
    derivative of the strain with respect to temperature.

    The transformation is located on ``dS/dT`` rather than on the deviation of
    the strain from an extrapolated tangent. A real baseline is never exactly
    straight, and the *integral* of a slight curvature is indistinguishable
    from the beginning of a transformation, which is why deviation-based
    detection drifted 80-150 K early on real data. On the derivative the two
    separate: the curvature of the baseline is a small offset, the
    transformation a large, localised excursion.

    The limits are the points where the derivative comes back to within
    ``deviation_fraction`` of the peak excursion, walking outwards from that
    peak, so a single noisy point near the edge of the data cannot pull a
    limit onto the edge of the data.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        is_cooling: Whether this is a cooling segment.
        margin: Fraction of the data at each end taken as baseline (0.1-0.4).
        deviation_fraction: Fraction of the peak excursion of the derivative
            that still counts as transforming. 0.05 reproduces both a
            synthetic sigmoid and a real Zry-4 dilatometry run.
        smooth_window_fraction: Savitzky-Golay window as a fraction of length.
        polyorder: Polynomial order for smoothing.
        min_points_smooth: Minimum smoothing window length.
        baseline_min_r2: R² below which a baseline window is reported as not
            linear, i.e. as reaching into a transformation.

    Returns:
        Tuple ``(start_idx, end_idx)`` of indices into the input arrays, in
        array order (``start_idx < end_idx``), whatever the ramp direction.

    Raises:
        ValueError: If the arguments or the amount of data are invalid.
    """
    n_total = len(temperature)
    if n_total < 20:
        raise ValueError(
            f"Insufficient data points ({n_total}) to locate the transformation."
        )
    if not (0.0 < deviation_fraction < 1.0):
        raise ValueError("deviation_fraction must be between 0 and 1 (exclusive)")

    window_length = _smoothing_window(
        n_total, smooth_window_fraction, min_points_smooth
    )
    smooth_strain = strain
    if window_length:
        try:
            smooth_strain = smooth_data(
                strain, window_length=window_length, polyorder=polyorder
            )
        except ValueError:
            py_warnings.warn(
                "Smoothing the strain failed; locating the transformation on the "
                "raw signal.",
                UserWarning,
            )

    derivative = np.gradient(smooth_strain, temperature)
    if window_length:
        try:
            derivative = smooth_data(
                derivative, window_length=window_length, polyorder=polyorder
            )
        except ValueError:
            pass  # Unsmoothed derivative; the thresholds adapt to its noise

    start_mask, end_mask = get_linear_segment_masks(temperature, margin, is_cooling)
    if np.sum(start_mask) < 2 or np.sum(end_mask) < 2:
        raise ValueError(
            f"Margin {margin:.1%} leaves fewer than 2 points in a baseline segment."
        )

    # A baseline window that is not straight has reached into the
    # transformation (or into a second one), and every threshold below is
    # then measured against the wrong thing
    for mask, side in ((start_mask, "initial"), (end_mask, "final")):
        r2 = calculate_r2(
            temperature[mask],
            strain[mask],
            np.polyfit(temperature[mask], strain[mask], 1),
        )
        if r2 < baseline_min_r2:
            py_warnings.warn(
                f"The {side} baseline window is not linear (R² = {r2:.3f}). It "
                f"probably reaches into a transformation: narrow the analysis "
                f"range or lower `margin` (currently {margin:.1%}). The limits "
                f"below may be unreliable.",
                UserWarning,
            )

    # Deviation of the derivative from each baseline's own slope
    dev_start = np.abs(derivative - float(np.median(derivative[start_mask])))
    dev_end = np.abs(derivative - float(np.median(derivative[end_mask])))

    # A limit has to clear both the scatter of its own baseline and a fixed
    # fraction of the excursion, so neither a noisy nor a clean curve
    # degenerates. Both are robust estimators: a single spike in the raw data
    # survives smoothing as a large spike in the derivative, and would
    # otherwise set the scale for everything else.
    threshold_start = max(
        deviation_fraction * float(np.percentile(dev_start, 99.5)),
        3.0 * _mad_scale(derivative[start_mask]),
    )
    threshold_end = max(
        deviation_fraction * float(np.percentile(dev_end, 99.5)),
        3.0 * _mad_scale(derivative[end_mask]),
    )

    # The transformation is the longest run of points over the threshold, not
    # the first one: an isolated spike is a run of one or two points, and
    # walking outwards from the peak would start on it.
    start_idx = _longest_run(dev_start > threshold_start)[0]
    end_idx = _longest_run(dev_end > threshold_end)[1]

    if start_idx >= end_idx:
        py_warnings.warn(
            f"The transformation limits came out in the wrong order "
            f"(indices {start_idx}, {end_idx}). The two baselines may be picking "
            f"up the same feature. Using them in array order.",
            UserWarning,
        )
        start_idx, end_idx = min(start_idx, end_idx), max(start_idx, end_idx)
        if start_idx == end_idx:  # Degenerate: fall back to the search interval
            start_idx, end_idx = int(n_total * 0.15), int(n_total * 0.85)

    return start_idx, end_idx


def find_inflection_points(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    is_cooling: bool = False,
    margin: float = 0.2,
    smooth_window_fraction: float = 0.05,
    polyorder: int = 2,
    min_points_smooth: int = 5,
    deviation_fraction: float = 0.05,
) -> Tuple[float, float]:
    """
    Find the transformation start and end temperatures.

    Thin wrapper over :func:`find_transformation_limits` that returns
    temperatures ordered by the direction of the ramp.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        is_cooling: Whether this is a cooling segment.
        margin: Fraction of the data at each end taken as baseline (0.1-0.4).
        smooth_window_fraction: Savitzky-Golay window as a fraction of length.
        polyorder: Polynomial order for smoothing.
        min_points_smooth: Minimum smoothing window length.
        deviation_fraction: Fraction of the peak excursion of ``dS/dT`` that
            still counts as transforming.

    Returns:
        Tuple of start and end temperatures, ordered according to the
        heating/cooling convention (start > end for cooling).

    Raises:
        ValueError: If margin is invalid or the data are insufficient.
    """
    if not (0.1 <= margin <= 0.4):
        raise ValueError("Margin must be between 0.1 and 0.4")

    start_idx, end_idx = find_transformation_limits(
        temperature,
        strain,
        is_cooling=is_cooling,
        margin=margin,
        deviation_fraction=deviation_fraction,
        smooth_window_fraction=smooth_window_fraction,
        polyorder=polyorder,
        min_points_smooth=min_points_smooth,
    )

    start_temp = float(temperature[start_idx])
    end_temp = float(temperature[end_idx])

    # Heating starts low and ends high; cooling the other way round
    if is_cooling and start_temp < end_temp:
        start_temp, end_temp = end_temp, start_temp
    elif not is_cooling and start_temp > end_temp:
        start_temp, end_temp = end_temp, start_temp

    return start_temp, end_temp


def find_midpoint_temperature(
    temperature: NDArray[np.float64],
    transformed_fraction: NDArray[np.float64],
    start_temp: float,
    end_temp: float,
    is_cooling: bool = False,
) -> float:
    """Find temperature at 50% transformation (T50%). Interpolates if needed."""

    # Define mask for the transformation region based on start/end temps and direction
    if is_cooling:
        # For cooling, start_temp > end_temp
        mask = (temperature <= start_temp) & (temperature >= end_temp)
    else:
        # For heating, start_temp < end_temp
        mask = (temperature >= start_temp) & (temperature <= end_temp)

    valid_temp = temperature[mask]
    valid_fraction = transformed_fraction[mask]

    if len(valid_temp) < 2:
        # Not enough points in the transformation region for interpolation
        # Fallback: simple average (might be inaccurate)
        py_warnings.warn(
            "Less than 2 points found in the transformation region. Midpoint temperature is estimated as the average of start and end temperatures.",
            UserWarning,
        )
        return float((start_temp + end_temp) / 2.0)

    try:
        # Check if 0.5 is within the range of calculated fractions
        min_frac, max_frac = min(valid_fraction), max(valid_fraction)

        if min_frac <= 0.5 <= max_frac:
            # Interpolate temperature as a function of fraction
            # Ensure fraction is monotonically increasing for interpolation
            if is_cooling:
                # Fraction decreases from ~1 to ~0 as temp decreases. Interpolate T(fraction).
                # Need to sort by fraction descending if using interp1d directly.
                # Or, interpolate T(1-fraction) if fraction represents the low-T phase.
                # Let's assume transformed_fraction represents the forming phase (0->1).
                # For cooling, this means low-T phase. T decreases as fraction increases.
                sort_indices = np.argsort(valid_fraction)  # Sort by increasing fraction
                interp_func = np.interp
                mid_temp = interp_func(
                    0.5, valid_fraction[sort_indices], valid_temp[sort_indices]
                )

            else:
                # Heating: Fraction increases from ~0 to ~1 as temp increases. Interpolate T(fraction).
                sort_indices = np.argsort(valid_fraction)  # Sort by increasing fraction
                interp_func = np.interp
                mid_temp = interp_func(
                    0.5, valid_fraction[sort_indices], valid_temp[sort_indices]
                )

            return float(mid_temp)
        else:
            # 0.5 is outside the calculated fraction range within the identified T_start/T_end
            py_warnings.warn(
                f"Transformed fraction within the identified range [{min_frac:.3f}, {max_frac:.3f}] "
                f"does not encompass 0.5. Midpoint temperature is estimated as the average of start and end temperatures.",
                UserWarning,
            )
            return float((start_temp + end_temp) / 2.0)

    except Exception as e:
        # Fallback if interpolation fails for any reason
        py_warnings.warn(
            f"Interpolation for midpoint temperature failed: {e}. Using average of start and end temperatures.",
            UserWarning,
        )
        return float((start_temp + end_temp) / 2.0)


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


def calculate_transformed_fraction(
    strain: NDArray[np.float64],
    pred_start: NDArray[np.float64],  # Extrapolated start baseline
    pred_end: NDArray[np.float64],  # Extrapolated end baseline
    start_idx: int,  # Index where transformation starts
    end_idx: int,  # Index where transformation ends
    is_cooling: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate the transformed fraction using the lever rule between the
    extrapolated baselines within the identified transformation indices.

    Args:
        strain: Array of actual strain values.
        pred_start: Extrapolated baseline from the starting state.
        pred_end: Extrapolated baseline from the ending state.
        start_idx: Index marking the start of the transformation region.
        end_idx: Index marking the end of the transformation region.
        is_cooling: Boolean indicating direction.

    Returns:
        NDArray[np.float64]: Array of transformed fraction values (0 to 1).
    """
    transformed_fraction = np.zeros_like(strain)
    n_total = len(strain)

    # Ensure indices are valid and ordered
    start_idx = max(0, min(start_idx, n_total - 1))
    end_idx = max(0, min(end_idx, n_total - 1))
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx  # Ensure start <= end index

    # Define slice for the transformation region
    transformation_slice = slice(start_idx, end_idx + 1)

    # Calculate fraction within the transformation region
    height_total = pred_end[transformation_slice] - pred_start[transformation_slice]
    height_current = strain[transformation_slice] - pred_start[transformation_slice]

    # Avoid division by zero
    valid_total = np.abs(height_total) > 1e-9
    raw_fraction = np.zeros_like(height_current)
    np.divide(height_current, height_total, out=raw_fraction, where=valid_total)

    # Assign fraction values based on direction and region
    if is_cooling:
        # Cooling: Fraction (of low-T phase) goes 0 -> 1 as process proceeds (temp decreases)
        # Raw fraction calculated is fraction of the 'end' state (low-T phase)
        transformed_fraction[transformation_slice] = raw_fraction
        transformed_fraction[:start_idx] = 0.0  # Before start index (higher temp)
        transformed_fraction[end_idx + 1 :] = (
            1.0  # After end index (lower temp) - assuming full transformation
        )
    else:
        # Heating: Fraction (of high-T phase) goes 0 -> 1 as process proceeds (temp increases)
        # Raw fraction calculated is fraction of the 'end' state (high-T phase)
        transformed_fraction[transformation_slice] = raw_fraction
        transformed_fraction[:start_idx] = 0.0  # Before start index (lower temp)
        transformed_fraction[end_idx + 1 :] = (
            1.0  # After end index (higher temp) - assuming full transformation
        )

    # Clip to handle noise or extrapolation issues leading to values outside [0, 1]
    return np.clip(transformed_fraction, 0, 1)


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
) -> Dict[str, Union[float, List[str]]]:
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
        Dict containing R² values, margin, deviation fraction, and a list of warnings.
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

    return {
        "r2_start": float(r2_start),
        "r2_end": float(r2_end),
        "margin_used": float(margin_percent),
        "deviation_fraction": float(deviation_fraction),
        "warnings": warnings_list,  # Include list of warnings
    }


def calculate_r2(
    x: NDArray[np.float64], y: NDArray[np.float64], p: NDArray[np.float64]
) -> float:
    """Calculate R² (coefficient of determination) value for a polynomial fit."""
    if len(x) < 2:
        # Cannot calculate R² with less than 2 points
        return np.nan
    if np.all(y == y[0]):
        # If y is constant, R² is ill-defined or arguably 0 if fit is also constant, 1 if fit matches.
        # Let's return NaN as SS_tot would be zero.
        return np.nan

    y_pred = np.polyval(p, x)
    ss_res: float = float(np.sum((y - y_pred) ** 2))
    ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))

    if ss_tot < 1e-15:  # Avoid division by zero if y is effectively constant
        return 1.0 if ss_res < 1e-15 else 0.0  # Perfect fit if residuals are also zero

    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2)


def detect_noise_level(
    strain: NDArray[np.float64],
    window_size_fraction: float = 0.05,
    min_window: int = 10,
) -> float:
    """
    Estimate noise level in strain data using median of local standard deviations.

    Args:
        strain: Strain data array.
        window_size_fraction: Fraction of data length for window size.
        min_window: Minimum window size.

    Returns:
        Estimated noise level (median standard deviation).
    """
    n_total = len(strain)
    window_size = int(n_total * window_size_fraction)
    window_size = max(min_window, window_size)
    window_size = min(window_size, n_total // 2)  # Ensure window is not too large

    if window_size < 2:
        return float(np.std(strain) if n_total > 1 else 0.0)  # Explicit cast to float

    try:
        # Calculate standard deviation in sliding windows
        # Using pandas for efficient rolling calculation
        import pandas as pd

        rolling_std = (
            pd.Series(strain)
            .rolling(
                window=window_size, center=True, min_periods=max(2, window_size // 2)
            )
            .std()
        )
        # Use median of the calculated rolling standard deviations (ignoring NaNs at edges)
        median_std = np.nanmedian(rolling_std)
        return (
            float(median_std)
            if not np.isnan(median_std)
            else float(np.std(strain) if n_total > 1 else 0.0)
        )

    except ImportError:
        # Fallback if pandas is not available (less efficient)
        local_std = []
        step = max(1, window_size // 2)  # Use overlapping windows
        for i in range(0, n_total - window_size + 1, step):
            segment = strain[i : i + window_size]
            if len(segment) > 1:
                local_std.append(np.std(segment))
        return float(
            np.median(local_std)
            if local_std
            else (np.std(strain) if n_total > 1 else 0.0)
        )
