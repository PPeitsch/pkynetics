import warnings as py_warnings  # Use standard warnings library
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pkynetics.data_preprocessing.common_preprocessing import smooth_data
from pkynetics.technique_analysis.utilities import detect_segment_direction

# Type hint for the dictionary returned by analysis functions
ReturnDict = Dict[
    str, Union[float, bool, NDArray[np.float64], Dict[str, Union[float, List[str]]]]
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
            n_start = np.sum(start_mask)
            n_end = np.sum(end_mask)

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
        # If multiple margins meet the criteria, choose the one with the highest R²
        best_candidate = max(candidate_margins, key=lambda x: x["avg_r2"])
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
    return np.clip(transformed_fraction, 0, 1), before_extrap, after_extrap


def analyze_dilatometry_curve(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    method: str = "lever",
    margin_percent: Optional[float] = None,
    find_inflection_margin: float = 0.3,
    min_points_fit: int = 10,
    min_r2_optimal_margin: float = 0.99,
    deviation_threshold: Optional[float] = None,
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
                                'lever' method's `find_inflection_points` function. Default is 0.3.
        min_points_fit: Minimum number of points required for reliable linear fitting
                        in tangent/lever methods. Default is 10.
        min_r2_optimal_margin: Minimum R² required when using `find_optimal_margin`
                               in the tangent method. Default is 0.99.
        deviation_threshold: Deviation threshold for the tangent method. If None, it's calculated.

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
            deviation_threshold=deviation_threshold,
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
    find_inflection_margin: float = 0.3,
    min_points_fit: int = 5,  # Min points for fraction calculation fit
) -> ReturnDict:
    """
    Analyze dilatometry curve using the lever rule method.
    Finds transformation points based on deviation from tangents fitted using 'find_inflection_margin'.
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
    deviation_threshold: Optional[float] = None,
    min_points_fit: int = 10,
    min_r2_optimal_margin: float = 0.99,
) -> ReturnDict:
    """
    Analyze dilatometry curve using the tangent intersection method.
    Fits tangents based on 'margin_percent' (or finds optimal).
    Finds transformation points based on deviation from these tangents.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        is_cooling: Boolean indicating direction.
        margin_percent: Margin for fitting tangents. If None, finds optimal margin.
        deviation_threshold: Threshold for detecting deviation. If None, calculated automatically.
        min_points_fit: Minimum points for tangent fitting.
        min_r2_optimal_margin: Minimum R² for optimal margin search.

    Returns:
        Dictionary containing analysis results including fit quality.
    """
    warnings_list = []  # Collect warnings during execution

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

    # 4. Determine deviation threshold if not provided
    final_deviation_threshold: float
    if deviation_threshold is not None:
        final_deviation_threshold = float(deviation_threshold)
    else:
        final_deviation_threshold = calculate_deviation_threshold(
            strain, pred_start, pred_end, start_mask, end_mask
        )
        if (
            final_deviation_threshold < 1e-9
        ):  # Handle case with almost perfect fit / no noise
            noise_level = detect_noise_level(strain)  # Estimate noise from data
            final_deviation_threshold = max(
                noise_level * 3, 1e-7
            )  # Use noise estimate or a small floor value
            warnings_list.append(
                f"Calculated deviation threshold was near zero. Reset to {final_deviation_threshold:.2e} based on noise estimate."
            )

    # 5. Find transformation start/end points based on deviation
    start_idx, end_idx = find_transformation_points(
        temperature, strain, pred_start, pred_end, final_deviation_threshold, is_cooling
    )

    # Ensure indices are ordered correctly for slicing etc. (start < end index)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

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

    # 6. Calculate transformed fraction using the identified points and tangents
    transformed_fraction = calculate_transformed_fraction(
        strain, pred_start, pred_end, start_idx, end_idx, is_cooling
    )

    # 7. Find midpoint temperature (T50%)
    mid_temp = find_midpoint_temperature(
        temperature,
        transformed_fraction,
        report_start_temp,  # Use direction-aware temps
        report_end_temp,
        is_cooling,
    )

    # 8. Calculate fit quality metrics
    fit_quality = calculate_fit_quality(
        temperature,
        strain,
        p_start,
        p_end,
        start_mask,
        end_mask,
        final_margin_percent,
        final_deviation_threshold,
        warnings_list,  # Pass current warnings
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
            "deviation_threshold": final_deviation_threshold,
            "min_points_fit": min_points_fit,
            "min_r2_optimal_margin": min_r2_optimal_margin,
        },
    }


# --- Helper Functions ---


def find_inflection_points(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    is_cooling: bool = False,
    margin: float = 0.3,
    smooth_window_fraction: float = 0.05,  # Smoothing window as fraction of data length
    polyorder: int = 2,
    min_points_smooth: int = 5,
    min_points_fit: int = 5,  # Min points for the tangent fits in this function
    residual_std_multiplier: float = 3.0,  # Multiplier for noise threshold
) -> Tuple[float, float]:
    """
    Find transformation start/end points where the curve deviates significantly
    from linear tangents fitted to the initial and final segments defined by 'margin'.
    Uses smoothed data and adaptive noise thresholds.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        is_cooling: Whether this is a cooling segment.
        margin: Fraction of temperature range (0.1-0.4) to define linear regions for fitting tangents.
        smooth_window_fraction: Fraction of data length for Savitzky-Golay smoothing window.
        polyorder: Polynomial order for smoothing.
        min_points_smooth: Minimum window length for smoothing.
        min_points_fit: Minimum points required for fitting the tangents.
        residual_std_multiplier: Multiplier for standard deviation of residuals to set noise threshold.

    Returns:
        Tuple containing start and end temperatures of the transformation, ordered
        according to heating/cooling convention (start > end for cooling).

    Raises:
        ValueError: If margin is invalid, data is insufficient, or fitting fails.
    """
    # Validate margin value
    if not (0.1 <= margin <= 0.4):
        raise ValueError("Margin must be between 0.1 and 0.4")

    n_total = len(temperature)
    if n_total < max(
        min_points_fit * 4, min_points_smooth, 20
    ):  # Ensure enough points for smoothing, fitting, and detection
        raise ValueError(
            f"Insufficient data points ({n_total}). Need more points for reliable inflection point detection."
        )

    # Smooth data for more robust analysis
    window_length = int(n_total * smooth_window_fraction)
    window_length = max(min_points_smooth, window_length)
    if window_length % 2 == 0:
        window_length += 1  # Ensure odd
    window_length = min(
        window_length, n_total - 2
    )  # Ensure window is smaller than data length

    try:
        smooth_strain = smooth_data(
            strain, window_length=window_length, polyorder=polyorder
        )
    except ValueError as e:
        raise ValueError(f"Smoothing failed: {e}")

    # Define linear regions based on margin and heating/cooling direction
    start_fit_mask, end_fit_mask = get_linear_segment_masks(
        temperature, margin, is_cooling
    )

    # Ensure enough points in each region for fitting
    if np.sum(start_fit_mask) < min_points_fit:
        raise ValueError(
            f"Insufficient points ({np.sum(start_fit_mask)}) in initial segment (margin={margin:.1%}) "
            f"for fitting tangent. Need at least {min_points_fit}."
        )
    if np.sum(end_fit_mask) < min_points_fit:
        raise ValueError(
            f"Insufficient points ({np.sum(end_fit_mask)}) in final segment (margin={margin:.1%}) "
            f"for fitting tangent. Need at least {min_points_fit}."
        )

    # Fit tangent lines to linear regions using SMOOTHED data
    try:
        start_fit_coeffs = np.polyfit(
            temperature[start_fit_mask], smooth_strain[start_fit_mask], 1
        )
        end_fit_coeffs = np.polyfit(
            temperature[end_fit_mask], smooth_strain[end_fit_mask], 1
        )
    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f"Failed to fit tangents for inflection point detection: {e}")

    # Calculate extrapolations across the full range
    start_line = np.polyval(start_fit_coeffs, temperature)
    end_line = np.polyval(end_fit_coeffs, temperature)

    # Calculate residuals between SMOOTHED curve and extrapolations
    start_residuals = np.abs(smooth_strain - start_line)
    end_residuals = np.abs(smooth_strain - end_line)

    # Set adaptive noise thresholds based on standard deviation of residuals in the linear fit regions
    noise_start = np.std(start_residuals[start_fit_mask]) * residual_std_multiplier
    noise_end = np.std(end_residuals[end_fit_mask]) * residual_std_multiplier
    # Add a small floor value to prevent zero threshold if fit is perfect
    noise_start = max(noise_start, 1e-9)
    noise_end = max(noise_end, 1e-9)

    # Find indices where residuals exceed the noise threshold
    start_deviation_indices = np.where(start_residuals > noise_start)[0]
    end_deviation_indices = np.where(end_residuals > noise_end)[0]

    # Determine the first and last points of significant deviation
    # Consider the direction:
    # - Heating: Start deviation is the first index deviating from start_line.
    #            End deviation is the last index deviating from end_line.
    # - Cooling: Start deviation is the last index deviating from start_line (high temp).
    #            End deviation is the first index deviating from end_line (low temp).

    start_idx_transform: Optional[int] = None
    end_idx_transform: Optional[int] = None

    if is_cooling:
        # Start (high temp): Last index deviating significantly from the high-temp baseline
        if len(start_deviation_indices) > 0:
            start_idx_transform = start_deviation_indices[
                -1
            ]  # Last point where high-T fit is still good? No, first deviation point.
            # Search from beginning (high temp) for first deviation from start_line
            for idx in np.argsort(-temperature):  # Iterate high to low temp
                if start_residuals[idx] > noise_start:
                    start_idx_transform = idx
                    break

        # End (low temp): First index deviating significantly from the low-temp baseline
        if len(end_deviation_indices) > 0:
            # Search from end (low temp) for first deviation from end_line
            for idx in np.argsort(temperature):  # Iterate low to high temp
                if end_residuals[idx] > noise_end:
                    end_idx_transform = idx
                    break

    else:  # Heating
        # Start (low temp): First index deviating significantly from the low-temp baseline
        if len(start_deviation_indices) > 0:
            # Search from beginning (low temp) for first deviation from start_line
            for idx in np.argsort(temperature):  # Iterate low to high temp
                if start_residuals[idx] > noise_start:
                    start_idx_transform = idx
                    break

        # End (high temp): Last index deviating significantly from the high-temp baseline
        if len(end_deviation_indices) > 0:
            # Search from end (high temp) for first deviation from end_line
            for idx in np.argsort(-temperature):  # Iterate high to low temp
                if end_residuals[idx] > noise_end:
                    end_idx_transform = idx
                    break

    # Fallback if points cannot be detected reliably using residuals
    # Use a position-based fallback within the middle ~50% of data
    fallback_start_idx = int(n_total * 0.25)
    fallback_end_idx = int(n_total * 0.75)
    if is_cooling:
        # Sort descending for cooling
        sorted_indices = np.argsort(-temperature)
        fallback_start_idx = sorted_indices[fallback_start_idx]
        fallback_end_idx = sorted_indices[fallback_end_idx]
    else:
        # Sort ascending for heating
        sorted_indices = np.argsort(temperature)
        fallback_start_idx = sorted_indices[fallback_start_idx]
        fallback_end_idx = sorted_indices[fallback_end_idx]

    if start_idx_transform is None:
        py_warnings.warn(
            f"Could not reliably detect transformation start point using margin {margin:.1%} and threshold multiplier {residual_std_multiplier}. "
            f"Using fallback index {fallback_start_idx}.",
            UserWarning,
        )
        start_idx_transform = fallback_start_idx

    if end_idx_transform is None:
        py_warnings.warn(
            f"Could not reliably detect transformation end point using margin {margin:.1%} and threshold multiplier {residual_std_multiplier}. "
            f"Using fallback index {fallback_end_idx}.",
            UserWarning,
        )
        end_idx_transform = fallback_end_idx

    # Convert indices to temperatures
    start_temp = float(temperature[start_idx_transform])
    end_temp = float(temperature[end_idx_transform])

    # Ensure correct order based on direction for return value
    # Convention: start_temp is where transformation begins, end_temp where it ends.
    if is_cooling:
        # Cooling starts at higher temp, ends at lower temp
        if start_temp < end_temp:
            start_temp, end_temp = end_temp, start_temp  # Swap if needed
    else:
        # Heating starts at lower temp, ends at higher temp
        if start_temp > end_temp:
            start_temp, end_temp = end_temp, start_temp  # Swap if needed

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
    n_start = np.sum(start_mask)
    n_end = np.sum(end_mask)

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
    pred_start = np.polyval(p_start, temperature)
    pred_end = np.polyval(p_end, temperature)
    return pred_start, pred_end


def find_transformation_points(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    pred_start: NDArray[np.float64],  # Extrapolated start baseline
    pred_end: NDArray[np.float64],  # Extrapolated end baseline
    deviation_threshold: float,
    is_cooling: bool = False,
    smooth_deviation: bool = True,  # Option to smooth deviation signal
    smooth_window_fraction: float = 0.05,
    polyorder: int = 2,
    min_points_smooth: int = 5,
) -> Tuple[int, int]:
    """
    Find transformation start and end point indices based on where the actual
    strain deviates significantly from the extrapolated linear baselines.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        pred_start: Extrapolated baseline from the starting linear segment.
        pred_end: Extrapolated baseline from the ending linear segment.
        deviation_threshold: Threshold value for significant deviation.
        is_cooling: Boolean indicating direction.
        smooth_deviation: Whether to smooth the deviation signal before thresholding.
        smooth_window_fraction: Fraction for smoothing window.
        polyorder: Order for smoothing polynomial.
        min_points_smooth: Minimum points for smoothing window.

    Returns:
        Tuple containing the indices (start_idx, end_idx) of the transformation.
        Indices refer to the original temperature/strain arrays.
        The order (start_idx vs end_idx) reflects the position in the array, not necessarily the process start/end.
    """
    n_total = len(temperature)

    # Calculate absolute deviations from both baselines
    dev_start = np.abs(strain - pred_start)
    dev_end = np.abs(strain - pred_end)

    # Optionally smooth the deviation signals
    if smooth_deviation:
        window_length = int(n_total * smooth_window_fraction)
        window_length = max(min_points_smooth, window_length)
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, n_total - 2)
        try:
            if (
                window_length >= 3
            ):  # Savgol filter requires window >= polyorder + 1 (and odd)
                dev_start = smooth_data(
                    dev_start, window_length=window_length, polyorder=polyorder
                )
                dev_end = smooth_data(
                    dev_end, window_length=window_length, polyorder=polyorder
                )
            # else: smoothing window too small, use raw deviations
        except ValueError:
            py_warnings.warn(
                "Smoothing deviations failed, using raw deviation signals.", UserWarning
            )
            # Continue with raw dev_start, dev_end

    # Find indices where deviation exceeds the threshold
    start_deviation_indices = np.where(dev_start > deviation_threshold)[0]
    end_deviation_indices = np.where(dev_end > deviation_threshold)[0]

    # Determine the start and end indices of the transformation region
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

    # Find the first point deviating from the initial baseline
    if len(start_deviation_indices) > 0:
        start_idx = start_deviation_indices[0]

    # Find the last point deviating from the final baseline
    if len(end_deviation_indices) > 0:
        end_idx = end_deviation_indices[-1]

    # --- Fallback logic if indices are not found or seem unreasonable ---
    # Use position-based fallback within the middle ~50% of data if needed
    fallback_start_idx = int(n_total * 0.25)
    fallback_end_idx = int(n_total * 0.75)

    if start_idx is None:
        py_warnings.warn(
            f"Could not detect transformation start index using deviation threshold {deviation_threshold:.2e}. "
            f"Using fallback index {fallback_start_idx}.",
            UserWarning,
        )
        start_idx = fallback_start_idx

    if end_idx is None:
        py_warnings.warn(
            f"Could not detect transformation end index using deviation threshold {deviation_threshold:.2e}. "
            f"Using fallback index {fallback_end_idx}.",
            UserWarning,
        )
        end_idx = fallback_end_idx

    # Ensure indices are ordered (start_idx <= end_idx)
    if start_idx > end_idx:
        # This might happen if deviation detection is noisy or thresholds are wrong
        py_warnings.warn(
            f"Detected start index ({start_idx}) is after end index ({end_idx}). "
            f"Swapping them. Check deviation threshold and data quality.",
            UserWarning,
        )
        start_idx, end_idx = end_idx, start_idx

    # The returned indices define the segment [start_idx, end_idx] in the array.
    # The physical start/end temps depend on the 'is_cooling' flag.
    return start_idx, end_idx


def calculate_deviation_threshold(
    strain: NDArray[np.float64],
    pred_start: NDArray[np.float64],
    pred_end: NDArray[np.float64],
    start_mask: NDArray[np.bool_],  # Mask used for fitting start line
    end_mask: NDArray[np.bool_],  # Mask used for fitting end line
    std_multiplier: float = 3.0,
) -> float:
    """
    Calculate an adaptive threshold for deviation detection based on the standard
    deviation of residuals in the linear fitting regions.

    Args:
        strain: Array of actual strain values.
        pred_start: Extrapolated values from the start linear fit.
        pred_end: Extrapolated values from the end linear fit.
        start_mask: Boolean mask indicating the data points used for the start fit.
        end_mask: Boolean mask indicating the data points used for the end fit.
        std_multiplier: Factor to multiply the standard deviation by.

    Returns:
        float: Calculated deviation threshold.
    """
    # Calculate residuals only in the regions used for fitting
    start_residuals = np.abs(strain[start_mask] - pred_start[start_mask])
    end_residuals = np.abs(strain[end_mask] - pred_end[end_mask])

    # Use the maximum of the standard deviations from the two regions
    std_dev_start = np.std(start_residuals) if len(start_residuals) > 1 else 0.0
    std_dev_end = np.std(end_residuals) if len(end_residuals) > 1 else 0.0

    threshold = float(std_multiplier * max(std_dev_start, std_dev_end))

    # Ensure threshold is not zero or extremely small
    return max(threshold, 1e-9)


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
    deviation_threshold: float,
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
        deviation_threshold: The deviation threshold used.
        existing_warnings: List of warnings generated earlier in the process.
        r2_warn_threshold: R² value below which a warning is generated.

    Returns:
        Dict containing R² values, margin, threshold, and a list of warnings.
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
        "deviation_threshold": float(deviation_threshold),
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
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot < 1e-15:  # Avoid division by zero if y is effectively constant
        return 1.0 if ss_res < 1e-15 else 0.0  # Perfect fit if residuals are also zero

    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2)


# Added function from dilatometry_preprocessing.py for self-containment if needed by tangent method
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
        return (
            np.std(strain) if n_total > 1 else 0.0
        )  # Use global std if window is too small

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
            else (np.std(strain) if n_total > 1 else 0.0)
        )

    except ImportError:
        # Fallback if pandas is not available (less efficient)
        local_std = []
        step = max(1, window_size // 2)  # Use overlapping windows
        for i in range(0, n_total - window_size + 1, step):
            segment = strain[i : i + window_size]
            if len(segment) > 1:
                local_std.append(np.std(segment))
        return (
            float(np.median(local_std))
            if local_std
            else (np.std(strain) if n_total > 1 else 0.0)
        )
