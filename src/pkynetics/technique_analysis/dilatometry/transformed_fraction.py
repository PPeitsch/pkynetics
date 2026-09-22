"""How far the transformation has gone at each temperature."""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pkynetics.technique_analysis.utilities import detect_segment_direction

from .linear_segments import (
    extrapolate_linear_segments,
    fit_linear_segments,
    get_extrapolated_values,
    get_linear_segment_masks,
)
from .transformation_points import find_inflection_points


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


def max_backward_step(fraction: NDArray[np.float64]) -> float:
    """Largest step the transformed fraction takes backwards.

    The fraction is a direct reading of the curve, so noise in the strain
    propagates into it and it is not guaranteed to be monotonic: on the shipped
    Zry-4 cooling run the worst backward step is 0.28 % of full scale. That is
    measurement noise, not a sign error, and it is reported rather than
    smoothed away -- constraining the fraction to rise would make it something
    other than a reading of the curve.

    Args:
        fraction: Transformed fraction, in the interval [0, 1].

    Returns:
        The largest decrease between consecutive points, as a fraction of full
        scale, or 0.0 if the fraction never goes backwards.
    """
    if fraction.size < 2:
        return 0.0
    steps = np.diff(np.asarray(fraction, dtype=np.float64))
    backward = steps[steps < 0]
    return float(-backward.min()) if backward.size else 0.0
