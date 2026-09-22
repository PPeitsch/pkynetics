"""The tangent method."""

import warnings as py_warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..linear_segments import (
    calculate_fit_quality,
    extrapolate_linear_segments,
    find_optimal_margin,
    fit_linear_segments,
    get_extrapolated_values,
    get_linear_segment_masks,
)
from ..transformation_points import (
    find_inflection_points,
    find_midpoint_temperature,
    find_transformation_limits,
)
from ..transformed_fraction import calculate_transformed_fraction
from ..types import ReturnDict


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
