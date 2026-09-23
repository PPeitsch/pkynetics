"""The lever method."""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..detection import DEFAULT_DETECTION
from ..linear_segments import (
    extrapolate_linear_segments,
    fit_linear_segments,
    get_linear_segment_masks,
)
from ..transformation_points import find_inflection_points, find_midpoint_temperature
from ..transformed_fraction import (
    calculate_transformed_fraction,
    calculate_transformed_fraction_lever,
    max_backward_step,
)
from ..types import ReturnDict


def lever_method(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    is_cooling: bool,
    margin_percent_fraction: float = 0.2,
    find_inflection_margin: float = 0.2,
    min_points_fit: int = 5,  # Min points for fraction calculation fit
    detection: str = DEFAULT_DETECTION,
    **detection_options: Any,
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
        detection: Which detector locates the transformation limits.
        **detection_options: Passed through to the chosen detector.

    Returns:
        Dictionary containing analysis results.
    """
    # 1. Find transformation start and end points using the specified inflection margin
    start_temp, end_temp = find_inflection_points(
        temperature,
        strain,
        is_cooling,
        margin=find_inflection_margin,
        detection=detection,
        **detection_options,
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
        # The fraction is a raw reading of the curve; this says how far noise
        # pushes it backwards. See `max_backward_step`.
        "max_backward_step": max_backward_step(transformed_fraction),
        "temperature": temperature,  # Include temperature for context
        "strain": strain,  # Include strain for context
        "before_extrapolation": before_extrap,
        "after_extrapolation": after_extrap,
        "is_cooling": is_cooling,
        "parameters": {  # Store parameters used
            "margin_percent_fraction": margin_percent_fraction,
            "find_inflection_margin": find_inflection_margin,
            "min_points_fit": min_points_fit,
            "detection": detection,
        },
    }
