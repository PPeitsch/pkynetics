"""Where a transformation starts and ends, located on the derivative of
the strain with respect to temperature."""

import warnings as py_warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .curve_features import _strain_derivative
from .linear_segments import get_linear_segment_masks
from .types import TransformationLimits
from .utilities import _longest_run, _mad_scale, _smoothing_window, calculate_r2


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
) -> TransformationLimits:
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

    The derivative itself comes from a local polynomial fit
    (:func:`_strain_derivative`) rather than from differencing a smoothed
    signal, because the noise of a two-point difference grows as the sample
    spacing shrinks. On a run recorded every 0.25 K that noise, not the
    transformation, was setting the threshold.

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
    derivative = _strain_derivative(temperature, strain, window_length, polyorder)

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

    return TransformationLimits(int(start_idx), int(end_idx))


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
