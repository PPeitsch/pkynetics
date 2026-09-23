"""Where a transformation starts and ends, located on the derivative of
the strain with respect to temperature."""

import warnings as py_warnings
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .curve_features import _strain_derivative
from .detection import DEFAULT_DETECTION, DetectionContext, get_detector
from .detection.adaptive import stable_margin
from .linear_segments import get_linear_segment_masks
from .types import TransformationLimits
from .utilities import _longest_run, _mad_scale, _smoothing_window, calculate_r2


def find_transformation_limits(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    is_cooling: bool = False,
    margin: Union[float, str] = 0.2,
    deviation_fraction: float = 0.05,
    smooth_window_fraction: float = 0.05,
    polyorder: int = 2,
    min_points_smooth: int = 5,
    baseline_min_r2: float = 0.99,
    detection: str = DEFAULT_DETECTION,
    **detection_options: Any,
) -> TransformationLimits:
    """
    Find the indices where the transformation starts and ends.

    Which rule is used to find them is ``detection``; the default,
    ``"derivative"``, locates them on ``dS/dT`` and is what this function has
    always done. See
    :mod:`~pkynetics.technique_analysis.dilatometry.detection` for the
    alternatives and :func:`available_detectors` for their names.

    ``detection`` answers a different question from the ``method`` argument of
    :func:`analyze_dilatometry_curve`: this one is *where* the transformation
    is, that one is *how far it has gone*. They are chosen independently.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        is_cooling: Whether this is a cooling segment.
        margin: Fraction of the data at each end taken as baseline (0.1-0.4),
            or ``"auto"`` to choose the one whose answer holds over the widest
            range of margins. The margin has narrow dead zones -- on the Zry-4
            heating run 0.15 to 0.17 collapse the bracket to 15 K where 0.18
            to 0.25 give 97 -- and nothing about a curve says where they are,
            so ``"auto"`` is worth its ~25 runs of the detector when the data
            are unfamiliar. See
            :mod:`~pkynetics.technique_analysis.dilatometry.detection.adaptive`.
        deviation_fraction: Fraction of the peak excursion that still counts
            as transforming. An option of the ``"derivative"`` detector, named
            here because it was a parameter of this function before there were
            others.
        smooth_window_fraction: Savitzky-Golay window as a fraction of length.
        polyorder: Polynomial order for smoothing.
        min_points_smooth: Minimum smoothing window length.
        baseline_min_r2: R² below which a baseline window is reported as not
            linear, i.e. as reaching into a transformation.
        detection: Which detector to use. Defaults to ``"derivative"``.
        **detection_options: Passed through to the chosen detector.

    Returns:
        :class:`TransformationLimits`, a named tuple of indices into the input
        arrays, in array order (``start_idx < end_idx``) whatever the ramp
        direction. It unpacks as ``start_idx, end_idx``.

    Raises:
        ValueError: If the arguments or the amount of data are invalid, or if
            no detector is registered under ``detection``.
    """
    n_total = len(temperature)
    if n_total < 20:
        raise ValueError(
            f"Insufficient data points ({n_total}) to locate the transformation."
        )

    detector = get_detector(detection)
    if detection.lower() == DEFAULT_DETECTION:
        detection_options.setdefault("deviation_fraction", deviation_fraction)

    window_length = _smoothing_window(
        n_total, smooth_window_fraction, min_points_smooth
    )

    def run(chosen_margin: float) -> TransformationLimits:
        context = DetectionContext(
            temperature=temperature,
            strain=strain,
            is_cooling=is_cooling,
            margin=chosen_margin,
            window_length=window_length,
            polyorder=polyorder,
            baseline_min_r2=baseline_min_r2,
        )
        return detector(context, **detection_options)

    if isinstance(margin, str):
        if margin.lower() != "auto":
            raise ValueError(f"margin must be a fraction or 'auto', not '{margin}'.")
        chosen, _ = stable_margin(run, n_total)
        # Re-run at the chosen margin rather than reuse the explored result,
        # so the warnings that apply to it are raised where the caller sees
        # them; the ones from margins that were tried and dropped are not.
        return run(chosen)

    return run(float(margin))


def find_inflection_points(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    is_cooling: bool = False,
    margin: float = 0.2,
    smooth_window_fraction: float = 0.05,
    polyorder: int = 2,
    min_points_smooth: int = 5,
    deviation_fraction: float = 0.05,
    detection: str = DEFAULT_DETECTION,
    **detection_options: Any,
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
        detection: Which detector to use. Defaults to ``"derivative"``.
        **detection_options: Passed through to the chosen detector.

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
        detection=detection,
        **detection_options,
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
