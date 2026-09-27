"""The derivative detector: the transformation on ``dS/dT``.

This is what the module has done since #97, and it stays the default. It is
here rather than in ``transformation_points`` so that it sits next to the
alternatives it is now one of.
"""

import warnings as py_warnings
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_coeffs, savgol_filter

from ..curve_features import _strain_derivative
from ..linear_segments import get_linear_segment_masks
from ..types import TransformationLimits
from ..utilities import _dominant_run, _mad_scale, calculate_r2
from .base import DetectionContext

#: Most times the baselines are re-estimated around the limits they give.
#: Two or three passes settle it on real runs.
_MAX_PASSES = 10


def _derivative_noise(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    mask: NDArray[np.bool_],
    window_length: int,
    polyorder: int,
) -> float:
    """Scatter the measurement puts on :func:`_strain_derivative` within ``mask``.

    The spread of the derivative itself is not this. A baseline whose slope
    drifts or undulates spreads its derivative just as much as noise does, and
    on the Zry-4 heating run that drift, not the noise, set the detection
    threshold at 9-11 % of the peak excursion instead of 5 % (issue #115).

    Instead, the noise is measured where it lives, on the strain: its scatter
    around the same local fit the derivative comes from, which a slow drift
    does not reach. It is carried over to the derivative the way the fit
    carries it, through the norm of the differentiating filter's coefficients,
    and divided by the temperature step.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        mask: The points to measure the noise over.
        window_length: The window :func:`_strain_derivative` was given.
        polyorder: The polynomial order it was given.

    Returns:
        A standard deviation of ``dS/dT``, robust to isolated spikes.
    """
    if window_length:
        try:
            residual = strain - savgol_filter(strain, window_length, polyorder)
            dt_di = savgol_filter(temperature, window_length, polyorder, deriv=1)
        except ValueError:
            pass  # the derivative fell back to differencing: so does this
        else:
            gain = float(
                np.linalg.norm(
                    savgol_coeffs(window_length, polyorder, deriv=1, use="dot")
                )
            )
            step = float(np.median(np.abs(dt_di[mask])))
            if step > 0:
                return _mad_scale(residual[mask]) * gain / step
    derivative = _strain_derivative(temperature, strain, window_length, polyorder)
    return _mad_scale(derivative[mask])


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


def derivative_limits(
    context: DetectionContext,
    deviation_fraction: float = 0.05,
) -> TransformationLimits:
    """Locate the transformation on the derivative of the strain.

    The transformation is located on ``dS/dT`` rather than on the deviation of
    the strain from an extrapolated tangent. A real baseline is never exactly
    straight, and the *integral* of a slight curvature is indistinguishable
    from the beginning of a transformation, which is why deviation-based
    detection drifted 80-150 K early on real data. On the derivative the two
    separate: the curvature of the baseline is a small offset, the
    transformation a large, localised excursion.

    The limits are the ends of the stretch where the derivative stays further
    than ``deviation_fraction`` of the peak excursion from each baseline. Of
    the stretches that do, the one carrying the most deviation is taken, so
    neither a single noisy point nor a long, shallow departure -- the two
    baselines have different slopes -- can stand in for the transformation.

    Each baseline is what is left of its margin window once the transformation
    is taken out of it. A window that reaches into the transformation takes the
    slope and the scatter of the transformation for those of the baseline, and
    widens the threshold until the detected interval shrinks: margin 0.30 put
    the end of the Zry-4 heating run at 926 degC instead of 937 (issue #115).
    So the limits are found, the stretch they bracket -- plus half a smoothing
    window, the reach of the local fit -- is removed from both windows, and
    the limits are found again, until they stop moving.

    The threshold has to clear the noise of the measurement, not the spread of
    the derivative over the baseline: a baseline whose slope drifts spreads its
    derivative too, and counting that as noise raised the threshold to 9-11 %
    of the peak excursion on the same run (see :func:`_derivative_noise`).

    The derivative itself comes from a local polynomial fit
    (:func:`_strain_derivative`) rather than from differencing a smoothed
    signal, because the noise of a two-point difference grows as the sample
    spacing shrinks. On a run recorded every 0.25 K that noise, not the
    transformation, was setting the threshold.

    Args:
        context: The curve and the shared detection settings.
        deviation_fraction: Fraction of the peak excursion of the derivative
            that still counts as transforming. 0.05 reproduces both a
            synthetic sigmoid and a real Zry-4 dilatometry run.

    Returns:
        :class:`TransformationLimits`, in array order.

    Raises:
        ValueError: If the arguments or the amount of data are invalid.
    """
    temperature = context.temperature
    strain = context.strain
    is_cooling = context.is_cooling
    margin = context.margin
    polyorder = context.polyorder
    baseline_min_r2 = context.baseline_min_r2
    window_length = context.window_length

    # The amount of data is checked once, in `find_transformation_limits`,
    # since every detector needs the same minimum.
    n_total = len(temperature)
    if not (0.0 < deviation_fraction < 1.0):
        raise ValueError("deviation_fraction must be between 0 and 1 (exclusive)")

    derivative = _strain_derivative(temperature, strain, window_length, polyorder)

    start_window, end_window = get_linear_segment_masks(temperature, margin, is_cooling)
    if np.sum(start_window) < 2 or np.sum(end_window) < 2:
        raise ValueError(
            f"Margin {margin:.1%} leaves fewer than 2 points in a baseline segment."
        )

    def locate(
        start_mask: NDArray[np.bool_], end_mask: NDArray[np.bool_]
    ) -> Tuple[int, int]:
        """The limits, measured against the baselines in the two masks."""
        limits = []
        for mask, side in ((start_mask, 0), (end_mask, 1)):
            # Deviation of the derivative from the baseline's own slope
            deviation = np.abs(derivative - float(np.median(derivative[mask])))
            # A limit has to clear both the noise of its own baseline and a
            # fixed fraction of the excursion, so neither a noisy nor a clean
            # curve degenerates. The excursion is a high percentile rather
            # than the maximum: a single spike in the raw data survives
            # smoothing as a large spike in the derivative, and would
            # otherwise set the scale for everything else.
            threshold = max(
                deviation_fraction * float(np.percentile(deviation, 99.5)),
                3.0
                * _derivative_noise(
                    temperature, strain, mask, window_length, polyorder
                ),
            )
            # The transformation is the run of points over the threshold that
            # carries the most deviation. Not the first one, and not the one
            # holding the highest point: an isolated spike is a run of one or
            # two points. Not the longest one either: the two baselines have
            # different slopes, so against the final baseline the whole
            # stretch before the transformation deviates by that difference,
            # and once the threshold dips below it that stretch can outlast
            # the transformation (issue #115: 820-839 degC instead of 839-938
            # on the Zry-4 heating run).
            limits.append(_dominant_run(deviation > threshold, deviation)[side])
        return limits[0], limits[1]

    # Take the transformation out of the baseline windows and look again,
    # until the limits stop moving. A window left with less than a smoothing
    # window of baseline cannot be trusted either, so the last limits that
    # had enough of it stand.
    index = np.arange(n_total)
    reach = window_length // 2
    min_baseline = max(window_length, 3)
    start_mask, end_mask = start_window, end_window
    start_idx, end_idx = locate(start_mask, end_mask)
    seen = {(start_idx, end_idx)}
    for _ in range(_MAX_PASSES):
        low, high = min(start_idx, end_idx), max(start_idx, end_idx)
        outside = (index < low - reach) | (index > high + reach)
        trimmed_start, trimmed_end = start_window & outside, end_window & outside
        if np.array_equal(trimmed_start, start_mask) and np.array_equal(
            trimmed_end, end_mask
        ):
            break
        if min(np.sum(trimmed_start), np.sum(trimmed_end)) < min_baseline:
            py_warnings.warn(
                f"Once the transformation is taken out, a baseline window has "
                f"fewer than {min_baseline} points left: the run has too little "
                f"baseline on one side. Widen the analysis range, or raise "
                f"`margin` (currently {margin:.1%}). The limits below may be "
                f"unreliable.",
                UserWarning,
            )
            break
        start_mask, end_mask = trimmed_start, trimmed_end
        start_idx, end_idx = locate(start_mask, end_mask)
        if (start_idx, end_idx) in seen:
            break  # settled, or cycling between answers it has already given
        seen.add((start_idx, end_idx))

    # A baseline that is not straight even with the transformation taken out
    # has reached into a second one, and every threshold above is then
    # measured against the wrong thing
    for mask, side_name in ((start_mask, "initial"), (end_mask, "final")):
        r2 = calculate_r2(
            temperature[mask],
            strain[mask],
            np.polyfit(temperature[mask], strain[mask], 1),
        )
        if r2 < baseline_min_r2:
            py_warnings.warn(
                f"The {side_name} baseline window is not linear (R² = {r2:.3f}). It "
                f"probably reaches into a transformation: narrow the analysis "
                f"range or lower `margin` (currently {margin:.1%}). The limits "
                f"below may be unreliable.",
                UserWarning,
            )

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
