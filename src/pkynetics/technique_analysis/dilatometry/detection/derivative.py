"""The derivative detector: the transformation on ``dS/dT``.

This is what the module has done since #97, and it stays the default. It is
here rather than in ``transformation_points`` so that it sits next to the
alternatives it is now one of.
"""

import warnings as py_warnings

import numpy as np

from ..curve_features import _strain_derivative
from ..linear_segments import get_linear_segment_masks
from ..types import TransformationLimits
from ..utilities import _longest_run, _mad_scale, calculate_r2
from .base import DetectionContext


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
