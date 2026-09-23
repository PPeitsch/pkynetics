"""The double-tangent detector: extrapolated onset and offset.

Three lines rather than two. The baseline on either side is fitted and
extrapolated, a third line is fitted to the steepest part of the
transformation, and the limits are where that third line crosses the other
two. It is the extrapolated onset of the thermal-analysis literature, and the
construction ASTM E228 describes for reading a transformation off a
dilatometry curve.
"""

import warnings as py_warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..curve_features import _strain_derivative
from ..linear_segments import get_linear_segment_masks
from ..types import TransformationLimits
from .base import DetectionContext


def _intersection(
    line_a: NDArray[np.float64], line_b: NDArray[np.float64]
) -> Optional[float]:
    """Temperature where two fitted lines cross, or None if they do not.

    Args:
        line_a: Coefficients of the first line, as :func:`numpy.polyfit` returns.
        line_b: Coefficients of the second line.

    Returns:
        The temperature of the intersection, or ``None`` when the lines are
        parallel to within floating-point resolution.
    """
    slope_difference = float(line_a[0] - line_b[0])
    scale = max(abs(float(line_a[0])), abs(float(line_b[0])))
    if abs(slope_difference) <= 1e-12 * max(scale, 1.0):
        return None
    return float((line_b[1] - line_a[1]) / slope_difference)


def double_tangent_limits(
    context: DetectionContext,
    tangent_window_fraction: float = 0.1,
) -> TransformationLimits:
    """Locate the transformation where its tangent meets each baseline.

    The construction is the one drawn by hand on a chart: extend the straight
    expansion before the transformation, extend the straight expansion after
    it, draw the tangent through the steepest part of the transformation, and
    read off the two crossings.

    Where it sits among the others: unlike ``offset`` it needs no threshold at
    all, so nothing has to be tuned per run and there is no value to get
    wrong. What it assumes instead is that the transformation *has* a single
    steepest part that a straight line describes -- on a transformation that
    proceeds in two stages, the tangent is fitted across both and the limits
    it reports belong to neither.

    Being built from extrapolations, it also reports limits that need not be
    points on the curve at all: the crossings are temperatures, converted here
    to the nearest sample. A crossing that lands outside the measured range is
    reported as such rather than clamped silently.

    Args:
        context: The curve and the shared detection settings.
        tangent_window_fraction: Width of the window fitted through the
            steepest part of the transformation, as a fraction of the run.
            Wide enough to average out noise, narrow enough to stay on the
            straight part of the transformation.

    Returns:
        :class:`TransformationLimits`, in array order.

    Raises:
        ValueError: If ``tangent_window_fraction`` is not between 0 and 1, or
            a baseline window is too short to fit.
    """
    if not (0.0 < tangent_window_fraction < 1.0):
        raise ValueError("tangent_window_fraction must be between 0 and 1 (exclusive)")

    temperature = context.temperature
    strain = context.strain
    n_total = len(temperature)

    start_mask, end_mask = get_linear_segment_masks(
        temperature, context.margin, context.is_cooling
    )
    if np.sum(start_mask) < 2 or np.sum(end_mask) < 2:
        raise ValueError(
            f"Margin {context.margin:.1%} leaves fewer than 2 points in a "
            f"baseline segment."
        )

    baseline_start = np.polyfit(temperature[start_mask], strain[start_mask], 1)
    baseline_end = np.polyfit(temperature[end_mask], strain[end_mask], 1)

    # The steepest part, found on the derivative rather than on the strain:
    # what makes a point "steep" is its slope, and reading it off the
    # derivative is the same measurement the other detectors already trust.
    derivative = _strain_derivative(
        temperature, strain, context.window_length, context.polyorder
    )
    baseline_slope = float(np.median(derivative[start_mask | end_mask]))
    steepest = int(np.argmax(np.abs(derivative - baseline_slope)))

    # The window cannot come out too narrow to fit a line through: the floor
    # of 2 leaves at least three points however small the fraction is, and
    # `find_transformation_limits` has already refused runs under 20 points.
    half_window = max(2, int(n_total * tangent_window_fraction / 2))
    lo = max(0, steepest - half_window)
    hi = min(n_total, steepest + half_window + 1)
    tangent = np.polyfit(temperature[lo:hi], strain[lo:hi], 1)

    crossings: Tuple[Optional[float], Optional[float]] = (
        _intersection(tangent, baseline_start),
        _intersection(tangent, baseline_end),
    )
    if any(crossing is None for crossing in crossings):
        py_warnings.warn(
            "The tangent through the transformation is parallel to one of "
            "the baselines, so they never cross. There may be no "
            "transformation in this range. Falling back to the search "
            "interval.",
            UserWarning,
        )
        return TransformationLimits(int(n_total * 0.15), int(n_total * 0.85))

    lowest, highest = float(np.min(temperature)), float(np.max(temperature))
    for crossing, side in zip(crossings, ("initial", "final")):
        assert crossing is not None  # narrowed by the check above
        if not lowest <= crossing <= highest:
            py_warnings.warn(
                f"The tangent crosses the {side} baseline at "
                f"{crossing:.1f} °C, outside the measured range "
                f"({lowest:.1f}-{highest:.1f} °C). The limit on this side is "
                f"the nearest measured point, not the crossing.",
                UserWarning,
            )

    indices = sorted(
        int(np.argmin(np.abs(temperature - crossing)))
        for crossing in crossings
        if crossing is not None
    )
    start_idx, end_idx = indices[0], indices[-1]
    if start_idx == end_idx:
        py_warnings.warn(
            "The two crossings fall on the same point, so they cannot "
            "bracket a transformation. Falling back to the search interval.",
            UserWarning,
        )
        return TransformationLimits(int(n_total * 0.15), int(n_total * 0.85))

    return TransformationLimits(start_idx, end_idx)
