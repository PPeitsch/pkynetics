"""The second-derivative detector: the transformation by its curvature.

Where ``dS/dT`` asks how fast the strain is changing, ``d2S/dT2`` asks where
that rate itself turns. A transformation bends the curve away from its
baseline and back onto another, so it shows up as a pair of curvature
extrema -- one at each end.
"""

import warnings as py_warnings

import numpy as np

from ..curve_features import _strain_derivative
from ..types import TransformationLimits
from .base import DetectionContext


def second_derivative_limits(
    context: DetectionContext,
    prominence_fraction: float = 0.25,
) -> TransformationLimits:
    """Locate the transformation at the extrema of ``d2S/dT2``.

    The two extrema of the curvature are where the curve leaves its first
    baseline and where it settles onto the second, so they bracket the
    transformation directly rather than through a threshold.

    That makes the method sharp on a clean transition and its own worst enemy
    on a noisy one: curvature is a second derivative, so it amplifies noise
    twice over, and on a run with a gentle transformation the largest
    curvature can belong to a wobble in the baseline rather than to the
    transformation. ``prominence_fraction`` is the guard -- an extremum that
    small relative to the largest one is reported as untrustworthy.

    Both derivatives are taken with the same local polynomial fit the first
    derivative uses, applied twice, so the non-uniform temperature ramp does
    not bias the result.

    Args:
        context: The curve and the shared detection settings.
        prominence_fraction: How large the smaller of the two curvature
            extrema has to be, relative to the larger, before the bracket is
            trusted. Below it, a warning is raised.

    Returns:
        :class:`TransformationLimits`, in array order.

    Raises:
        ValueError: If ``prominence_fraction`` is not between 0 and 1.
    """
    if not (0.0 < prominence_fraction < 1.0):
        raise ValueError("prominence_fraction must be between 0 and 1 (exclusive)")

    temperature = context.temperature
    n_total = len(temperature)

    first = _strain_derivative(
        temperature, context.strain, context.window_length, context.polyorder
    )
    second = _strain_derivative(
        temperature, first, context.window_length, context.polyorder
    )

    # The ends of the run are excluded: a Savitzky-Golay fit is least reliable
    # where its window runs off the data, and a second derivative shows that
    # twice as loudly. The margin is the one the baselines already use.
    edge = max(1, int(n_total * context.margin * 0.5))
    interior = slice(edge, n_total - edge)
    curvature = second[interior]
    if curvature.size < 2:
        raise ValueError(
            f"Margin {context.margin:.1%} leaves too few interior points "
            f"({curvature.size}) to find the curvature extrema."
        )

    # A curve with no curvature has no transformation to bracket. What is
    # left on a perfectly linear run is floating-point residue, and its
    # argmax is an arbitrary index, so the scale to compare against is the
    # curvature a transformation spanning the whole run would have.
    strain_scale = float(np.ptp(context.strain))
    temperature_span = float(np.ptp(temperature))
    if strain_scale <= 0.0 or temperature_span <= 0.0:
        py_warnings.warn(
            "The strain does not curve anywhere in the run, so there is no "
            "transformation to bracket. Falling back to the search interval.",
            UserWarning,
        )
        return TransformationLimits(int(n_total * 0.15), int(n_total * 0.85))
    if float(np.max(np.abs(curvature))) <= 1e-6 * strain_scale / temperature_span**2:
        py_warnings.warn(
            "The curvature of the strain is at the level of numerical noise, "
            "so there is no transformation to bracket. Falling back to the "
            "search interval.",
            UserWarning,
        )
        return TransformationLimits(int(n_total * 0.15), int(n_total * 0.85))

    peak = int(np.argmax(curvature)) + edge
    trough = int(np.argmin(curvature)) + edge

    high = abs(float(second[peak]))
    low = abs(float(second[trough]))
    larger, smaller = max(high, low), min(high, low)
    if larger > 0.0 and smaller < prominence_fraction * larger:
        py_warnings.warn(
            f"The two curvature extrema are very uneven ({smaller / larger:.1%} "
            f"of each other). One of them is probably not the end of a "
            f"transformation: the limits below may be unreliable.",
            UserWarning,
        )

    start_idx, end_idx = sorted((peak, trough))
    if start_idx == end_idx:
        py_warnings.warn(
            "The curvature has a single extremum, so it cannot bracket a "
            "transformation. Falling back to the search interval.",
            UserWarning,
        )
        return TransformationLimits(int(n_total * 0.15), int(n_total * 0.85))

    return TransformationLimits(int(start_idx), int(end_idx))
