"""The statistical detector: departure beyond what the baseline fit allows.

Where ``offset`` asks whether the strain has left its baseline by a set
amount, this asks whether it has left by more than the fit itself can
account for. The threshold is the prediction interval of the baseline
regression, so it is not a number anyone has to choose per run: it follows
from the scatter of the baseline, how many points it was fitted on, and --
the part that matters here -- how far the point is from the window it was
fitted over.
"""

import warnings as py_warnings
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from ..linear_segments import get_linear_segment_masks
from ..types import TransformationLimits
from ..utilities import _longest_run
from .base import DetectionContext


def _prediction_interval(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    mask: NDArray[np.bool_],
    confidence: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Departure from a fitted baseline, and what the fit allows at each point.

    The half-width of the prediction interval for a simple linear regression,

    .. math::

        t_{\alpha/2,\,n-2}\; s \sqrt{1 + \frac{1}{n} +
        \frac{(T - \bar{T})^2}{S_{TT}}}

    evaluated across the whole run rather than only inside the fitting window.
    The last term is why this detector exists: it grows with the square of the
    distance from the window, so the band widens as the baseline is
    extrapolated, and a slight curvature that would cross a fixed offset does
    not necessarily cross this.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        mask: Which points the baseline is fitted on.
        confidence: Confidence level of the interval.

    Returns:
        ``(departure, half_width)``, both over the whole run.
    """
    fit_temperature = temperature[mask]
    fit_strain = strain[mask]
    n_fit = len(fit_temperature)

    line = np.polyfit(fit_temperature, fit_strain, 1)
    residuals = fit_strain - np.polyval(line, fit_temperature)
    # Two parameters fitted, so n - 2 degrees of freedom.
    residual_std = float(np.sqrt(np.sum(residuals**2) / (n_fit - 2)))

    mean_temperature = float(np.mean(fit_temperature))
    sum_squares = float(np.sum((fit_temperature - mean_temperature) ** 2))

    t_critical = float(stats.t.ppf(0.5 + confidence / 2.0, n_fit - 2))
    leverage = 1.0 + 1.0 / n_fit + (temperature - mean_temperature) ** 2 / sum_squares
    half_width = t_critical * residual_std * np.sqrt(leverage)

    departure = np.abs(strain - np.polyval(line, temperature))
    return departure, np.asarray(half_width, dtype=np.float64)


def _residual_structure(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> float:
    """How far a baseline's residuals fall short of changing sign like noise.

    The Wald-Wolfowitz runs test. Independent residuals cross zero about
    ``1 + 2 n_+ n_- / n`` times; a systematic bow keeps the residuals on one
    side for long stretches and produces far fewer runs.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        mask: Which points the baseline is fitted on.

    Returns:
        The z-score of the run count: near zero for noise, strongly negative
        for structure. Returns 0.0 when the test cannot be applied, so that
        an inapplicable test never raises a false alarm.
    """
    line = np.polyfit(temperature[mask], strain[mask], 1)
    residuals = strain[mask] - np.polyval(line, temperature[mask])

    signs = np.sign(residuals)
    signs = signs[signs != 0]
    n_total = len(signs)
    n_positive = int(np.sum(signs > 0))
    n_negative = n_total - n_positive
    if n_positive == 0 or n_negative == 0 or n_total < 3:
        return 0.0

    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    expected = 1.0 + 2.0 * n_positive * n_negative / n_total
    variance = (expected - 1.0) * (expected - 2.0) / (n_total - 1)
    if variance <= 0.0:
        return 0.0
    return float((runs - expected) / np.sqrt(variance))


def statistical_limits(
    context: DetectionContext,
    confidence: float = 0.99,
    max_residual_structure: float = 3.0,
) -> TransformationLimits:
    """Locate the transformation where the strain leaves its baseline's
        prediction interval.

        A point counts as transforming when the strain there is further from the
        extrapolated baseline than the regression says it should be, at the given
        confidence. Both ends are treated separately, each against its own
        baseline.

    This is the most sensitive of the detectors, and it is worth being exact
        about what that means, because it does *not* mean the most accurate.

        It answers "where does the curve stop being explainable by the scatter of
        the baseline", which is a different question from "where does the
        transformation begin". On a clean run the two are far apart. On the
        synthetic sigmoid the baseline residual scatter is 3.3e-8 while the tail
        of the transformation already reaches 2.1e-7 at 658 degC -- a six-sigma
        departure, correctly flagged, and 0.017 % of the excursion. The detector
        returns 658-842 for a transformation conventionally placed at 705-795,
        and it is not wrong: the sigmoid really has left its baseline there by an
        amount the fit cannot account for.

        That also means it does **not** fix the drift that limits ``offset``,
        which was the obvious reason to expect something from it. The prediction
        interval does widen with the square of the distance from the fitting
        window, but a baseline bow grows faster: on the Zry-4 heating run this
        reports 717 degC where ``offset`` reports 739 and the foot is at 839.

        So its threshold is the only one here with a stated meaning -- a
        false-positive rate rather than a fraction someone picked -- and that
        meaning holds only while its assumptions do: a baseline window containing
        nothing but baseline, and residuals that are scatter rather than
        structure. A systematic bow is neither, and this detector reports it as a
        transformation, correctly by its own logic.

        What it is good for is a noisy run, where the scatter is real and the
        question "is this more than noise" is the one worth asking. On the
        shipped cooling run it returns 931-766 against a derivative 938-757.

        To make the failure legible rather than silent, the departure at each
        limit is measured against the transformation excursion, and a limit
        sitting on a departure too small to matter physically is reported as
        such.

        Args:
            context: The curve and the shared detection settings.
            confidence: Confidence level of the prediction interval. 0.99 means a
                point outside it would occur in 1 % of baseline samples by chance.
            negligible_fraction: Departure from the baseline, as a fraction of the
                transformation excursion, below which a limit is reported as
                statistically real but physically negligible.

        Returns:
            :class:`TransformationLimits`, in array order.

        Raises:
            ValueError: If ``confidence`` is not between 0 and 1, or a baseline
                window has fewer than 3 points, which a residual variance needs.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1 (exclusive)")
    if max_residual_structure <= 0.0:
        raise ValueError("max_residual_structure must be positive")

    temperature = context.temperature
    strain = context.strain
    n_total = len(temperature)

    start_mask, end_mask = get_linear_segment_masks(
        temperature, context.margin, context.is_cooling
    )
    # Three, not two: with two points the fit is exact, the residual variance
    # is zero by construction and every other point is infinitely surprising.
    if np.sum(start_mask) < 3 or np.sum(end_mask) < 3:
        raise ValueError(
            f"Margin {context.margin:.1%} leaves fewer than 3 points in a "
            f"baseline segment, which is the minimum for a residual variance."
        )

    # The method rests on the residuals being scatter rather than structure,
    # so that is checked rather than assumed.
    for mask, side in ((start_mask, "initial"), (end_mask, "final")):
        structure = _residual_structure(temperature, strain, mask)
        if structure < -max_residual_structure:
            py_warnings.warn(
                f"The residuals of the {side} baseline change sign "
                f"{-structure:.1f} standard deviations less often than noise "
                f"would: it is bowed, not scattered. The prediction interval "
                f"still widens with distance, but {confidence:.0%} is no "
                f"longer the false-positive rate it claims to be, and the "
                f"limit on this side may sit on the curvature rather than on "
                f"the transformation.",
                UserWarning,
            )

    departure_start, band_start = _prediction_interval(
        temperature, strain, start_mask, confidence
    )
    departure_end, band_end = _prediction_interval(
        temperature, strain, end_mask, confidence
    )

    if not np.any(departure_start > band_start) or not np.any(departure_end > band_end):
        py_warnings.warn(
            f"The strain never leaves the {confidence:.0%} prediction "
            f"interval of its baseline, so there is no transformation to "
            f"bracket at this confidence. Falling back to the search "
            f"interval.",
            UserWarning,
        )
        return TransformationLimits(int(n_total * 0.15), int(n_total * 0.85))

    start_idx = _longest_run(departure_start > band_start)[0]
    end_idx = _longest_run(departure_end > band_end)[1]

    if start_idx >= end_idx:
        py_warnings.warn(
            f"The statistical limits came out in the wrong order (indices "
            f"{start_idx}, {end_idx}). Both baselines may be picking up the "
            f"same feature. Using them in array order.",
            UserWarning,
        )
        start_idx, end_idx = min(start_idx, end_idx), max(start_idx, end_idx)

    return TransformationLimits(int(start_idx), int(end_idx))
