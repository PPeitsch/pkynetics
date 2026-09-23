"""The offset detector: a fixed departure from the extrapolated baseline.

The analogue of the 0.2 % offset of a tensile test. Each baseline is fitted
and extrapolated across the run, and the transformation is taken to begin
where the strain has departed from it by a set amount.
"""

import warnings as py_warnings

import numpy as np

from ..linear_segments import get_linear_segment_masks
from ..types import TransformationLimits
from ..utilities import _longest_run
from .base import DetectionContext


def offset_limits(
    context: DetectionContext,
    offset_fraction: float = 0.05,
) -> TransformationLimits:
    """Locate the transformation by its departure from each baseline.

    This is not the deviation-based detection #97 removed, but it shares that
        method's weakness and it is worth being precise about which part. What
        #97 removed was a threshold taken from the scatter of the residuals
        *inside the fitting window*, which measures how straight the baseline is
        rather than how large the transformation is: on a clean curve it collapsed
        to nothing and the first point examined cleared it. The threshold here is
        a fixed fraction of the transformation's own excursion, so it cannot
        collapse, and that is what makes the method reproducible between runs.

        What it still inherits is the **drift**. The departure of the strain from
        an extrapolated line is an integral, and the integral of a baseline that
        is very slightly curved grows without any transformation happening. On the
        shipped Zry-4 heating run, whose initial baseline fits a line at
        R2 = 0.999, that residual curvature is already 1.3 % of the transformation
        excursion inside the fitting window alone, and keeps growing beyond it:

        ========== ============= ============== =============
        ``offset`` Synthetic     Zry-4 heating  Zry-4 cooling
        (true)     (705-795)     (839-936)      (938-757)
        ========== ============= ============== =============
        2 %        711-789       **722**-929    926-772
        5 %        721-779       **739**-923    921-783
        10 %       728-772       **775**-918    913-809
        20 %       736-764       857-910        897-835
        ========== ============= ============== =============

        So the method is reproducible but biased, and the bias is not uniform: on
        a curve with a truly straight baseline it brackets conservatively and
        predictably, while on one with a slight bow it reports a start far too
        early. There is no ``offset_fraction`` that is right for every run, which
        is why this detector warns when its threshold is not clear of the
        baseline's own departure, and why a result worth trusting is one that
        agrees with ``detection="derivative"``.

        Args:
            context: The curve and the shared detection settings.
            offset_fraction: Departure from the baseline that counts as
                transforming, as a fraction of the transformation's full
                excursion. The classical 0.2 % of mechanical testing is an
                absolute strain, which does not carry across instruments and
                units, so it is expressed relative to the excursion instead. The
                default of 0.05 clears the baseline departure of both shipped
                runs; see the table above before trusting it on a new one.

        Returns:
            :class:`TransformationLimits`, in array order.

        Raises:
            ValueError: If ``offset_fraction`` is not between 0 and 1, or a
                baseline window is too short to fit.
    """
    if not (0.0 < offset_fraction < 1.0):
        raise ValueError("offset_fraction must be between 0 and 1 (exclusive)")

    temperature = context.temperature
    strain = context.strain

    start_mask, end_mask = get_linear_segment_masks(
        temperature, context.margin, context.is_cooling
    )
    if np.sum(start_mask) < 2 or np.sum(end_mask) < 2:
        raise ValueError(
            f"Margin {context.margin:.1%} leaves fewer than 2 points in a "
            f"baseline segment."
        )

    # Each end is measured against its own baseline, extrapolated across the
    # whole run: the transformation is where the curve has left the line the
    # material was on before it, and the line it settles onto afterwards is a
    # different one.
    dev_start = np.abs(
        strain
        - np.polyval(
            np.polyfit(temperature[start_mask], strain[start_mask], 1), temperature
        )
    )
    dev_end = np.abs(
        strain
        - np.polyval(
            np.polyfit(temperature[end_mask], strain[end_mask], 1), temperature
        )
    )

    # The excursion is read robustly: a single spike in the raw strain would
    # otherwise set the scale the offset is a fraction of.
    excursion_start = float(np.percentile(dev_start, 99.5))
    excursion_end = float(np.percentile(dev_end, 99.5))
    threshold_start = offset_fraction * excursion_start
    threshold_end = offset_fraction * excursion_end

    # A curve that never leaves its own baseline has no transformation to
    # bracket. The comparison is against the scale of the strain rather than
    # against zero: on a perfectly linear run the departure is floating-point
    # residue, which is not zero and would otherwise be treated as a feature.
    strain_scale = float(np.ptp(strain))
    negligible = 1e-9 * strain_scale
    if strain_scale <= 0.0 or min(excursion_start, excursion_end) <= negligible:
        py_warnings.warn(
            "The strain does not depart from its baseline anywhere in the "
            "run, so there is no transformation to bracket. Falling back to "
            "the search interval.",
            UserWarning,
        )
        n_total = len(temperature)
        return TransformationLimits(int(n_total * 0.15), int(n_total * 0.85))

    # How far each baseline departs from its own fitted line, inside the
    # window it was fitted on. That is the floor the offset has to clear: a
    # threshold below it is measuring the bow of the baseline, not the
    # transformation, and the departure only grows outside the window.
    for mask, dev, threshold, side in (
        (start_mask, dev_start, threshold_start, "initial"),
        (end_mask, dev_end, threshold_end, "final"),
    ):
        floor = float(np.max(dev[mask]))
        if threshold < 2.0 * floor:
            excursion = threshold / offset_fraction
            py_warnings.warn(
                f"The {side} baseline departs from its own fitted line by "
                f"{floor / excursion:.1%} of the transformation excursion, "
                f"against an offset of {offset_fraction:.1%}. The offset is "
                f"not clear of the baseline's own curvature, so the limit on "
                f"this side will be reported too early. Raise "
                f"`offset_fraction`, narrow the analysis range, or compare "
                f"against `detection='derivative'`.",
                UserWarning,
            )

    start_idx = _longest_run(dev_start > threshold_start)[0]
    end_idx = _longest_run(dev_end > threshold_end)[1]

    if start_idx >= end_idx:
        py_warnings.warn(
            f"The offset limits came out in the wrong order (indices "
            f"{start_idx}, {end_idx}). `offset_fraction` "
            f"({offset_fraction:.1%}) may be large enough that the two ends "
            f"meet. Using them in array order.",
            UserWarning,
        )
        start_idx, end_idx = min(start_idx, end_idx), max(start_idx, end_idx)

    return TransformationLimits(int(start_idx), int(end_idx))
