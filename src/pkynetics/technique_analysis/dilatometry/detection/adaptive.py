"""Choosing the baseline margin from the curve instead of by hand.

``margin`` is the fraction of the run taken as baseline at each end. It has
always been a number the caller sets, with a default of 0.2 that works on the
runs this package ships -- and the reason to replace that with something
measured is not convenience. It is that the margin has narrow dead zones, and
nothing about a curve tells you where they are.

On the Zry-4 heating run the derivative detector returns 839-936 degC for any
margin from 0.18 to 0.25, and 822-837 -- a 15 K bracket on a 97 K
transformation -- for 0.15, 0.16 and 0.17. Both ends of that range look
equally reasonable, and the baselines fit a line just as well inside the dead
zone (R2 = 0.9989) as outside it (R2 = 0.9991), so linearity cannot tell them
apart either.

What does tell them apart is stability: the right margin is one whose answer
does not depend on the margin. A dead zone is narrow by nature -- that is what
makes it a trap rather than a region -- so the answer that holds across the
widest stretch of margins is the one to take.
"""

import warnings as py_warnings
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..types import TransformationLimits

#: Margins tried when ``margin="auto"``, spanning the range the module accepts.
DEFAULT_CANDIDATES: Tuple[float, ...] = tuple(
    round(float(value), 2) for value in np.arange(0.10, 0.351, 0.01)
)


def _plateaus(
    ordered: List[Tuple[float, TransformationLimits]],
    tolerance: int,
) -> List[List[Tuple[float, TransformationLimits]]]:
    """Group consecutive margins that give the same answer.

    Args:
        ordered: ``(margin, limits)`` pairs, by increasing margin.
        tolerance: How many points each limit may move and still count as the
            same answer.

    Returns:
        The groups, in the order they were found.
    """
    groups: List[List[Tuple[float, TransformationLimits]]] = []
    for margin, limits in ordered:
        if groups:
            _, previous = groups[-1][-1]
            same = (
                abs(limits.start_idx - previous.start_idx) <= tolerance
                and abs(limits.end_idx - previous.end_idx) <= tolerance
            )
            if same:
                groups[-1].append((margin, limits))
                continue
        groups.append([(margin, limits)])
    return groups


def stable_margin(
    run: Callable[[float], TransformationLimits],
    n_total: int,
    candidates: Sequence[float] = DEFAULT_CANDIDATES,
    tolerance_fraction: float = 0.01,
) -> Tuple[float, TransformationLimits]:
    """The margin whose answer holds over the widest range of margins.

    Runs the detector at each candidate margin, groups the margins that give
    the same limits, and returns the middle of the largest group. Warnings
    raised while exploring are suppressed: they belong to margins that were
    tried and mostly not chosen, and the caller re-runs the detector at the
    chosen margin, where the warnings that do apply are raised normally.

    Args:
        run: Runs the detector at a given margin. May raise ``ValueError`` for
            a margin that leaves too little data, which is treated as that
            margin being unusable rather than as a failure.
        n_total: Number of points in the run, for the tolerance.
        candidates: Margins to try, in increasing order.
        tolerance_fraction: How far the limits may move, as a fraction of the
            run, and still count as the same answer.

    Returns:
        ``(margin, limits)`` for the middle of the widest plateau.

    Raises:
        ValueError: If no candidate margin produces a result at all.
    """
    results: Dict[float, TransformationLimits] = {}
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore")
        for margin in candidates:
            try:
                results[margin] = run(margin)
            except ValueError:
                continue  # this margin leaves too little data; try the next

    if not results:
        raise ValueError(
            f"No margin between {min(candidates):.0%} and {max(candidates):.0%} "
            f"leaves enough data on both sides to locate a transformation."
        )

    ordered = sorted(results.items())
    tolerance = max(1, int(n_total * tolerance_fraction))
    groups = _plateaus(ordered, tolerance)
    widest = max(groups, key=len)

    if len(widest) == 1:
        py_warnings.warn(
            "No margin gives an answer that holds for a neighbouring margin, "
            "so there is no stable choice to make: the limits depend on the "
            "margin throughout. Using the middle of the range tried. Narrow "
            "the analysis range, or set `margin` explicitly and read the "
            "result with that in mind.",
            UserWarning,
        )

    return widest[len(widest) // 2]
