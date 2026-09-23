"""Where the transformation starts and ends.

One question, several ways to answer it. The detector is chosen with
``detection=`` wherever an analysis is run; it is independent of ``method=``,
which picks how the transformed fraction is measured.

Adding a detector is adding a function with the signature in
:class:`~...detection.base.Detector` and a line in :data:`DETECTORS`.
"""

from typing import Dict, List

from .base import DetectionContext, Detector
from .derivative import derivative_limits
from .double_tangent import double_tangent_limits
from .offset import offset_limits
from .second_derivative import second_derivative_limits

#: Every detector, by the name ``detection=`` takes.
DETECTORS: Dict[str, Detector] = {
    "derivative": derivative_limits,
    "double_tangent": double_tangent_limits,
    "offset": offset_limits,
    "second_derivative": second_derivative_limits,
}

#: What ``detection=`` defaults to: the behaviour the module has had since #97.
DEFAULT_DETECTION = "derivative"


def available_detectors() -> List[str]:
    """The names ``detection=`` accepts, in a stable order."""
    return sorted(DETECTORS)


def get_detector(name: str) -> Detector:
    """The detector registered under ``name``.

    Args:
        name: A detector name, as ``detection=`` takes it.

    Returns:
        The detector function.

    Raises:
        ValueError: If no detector is registered under that name. The message
            lists the ones that are.
    """
    try:
        return DETECTORS[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown detection method: '{name}'. "
            f"Available: {', '.join(available_detectors())}."
        ) from None


__all__ = [
    "DETECTORS",
    "DEFAULT_DETECTION",
    "DetectionContext",
    "Detector",
    "available_detectors",
    "get_detector",
    "derivative_limits",
    "double_tangent_limits",
    "offset_limits",
    "second_derivative_limits",
]
