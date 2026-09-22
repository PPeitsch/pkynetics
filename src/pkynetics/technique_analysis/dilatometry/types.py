"""Type definitions for the dilatometry module."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

#: What the analysis functions return: a flat mapping of result names to
#: scalars, flags, arrays, or the nested quality-metrics mapping.
ReturnDict = Dict[
    str,
    Union[float, bool, str, NDArray[np.float64], Dict[str, Union[float, List[str]]]],
]


class TransformationLimits(NamedTuple):
    """Where a transformation starts and ends, in array order.

    A :class:`~typing.NamedTuple` rather than a dataclass on purpose: every
    caller of :func:`find_transformation_limits` unpacks the result as
    ``start_idx, end_idx = ...``, and that keeps working.
    """

    start_idx: int
    end_idx: int


@dataclass
class FitQuality:
    """How well the baselines fit, and what looked wrong while fitting them.

    Built by :func:`calculate_fit_quality`. The analysis result carries it as
    a plain mapping -- see :meth:`as_dict` -- so what callers index has not
    changed; the dataclass is what the code passes around.
    """

    r2_start: float
    r2_end: float
    margin_used: float
    deviation_fraction: float
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Union[float, List[str]]]:
        """The mapping form that goes into the analysis result."""
        return asdict(self)


__all__ = ["ReturnDict", "TransformationLimits", "FitQuality"]
