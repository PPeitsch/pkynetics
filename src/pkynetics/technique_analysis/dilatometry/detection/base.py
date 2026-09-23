"""What every detector receives, and what it has to return.

A detector answers one question -- where does the transformation start and
end -- and nothing else. How far it has gone by then is the job of
:mod:`~pkynetics.technique_analysis.dilatometry.methods`, and the two are
chosen independently: any detector combines with either method.

The shared inputs are gathered into :class:`DetectionContext` so that adding
a detector means writing one function, not threading another parameter
through the whole call chain. Anything specific to a single detector stays a
keyword argument of that detector.
"""

from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..types import TransformationLimits


class DetectionContext(NamedTuple):
    """The curve and the settings every detector needs."""

    #: Array of temperature values (°C).
    temperature: NDArray[np.float64]
    #: Array of strain or relative length change values.
    strain: NDArray[np.float64]
    #: Whether this is a cooling segment.
    is_cooling: bool
    #: Fraction of the data at each end taken as baseline.
    margin: float
    #: Savitzky-Golay window, already resolved to a point count, for
    #: detectors that smooth or differentiate.
    window_length: int
    #: Polynomial order for that smoothing.
    polyorder: int
    #: R² below which a baseline window is reported as reaching into a
    #: transformation.
    baseline_min_r2: float


#: A detector: takes a :class:`DetectionContext` and its own keyword options,
#: and answers with the limits. The options are deliberately not pinned down
#: here -- each detector names the ones it takes, so that passing an option
#: meant for another one is an error at the call rather than a silently
#: ignored keyword.
Detector = Callable[..., TransformationLimits]


__all__ = ["DetectionContext", "Detector"]
