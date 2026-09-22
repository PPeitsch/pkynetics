"""Type definitions for the dilatometry module."""

from typing import Dict, List, Union

import numpy as np
from numpy.typing import NDArray

#: What the analysis functions return: a flat mapping of result names to
#: scalars, flags, arrays, or the nested quality-metrics mapping.
ReturnDict = Dict[
    str,
    Union[float, bool, str, NDArray[np.float64], Dict[str, Union[float, List[str]]]],
]

__all__ = ["ReturnDict"]
