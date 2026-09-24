"""Smoothing a signal, by one of several methods.

Every smoothing in the package that is not part of a derivative goes through
:func:`smooth_data`, so a method added here is available everywhere. The
method is chosen with ``method=``; :func:`available_smoothing_methods` lists
the names it takes.

Derivatives are deliberately not here. Smoothing a signal and then
differencing it point to point leaves the noise of a two-point difference,
which grows as the sample spacing shrinks; a derivative has to come from
differentiating a local fit instead, as the dilatometry analysis does.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

_Smoother = Callable[
    [NDArray[np.float64], int, int, Optional[NDArray[np.float64]]],
    NDArray[np.float64],
]


def _savgol(
    data: NDArray[np.float64],
    window_length: int,
    polyorder: int,
    x: Optional[NDArray[np.float64]],
) -> NDArray[np.float64]:
    return np.asarray(savgol_filter(data, window_length, polyorder), dtype=np.float64)


def _moving_average(
    data: NDArray[np.float64],
    window_length: int,
    polyorder: int,
    x: Optional[NDArray[np.float64]],
) -> NDArray[np.float64]:
    # Centred, and narrowed symmetrically near the ends rather than padded:
    # padding with zeros or with the edge value pulls the ends of any signal
    # with an offset or a slope, while a symmetric window leaves a straight
    # line exactly where it was. The price is less smoothing near the ends,
    # down to none at the first and last point.
    n = len(data)
    offset = float(np.mean(data))
    cumulative = np.concatenate(([0.0], np.cumsum(data - offset)))
    idx = np.arange(n)
    half = np.minimum(window_length // 2, np.minimum(idx, n - 1 - idx))
    total = cumulative[idx + half + 1] - cumulative[idx - half]
    return np.asarray(total / (2 * half + 1) + offset, dtype=np.float64)


def _lowess(
    data: NDArray[np.float64],
    window_length: int,
    polyorder: int,
    x: Optional[NDArray[np.float64]],
) -> NDArray[np.float64]:
    # Imported here: statsmodels takes about a second to import, and only
    # this method needs it.
    from statsmodels.nonparametric.smoothers_lowess import lowess

    n = len(data)
    abscissa = np.arange(n, dtype=np.float64) if x is None else x
    fitted = lowess(data, abscissa, frac=window_length / n, return_sorted=False)
    return np.asarray(fitted, dtype=np.float64)


#: Every smoothing method, by the name ``method=`` takes.
SMOOTHERS: Dict[str, _Smoother] = {
    "lowess": _lowess,
    "moving_average": _moving_average,
    "savgol": _savgol,
}

#: What ``method=`` defaults to: the Savitzky-Golay filter the package has
#: always used.
DEFAULT_SMOOTHING = "savgol"


def available_smoothing_methods() -> List[str]:
    """The names ``method=`` accepts, in a stable order."""
    return sorted(SMOOTHERS)


def smooth_data(
    data: NDArray[np.float64],
    window_length: Optional[int] = None,
    polyorder: int = 3,
    method: str = DEFAULT_SMOOTHING,
    x: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Smooth data by the selected method.

    ``savgol``
        Savitzky-Golay: a least-squares polynomial of order ``polyorder`` fitted
        over a moving window. Keeps the height and the position of peaks better
        than an average of the same width.
    ``moving_average``
        A centred moving average. Near the ends the window narrows
        symmetrically instead of being padded, so the first and last points go
        through unsmoothed and a straight line comes out unchanged.
    ``lowess``
        Locally weighted linear regression with robustness iterations
        (Cleveland, 1979), fitted against ``x``. Resistant to isolated spikes,
        and the one method that uses the real abscissa rather than the sample
        index. The slowest: under two seconds for 20 000 points.

    All three are centred, so none of them shifts a feature along the
    abscissa. That is why there is no exponential or weighted moving average:
    both are causal, lag the signal, and would move every temperature an
    analysis reads off it in the same direction.

    Args:
        data: Input data to be smoothed.
        window_length: Length of the window, in points. Made odd if it is
            even. If None, about 5 % of the data length.
        polyorder: Order of the polynomial; used by ``savgol`` only, whose
            window is widened to ``polyorder + 2`` if it is narrower.
        method: One of :func:`available_smoothing_methods`.
        x: Abscissa of the data (temperature, time), for ``lowess`` only. If
            None, the sample index.

    Returns:
        Smoothed data array, same length as ``data``.

    Raises:
        ValueError: If ``method`` is unknown, if the window is not shorter
            than the data, if ``x`` is given for a method other than
            ``lowess``, or if ``x`` is not as long as ``data``.
    """
    smoother = SMOOTHERS.get(method)
    if smoother is None:
        raise ValueError(
            f"Unknown smoothing method {method!r}; "
            f"expected one of {available_smoothing_methods()}"
        )

    values = np.asarray(data, dtype=np.float64)
    abscissa: Optional[NDArray[np.float64]] = None
    if x is not None:
        if method != "lowess":
            raise ValueError(
                f"x is only used by 'lowess'; {method!r} works on the sample index"
            )
        abscissa = np.asarray(x, dtype=np.float64)
        if abscissa.shape != values.shape:
            raise ValueError("x must be as long as data")

    if window_length is None:
        # Calculate appropriate window length (odd number, ~5% of data length)
        window_length = min(len(values) - 2, int(len(values) * 0.05) // 2 * 2 + 1)

    if window_length % 2 == 0:
        window_length += 1

    min_window = polyorder + 2 if method == "savgol" else 3
    if window_length < min_window:
        window_length = min_window + (1 - min_window % 2)

    if window_length >= len(values):
        raise ValueError("Window length must be less than data length")

    return smoother(values, window_length, polyorder, abscissa)
