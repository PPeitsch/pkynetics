"""What the curve itself looks like: its derivative and its noise."""

import warnings as py_warnings

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter


def _strain_derivative(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    window_length: int,
    polyorder: int,
) -> NDArray[np.float64]:
    """``dS/dT`` from a local polynomial fit over ``window_length`` points.

    Differentiating the fit is not the same as differencing a smoothed signal.
    ``np.gradient`` on smoothed data is a two-point difference, whose noise
    grows as the spacing shrinks: on a run sampled every 0.25 K it left a
    baseline scatter of 7 % of the peak excursion, which then set the detection
    threshold and pulled the limits 30-110 K inside the transformation. The
    local fit uses every point in the window, so the same run comes down to
    0.1 %.

    Savitzky-Golay differentiates with respect to the sample index, not to
    temperature, and a dilatometry ramp is not uniform in temperature. Both
    derivatives are therefore taken against the index and divided:
    ``dS/dT = (dS/di) / (dT/di)``.

    The division is exact for a ramp whose rate changes smoothly. Where the rate
    changes abruptly -- a program that switches heating rate mid-run -- the
    local fit of ``T(i)`` cannot follow the kink, and ``dT/di`` is wrong for
    about half a window on either side of it: on a synthetic ramp stepping from
    0.6 to 0.2 K per point it doubled the error of the derivative there (#112).
    Analyse each constant-rate segment separately in that case.

    Args:
        temperature: Array of temperature values.
        strain: Array of strain values.
        window_length: Odd Savitzky-Golay window, or 0 to difference directly.
        polyorder: Polynomial order of the local fit.

    Returns:
        ``dS/dT``, same length as the inputs.
    """
    if not window_length:
        return np.asarray(np.gradient(strain, temperature), dtype=np.float64)

    try:
        ds_di = savgol_filter(strain, window_length, polyorder, deriv=1)
        dt_di = savgol_filter(temperature, window_length, polyorder, deriv=1)
    except ValueError:
        py_warnings.warn(
            "The local fit for the derivative failed; differencing the raw "
            "signal instead.",
            UserWarning,
        )
        return np.asarray(np.gradient(strain, temperature), dtype=np.float64)

    # Where the ramp stalls -- an isothermal hold, or a turning point between
    # a heating and a cooling leg -- dT/di goes to zero and dS/dT to infinity.
    # Those points carry no information about a transformation, so they are
    # held at the nearest usable value rather than allowed to become the peak
    # excursion every threshold is measured against.
    scale = float(np.median(np.abs(dt_di)))
    usable = np.abs(dt_di) > 1e-6 * scale if scale > 0 else np.zeros_like(dt_di, bool)
    if not usable.any():
        py_warnings.warn(
            "The temperature does not change over the run; the derivative of "
            "the strain with respect to it is undefined.",
            UserWarning,
        )
        return np.zeros_like(strain, dtype=np.float64)

    derivative = np.empty_like(strain, dtype=np.float64)
    derivative[usable] = ds_di[usable] / dt_di[usable]
    if not usable.all():
        idx = np.arange(len(strain))
        derivative[~usable] = np.interp(idx[~usable], idx[usable], derivative[usable])
    return derivative


def detect_noise_level(
    strain: NDArray[np.float64],
    window_size_fraction: float = 0.05,
    min_window: int = 10,
) -> float:
    """
    Estimate noise level in strain data using median of local standard deviations.

    Args:
        strain: Strain data array.
        window_size_fraction: Fraction of data length for window size.
        min_window: Minimum window size.

    Returns:
        Estimated noise level (median standard deviation).
    """
    n_total = len(strain)
    window_size = int(n_total * window_size_fraction)
    window_size = max(min_window, window_size)
    window_size = min(window_size, n_total // 2)  # Ensure window is not too large

    if window_size < 2:
        return float(np.std(strain) if n_total > 1 else 0.0)  # Explicit cast to float

    try:
        # Calculate standard deviation in sliding windows
        # Using pandas for efficient rolling calculation
        import pandas as pd

        rolling_std = (
            pd.Series(strain)
            .rolling(
                window=window_size, center=True, min_periods=max(2, window_size // 2)
            )
            .std()
        )
        # Use median of the calculated rolling standard deviations (ignoring NaNs at edges)
        median_std = np.nanmedian(rolling_std)
        return (
            float(median_std)
            if not np.isnan(median_std)
            else float(np.std(strain) if n_total > 1 else 0.0)
        )

    except ImportError:
        # Fallback if pandas is not available (less efficient)
        local_std = []
        step = max(1, window_size // 2)  # Use overlapping windows
        for i in range(0, n_total - window_size + 1, step):
            segment = strain[i : i + window_size]
            if len(segment) > 1:
                local_std.append(np.std(segment))
        return float(
            np.median(local_std)
            if local_std
            else (np.std(strain) if n_total > 1 else 0.0)
        )
