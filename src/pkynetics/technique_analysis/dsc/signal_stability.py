"""Signal stability detection module.

A region is stable when the signal is flat (STATISTICAL) or linear
(LINEAR_FIT) within its noise. The noise is estimated robustly from the
median absolute deviation of lagged differences, so thresholds adapt to the
data and are given as multiples of the noise level.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .types import StabilityMethod

FloatArray = NDArray[np.float64]


class SignalStabilityDetector:
    """Class for detecting stable regions in signals."""

    def __init__(
        self,
        method: Union[StabilityMethod, str] = StabilityMethod.STATISTICAL,
        min_points: int = 100,
        window_size: int = 50,
        threshold: float = 2.0,
    ):
        """
        Initialize stability detector.

        Args:
            method: STATISTICAL (flat within the noise) or LINEAR_FIT (linear
                within the noise)
            min_points: Minimum number of points of a stable region
            window_size: Size of the sliding window used to test stability.
                Slow drifts are only detected over windows long enough for
                them to exceed the noise: choose a window longer than the
                drifts to be excluded (e.g. several minutes of a ramp)
            threshold: Allowed scatter in a window (standard deviation for
                STATISTICAL, linear-fit residual for LINEAR_FIT) as a
                multiple of the noise level
        """
        self.method = StabilityMethod(method)
        if window_size < 3:
            raise ValueError("Window size must be at least 3")
        self.min_points = min_points
        self.window_size = window_size
        self.threshold = threshold

    def find_stable_regions(
        self,
        signal: FloatArray,
        x_values: Optional[FloatArray] = None,
        method: Optional[Union[StabilityMethod, str]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Find stable regions in a signal.

        Args:
            signal: Signal array
            x_values: Optional x-axis values (time, temperature...); indices
                are used if not given
            method: Override the detector's method

        Returns:
            List of (start_idx, end_idx) tuples, end exclusive
        """
        signal = np.asarray(signal, dtype=np.float64)
        n = len(signal)
        x = (
            np.arange(n, dtype=np.float64)
            if x_values is None
            else np.asarray(x_values, dtype=np.float64)
        )
        if len(x) != n:
            raise ValueError("Signal and x values must have the same length")

        window = self.window_size
        if n < max(window, self.min_points):
            return []

        method = self.method if method is None else StabilityMethod(method)
        std, residual = self._rolling_statistics(signal, x, window)
        scatter = std if method == StabilityMethod.STATISTICAL else residual
        stable_windows = scatter <= self.threshold * self.noise_level(signal)

        # A point is stable if a stable window covers it
        coverage = np.zeros(n + 1, dtype=np.int64)
        starts = np.flatnonzero(stable_windows)
        np.add.at(coverage, starts, 1)
        np.add.at(coverage, starts + window, -1)
        stable = np.cumsum(coverage[:n]) > 0

        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], stable.view(np.int8), [0])))
        )
        return [
            (int(a), int(b))
            for a, b in zip(edges[::2], edges[1::2])
            if b - a >= self.min_points
        ]

    @staticmethod
    def noise_level(signal: FloatArray, lag: int = 10) -> float:
        """
        Estimate the noise standard deviation.

        Uses the median absolute deviation of the differences y[i + lag] -
        y[i]: the median removes linear trends and steps or peaks have
        little weight. The lag (default 10 samples) skips the short-range
        correlation of instrument-filtered noise, for which neighbouring
        differences underestimate the scatter.
        """
        signal = np.asarray(signal, dtype=np.float64)
        lag = max(1, min(lag, len(signal) // 4))
        diff = signal[lag:] - signal[:-lag]
        mad = float(np.median(np.abs(diff - np.median(diff))))
        noise = 1.4826 * mad / np.sqrt(2)
        # Floor for noise-free data, above the rounding error of the
        # running-sum statistics
        scale = max(float(np.ptp(signal)), float(np.max(np.abs(signal))), 1e-12)
        return float(max(noise, 1e-6 * scale))

    @staticmethod
    def _rolling_statistics(
        y: FloatArray, x: FloatArray, window: int
    ) -> Tuple[FloatArray, FloatArray]:
        """Standard deviation and linear-fit residual (rms) of every window
        y[i:i + window], in O(n) from cumulative sums. x is centred per
        window for numerical stability."""

        def rolling_sum(values: FloatArray) -> FloatArray:
            c = np.concatenate(([0.0], np.cumsum(values)))
            result: FloatArray = c[window:] - c[:-window]
            return result

        # Centre y and x globally to limit cancellation
        yc = np.asarray(y - np.mean(y), dtype=np.float64)
        xc = np.asarray((x - np.mean(x)) / (np.ptp(x) or 1.0), dtype=np.float64)

        s_y, s_yy = rolling_sum(yc), rolling_sum(yc**2)
        s_x, s_xx = rolling_sum(xc), rolling_sum(xc**2)
        s_xy = rolling_sum(xc * yc)

        var_y = np.maximum(s_yy / window - (s_y / window) ** 2, 0.0)
        var_x = s_xx / window - (s_x / window) ** 2
        cov_xy = s_xy / window - (s_x / window) * (s_y / window)

        residual_var = np.where(
            var_x > 0, var_y - cov_xy**2 / np.where(var_x > 0, var_x, 1.0), var_y
        )
        return np.sqrt(var_y), np.sqrt(np.maximum(residual_var, 0.0))

    def evaluate_stability(
        self,
        signal: FloatArray,
        region: Tuple[int, int],
        x_values: Optional[FloatArray] = None,
    ) -> Dict[str, float]:
        """
        Stability metrics of a region.

        Args:
            signal: Signal array
            region: (start_idx, end_idx) tuple, end exclusive
            x_values: Optional x-axis values

        Returns:
            Dictionary with mean, std, slope, linear-fit residual, noise
            level (of the whole signal), length and the scatter relative to
            the noise ('relative_std', 'relative_residual')
        """
        signal = np.asarray(signal, dtype=np.float64)
        start, end = region
        if not 0 <= start < end <= len(signal) or end - start < 3:
            raise ValueError("Region must contain at least 3 points of the signal")

        x = (
            np.arange(len(signal), dtype=np.float64)
            if x_values is None
            else np.asarray(x_values, dtype=np.float64)
        )
        y, xs = signal[start:end], x[start:end]
        slope, intercept = np.polyfit(xs, y, 1)
        residual = float(np.sqrt(np.mean((y - (slope * xs + intercept)) ** 2)))
        noise = self.noise_level(signal)
        std = float(np.std(y))

        return {
            "mean": float(np.mean(y)),
            "std": std,
            "slope": float(slope),
            "residual": residual,
            "noise_level": noise,
            "length": float(end - start),
            "relative_std": std / noise,
            "relative_residual": residual / noise,
        }
