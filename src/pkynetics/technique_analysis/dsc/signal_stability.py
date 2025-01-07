"""Signal stability detection module."""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats


class StabilityMethod(Enum):
    """Available methods for stability detection."""

    DERIVATIVE = "derivative"  # Using signal derivative
    STATISTICAL = "statistical"  # Using statistical measures
    LINEAR_FIT = "linear_fit"  # Using linear regression
    ADAPTIVE = "adaptive"  # Using adaptive segmentation
    WAVELET = "wavelet"  # Using wavelet transform


class SignalStabilityDetector:
    """Class for detecting stable regions in signals."""

    def __init__(
        self,
        method: Union[StabilityMethod, str] = StabilityMethod.STATISTICAL,
        min_points: int = 100,
        **kwargs,
    ):
        """
        Initialize stability detector.

        Args:
            method: Method to use for stability detection
            min_points: Minimum number of points for a stable region
            **kwargs: Additional parameters for specific methods
        """
        if isinstance(method, str):
            method = StabilityMethod(method)

        self.method = method
        self.min_points = min_points
        self.params = kwargs

    def find_stable_regions(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Find stable regions in signal.

        Args:
            signal: Signal array
            x_values: Optional x-axis values (time or other independent variable)
            **kwargs: Additional method-specific parameters

        Returns:
            List of (start_idx, end_idx) tuples for stable regions
        """
        params = {**self.params, **kwargs}

        detection_methods = {
            StabilityMethod.DERIVATIVE: self._derivative_method,
            StabilityMethod.STATISTICAL: self._statistical_method,
            StabilityMethod.LINEAR_FIT: self._linear_fit_method,
            StabilityMethod.ADAPTIVE: self._adaptive_method,
            StabilityMethod.WAVELET: self._wavelet_method,
        }

        if self.method not in detection_methods:
            raise ValueError(f"Unknown stability detection method: {self.method}")

        return detection_methods[self.method](signal, x_values, **params)

    def _derivative_method(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
        threshold: float = 0.1,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """Detect stable regions using signal derivative."""
        # TODO: Implementar
        pass

    def _statistical_method(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
        window_size: int = 50,
        std_threshold: float = 0.1,
        overlap: float = 0.5,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Detect stable regions using statistical measures.

        Args:
            signal: Signal array
            x_values: Optional x-axis values (not used in this method)
            window_size: Size of the sliding window
            std_threshold: Maximum allowed standard deviation relative to signal range
            overlap: Fraction of overlap between windows (0 to 1)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of (start_idx, end_idx) tuples for stable regions
        """
        if len(signal) < window_size:
            return []

        # Calculate absolute threshold based on signal range
        signal_range = np.ptp(signal)
        abs_threshold = signal_range * std_threshold

        # Calculate window step size
        step = int(window_size * (1 - overlap))
        if step < 1:
            step = 1

        # Initialize variables
        stable_regions = []
        n_windows = (len(signal) - window_size) // step + 1
        window_stats = np.zeros(n_windows)

        # Calculate standard deviation for each window
        for i in range(n_windows):
            start_idx = i * step
            end_idx = start_idx + window_size
            window_stats[i] = np.std(signal[start_idx:end_idx])

        # Find regions where std is below threshold
        stable_mask = window_stats < abs_threshold

        # Find continuous stable regions
        start_idx = None
        for i in range(len(stable_mask)):
            if stable_mask[i] and start_idx is None:
                start_idx = i * step
            elif (
                not stable_mask[i] or i == len(stable_mask) - 1
            ) and start_idx is not None:
                end_idx = i * step + window_size
                if end_idx - start_idx >= self.min_points:
                    # Refine region boundaries
                    region_signal = signal[start_idx:end_idx]
                    local_stds = np.array(
                        [
                            np.std(region_signal[j : j + window_size])
                            for j in range(0, len(region_signal) - window_size, step)
                        ]
                    )
                    # Find the most stable section within the region
                    best_idx = np.argmin(local_stds)
                    refined_start = start_idx + best_idx * step
                    refined_end = min(refined_start + window_size, len(signal))
                    stable_regions.append((refined_start, refined_end))
                start_idx = None

        return stable_regions

    def _linear_fit_method(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
        window_size: int = 100,
        r2_threshold: float = 0.95,
        slope_tolerance: float = 0.1,
        overlap: float = 0.5,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Detect stable regions using linear regression.

        Args:
            signal: Signal array
            x_values: Optional x-axis values
            window_size: Size of the sliding window
            r2_threshold: Minimum R² value for linear fit
            slope_tolerance: Maximum allowed relative change in slope between windows
            overlap: Fraction of overlap between windows (0 to 1)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of (start_idx, end_idx) tuples for stable regions
        """
        if len(signal) < window_size:
            return []

        if x_values is None:
            x_values = np.arange(len(signal))

        # Calculate window step size
        step = int(window_size * (1 - overlap))
        if step < 1:
            step = 1

        # Initialize variables
        stable_regions = []
        n_windows = (len(signal) - window_size) // step + 1
        window_stats = np.zeros((n_windows, 3))  # [R², slope, intercept]

        # Calculate linear fit statistics for each window
        for i in range(n_windows):
            start_idx = i * step
            end_idx = start_idx + window_size
            window_x = x_values[start_idx:end_idx]
            window_y = signal[start_idx:end_idx]

            # Perform linear regression
            slope, intercept, r_value, _, _ = stats.linregress(window_x, window_y)
            window_stats[i] = [r_value**2, slope, intercept]

        # Find regions with good linear fit
        r2_mask = window_stats[:, 0] >= r2_threshold

        # Check for consistent slope
        slope_diffs = np.abs(np.diff(window_stats[:, 1]))
        mean_slope = np.mean(np.abs(window_stats[:, 1]))
        slope_mask = np.concatenate(
            ([True], slope_diffs <= slope_tolerance * mean_slope)
        )

        # Combine criteria
        stable_mask = r2_mask & slope_mask

        # Find continuous stable regions
        start_idx = None
        for i in range(len(stable_mask)):
            if stable_mask[i] and start_idx is None:
                start_idx = i * step
            elif (
                not stable_mask[i] or i == len(stable_mask) - 1
            ) and start_idx is not None:
                end_idx = i * step + window_size
                if end_idx - start_idx >= self.min_points:
                    # Refine region boundaries
                    region_signal = signal[start_idx:end_idx]
                    region_x = x_values[start_idx:end_idx]

                    # Calculate R² for sub-windows to find most linear section
                    r2_values = []
                    for j in range(0, len(region_signal) - window_size, step):
                        sub_x = region_x[j : j + window_size]
                        sub_y = region_signal[j : j + window_size]
                        _, _, r_value, _, _ = stats.linregress(sub_x, sub_y)
                        r2_values.append(r_value**2)

                    if r2_values:
                        # Find the most linear section
                        best_idx = np.argmax(r2_values)
                        refined_start = start_idx + best_idx * step
                        refined_end = min(refined_start + window_size, len(signal))
                        stable_regions.append((refined_start, refined_end))
                start_idx = None

        return stable_regions

    def _adaptive_method(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
        min_segment_size: int = 50,
        error_threshold: float = 0.1,
        max_depth: int = 10,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Detect stable regions using adaptive segmentation.

        This method recursively divides the signal into segments until each segment
        meets stability criteria or minimum size is reached.

        Args:
            signal: Signal array
            x_values: Optional x-axis values
            min_segment_size: Minimum allowed segment size
            error_threshold: Maximum allowed error relative to linear fit
            max_depth: Maximum recursion depth
            **kwargs: Additional parameters

        Returns:
            List of (start_idx, end_idx) tuples for stable regions
        """
        if x_values is None:
            x_values = np.arange(len(signal))

        stable_regions = []

        def segment_error(start: int, end: int) -> float:
            """Calculate error between signal and linear fit."""
            if end - start < 2:
                return np.inf

            x = x_values[start:end]
            y = signal[start:end]

            # Linear fit
            slope, intercept, _, _, _ = stats.linregress(x, y)
            y_fit = slope * x + intercept

            # Normalized error
            error = np.sqrt(np.mean((y - y_fit) ** 2)) / np.ptp(y)
            return error

        def find_split_point(start: int, end: int) -> int:
            """Find optimal point to split segment."""
            if end - start < 3:
                return start + (end - start) // 2

            x = x_values[start:end]
            y = signal[start:end]

            # Linear fit
            slope, intercept, _, _, _ = stats.linregress(x, y)
            y_fit = slope * x + intercept

            # Find point of maximum deviation
            errors = np.abs(y - y_fit)
            return start + np.argmax(errors)

        def recursive_segment(start: int, end: int, depth: int = 0) -> None:
            """Recursively segment signal."""
            if depth >= max_depth or end - start < min_segment_size:
                return

            error = segment_error(start, end)

            if error < error_threshold:
                if end - start >= self.min_points:
                    stable_regions.append((start, end))
                return

            # Split segment
            split = find_split_point(start, end)
            if split - start >= min_segment_size:
                recursive_segment(start, split, depth + 1)
            if end - split >= min_segment_size:
                recursive_segment(split, end, depth + 1)

        # Start recursive segmentation
        recursive_segment(0, len(signal))

        # Sort regions by start index
        stable_regions.sort(key=lambda x: x[0])
        return stable_regions

    def _wavelet_method(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
        wavelet: str = "db4",
        level: int = 3,
        threshold: float = 0.1,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Detect stable regions using wavelet transform.

        This method uses wavelet decomposition to identify regions with
        minimal high-frequency components.

        Args:
            signal: Signal array
            x_values: Optional x-axis values
            wavelet: Wavelet type to use
            level: Decomposition level
            threshold: Threshold for coefficient magnitudes
            **kwargs: Additional parameters

        Returns:
            List of (start_idx, end_idx) tuples for stable regions
        """
        try:
            import pywt
        except ImportError:
            raise ImportError(
                "pywt is required for wavelet method. Install with: pip install PyWavelets"
            )

        if len(signal) < self.min_points:
            return []

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Analyze detail coefficients
        detail_coeffs = coeffs[1:]

        # Calculate threshold for each level
        thresholds = []
        for detail in detail_coeffs:
            # Use signal range to normalize threshold
            detail_range = np.ptp(detail)
            thresholds.append(detail_range * threshold)

        # Initialize stability mask
        stability_mask = np.ones(len(signal), dtype=bool)

        # Analyze each level
        for detail, thresh in zip(detail_coeffs, thresholds):
            # Upsample detail coefficients to match signal length
            detail_upsampled = np.repeat(detail, 2 ** (level))[: len(signal)]

            # Update stability mask
            stability_mask &= np.abs(detail_upsampled) < thresh

        # Find continuous stable regions
        stable_regions = []
        start_idx = None

        for i in range(len(stability_mask)):
            if stability_mask[i] and start_idx is None:
                start_idx = i
            elif (
                not stability_mask[i] or i == len(stability_mask) - 1
            ) and start_idx is not None:
                end_idx = i
                if end_idx - start_idx >= self.min_points:
                    stable_regions.append((start_idx, end_idx))
                start_idx = None

        return stable_regions

    def evaluate_stability(
        self, signal: NDArray[np.float64], region: Tuple[int, int], **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate stability metrics for a given region.

        Args:
            signal: Signal array
            region: (start_idx, end_idx) tuple defining the region
            **kwargs: Additional parameters

        Returns:
            Dictionary of stability metrics
        """
        start_idx, end_idx = region
        segment = signal[start_idx:end_idx]

        metrics = {
            "std": float(np.std(segment)),
            "range": float(np.ptp(segment)),
            "linearity": self._calculate_linearity(segment),
            "length": end_idx - start_idx,
        }

        return metrics

    def _calculate_linearity(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """Calculate linearity metric for a signal segment."""
        if x_values is None:
            x_values = np.arange(len(signal))

        slope, intercept, r_value, _, _ = stats.linregress(x_values, signal)
        return float(r_value**2)  # R²
