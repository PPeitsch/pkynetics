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
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Detect stable regions using statistical measures.

        Uses a moving window to calculate local standard deviation
        and identifies regions where it's below the threshold.
        """
        # TODO: Implementar
        pass

    def _linear_fit_method(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
        window_size: int = 100,
        r2_threshold: float = 0.95,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Detect stable regions using linear regression.

        Fits linear model to windows of data and checks R² value
        to identify stable (linear) regions.
        """
        # TODO: Implementar
        pass

    def _adaptive_method(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
        min_segment_size: int = 50,
        error_threshold: float = 0.1,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Detect stable regions using adaptive segmentation.

        Recursively divides signal into segments until each segment
        meets stability criteria.
        """
        # TODO: Implementar
        pass

    def _wavelet_method(
        self,
        signal: NDArray[np.float64],
        x_values: Optional[NDArray[np.float64]] = None,
        wavelet: str = "haar",
        level: int = 3,
        threshold: float = 0.1,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Detect stable regions using wavelet transform.

        Uses wavelet decomposition to identify regions with
        minimal high-frequency components.
        """
        # TODO: Implementar
        pass

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
