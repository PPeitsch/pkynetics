"""Baseline correction methods for DSC data."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, signal
from scipy.interpolate import UnivariateSpline

from .types import BaselineResult


class BaselineCorrector:
    """Enhanced baseline correction for DSC data."""

    def __init__(self, smoothing_window: int = 21, smoothing_order: int = 3):
        """
        Initialize baseline corrector.

        Args:
            smoothing_window: Window size for Savitzky-Golay smoothing
            smoothing_order: Order for Savitzky-Golay smoothing
        """
        self.smoothing_window = smoothing_window
        self.smoothing_order = smoothing_order

        # Register available correction methods
        self.methods = {
            "linear": self._fit_linear_baseline,
            "polynomial": self._fit_polynomial_baseline,
            "spline": self._fit_spline_baseline,
            "asymmetric": self._fit_asymmetric_baseline,
            "rubberband": self._fit_rubberband_baseline,
            "auto": self._auto_baseline,
        }

    def correct(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        method: str = "auto",
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> BaselineResult:
        """
        Apply baseline correction with specified method.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            method: Correction method to use
            regions: Optional list of (start_temp, end_temp) for baseline regions
            **kwargs: Additional parameters for specific correction methods

        Returns:
            BaselineResult object containing correction results
        """
        if method not in self.methods:
            raise ValueError(f"Unknown baseline method: {method}")

        # Validate input data
        self._validate_data(temperature, heat_flow)

        # Smooth data if needed
        heat_flow_smooth = signal.savgol_filter(
            heat_flow, self.smoothing_window, self.smoothing_order
        )

        # Apply selected correction method
        correction_func = self.methods[method]
        baseline, params = correction_func(
            temperature, heat_flow_smooth, regions, **kwargs
        )

        # Calculate corrected data
        corrected_data = heat_flow - baseline

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            temperature, heat_flow, baseline, regions
        )

        return BaselineResult(
            baseline=baseline,
            corrected_data=corrected_data,
            method=method,
            parameters=params,
            quality_metrics=quality_metrics,
            regions=regions,
        )

    def optimize_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        method: str = "auto",
        n_regions: int = 4,
    ) -> BaselineResult:
        """
        Find optimal baseline regions automatically.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            method: Baseline method to use
            n_regions: Number of baseline regions to identify

        Returns:
            Optimized BaselineResult
        """
        # Find quiet regions in the data
        regions = self._find_quiet_regions(temperature, heat_flow, n_regions)

        # Try different region combinations
        best_result = None
        best_score = float("inf")

        for i in range(min(10, len(regions))):  # Limit number of combinations
            test_regions = regions[i : i + n_regions]
            result = self.correct(temperature, heat_flow, method, test_regions)
            score = self._evaluate_baseline_quality(result)

            if score < best_score:
                best_score = score
                best_result = result

        return best_result

    def _fit_linear_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Fit linear baseline through specified regions."""
        if regions is None:
            regions = [(temperature[0], temperature[-1])]

        # Collect points in baseline regions
        temp_points = []
        heat_points = []
        for start_temp, end_temp in regions:
            mask = (temperature >= start_temp) & (temperature <= end_temp)
            temp_points.extend(temperature[mask])
            heat_points.extend(heat_flow[mask])

        # Fit linear baseline
        coeffs = np.polyfit(temp_points, heat_points, 1)
        baseline = np.polyval(coeffs, temperature)

        params = {"slope": float(coeffs[0]), "intercept": float(coeffs[1])}

        return baseline, params

    def _fit_polynomial_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        degree: int = 3,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Fit polynomial baseline of specified degree."""
        if regions is None:
            n_points = len(temperature) // 10
            regions = [
                (temperature[0], temperature[n_points]),
                (temperature[-n_points:], temperature[-1]),
            ]

        # Collect points in baseline regions
        temp_points = []
        heat_points = []
        for start_temp, end_temp in regions:
            mask = (temperature >= start_temp) & (temperature <= end_temp)
            temp_points.extend(temperature[mask])
            heat_points.extend(heat_flow[mask])

        # Fit polynomial
        coeffs = np.polyfit(temp_points, heat_points, degree)
        baseline = np.polyval(coeffs, temperature)

        params = {"coefficients": coeffs.tolist(), "degree": degree}

        return baseline, params

    def _fit_spline_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        smoothing: float = 1.0,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Fit spline baseline with automatic knot selection."""
        if regions is None:
            regions = self._find_quiet_regions(temperature, heat_flow)

        # Collect points in baseline regions
        temp_points = []
        heat_points = []
        for start_temp, end_temp in regions:
            mask = (temperature >= start_temp) & (temperature <= end_temp)
            temp_points.extend(temperature[mask])
            heat_points.extend(heat_flow[mask])

        # Fit univariate spline
        spline = UnivariateSpline(temp_points, heat_points, s=smoothing)
        baseline = spline(temperature)

        params = {"smoothing": smoothing, "n_knots": len(spline.get_knots())}

        return baseline, params

    def _fit_asymmetric_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Fit asymmetric least squares baseline."""

        def als_baseline(
            y: NDArray[np.float64], lam: float, p: float, niter: int = 10
        ) -> NDArray[np.float64]:
            L = len(y)
            D = np.diff(np.eye(L), 2)
            w = np.ones(L)
            for i in range(niter):
                W = np.diag(w)
                Z = np.linalg.inv(W + lam * D.T.dot(D))
                z = Z.dot(w * y)
                w = p * (y > z) + (1 - p) * (y < z)
            return z

        # Parameters for asymmetric least squares
        lam = kwargs.get("lam", 1e5)
        p = kwargs.get("p", 0.001)

        baseline = als_baseline(heat_flow, lam, p)

        params = {"lambda": lam, "p": p}

        return baseline, params

    def _fit_rubberband_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Fit rubberband (convex hull) baseline."""
        # Find convex hull of the data points
        points = np.column_stack((temperature, heat_flow))
        hull = optimize.convex_hull_plot_2d(points)

        # Extract lower hull points
        hull_points = points[hull.vertices]
        lower_hull = hull_points[hull_points[:, 0].argsort()]

        # Interpolate between hull points
        baseline = np.interp(temperature, lower_hull[:, 0], lower_hull[:, 1])

        params = {"n_hull_points": len(lower_hull)}

        return baseline, params

    def _auto_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Automatically select and apply best baseline method."""
        methods = ["linear", "polynomial", "spline", "asymmetric"]
        best_score = float("inf")
        best_result = None

        for method in methods:
            result = self.correct(temperature, heat_flow, method, regions)
            score = self._evaluate_baseline_quality(result)

            if score < best_score:
                best_score = score
                best_result = result

        return best_result.baseline, best_result.parameters

    def _find_quiet_regions(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        n_regions: int = 4,
        window: int = 20,
    ) -> List[Tuple[float, float]]:
        """Find quiet (low variance) regions in the data."""
        # Calculate local variance
        rolling_var = np.array(
            [np.var(heat_flow[i : i + window]) for i in range(len(heat_flow) - window)]
        )

        # Find minima in variance
        var_minima = signal.argrelmin(rolling_var)[0]

        # Sort by variance value and select top n_regions
        sorted_minima = var_minima[np.argsort(rolling_var[var_minima])]
        selected_points = sorted_minima[:n_regions]

        # Create regions around selected points
        regions = []
        for point in sorted(selected_points):
            start = max(0, point - window // 2)
            end = min(len(temperature) - 1, point + window // 2)
            regions.append((float(temperature[start]), float(temperature[end])))

        return regions

    def _calculate_quality_metrics(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict:
        """Calculate quality metrics for baseline fit."""
        metrics = {}

        # Calculate residuals in baseline regions
        if regions:
            residuals = []
            for start_temp, end_temp in regions:
                mask = (temperature >= start_temp) & (temperature <= end_temp)
                residuals.extend(heat_flow[mask] - baseline[mask])

            metrics["baseline_rmse"] = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
            metrics["baseline_max_deviation"] = float(np.max(np.abs(residuals)))

        # Calculate overall metrics
        metrics["total_correction"] = float(np.sum(np.abs(heat_flow - baseline)))
        metrics["smoothness"] = float(np.mean(np.abs(np.diff(baseline, 2))))

        return metrics

    def _evaluate_baseline_quality(self, result: BaselineResult) -> float:
        """Evaluate overall quality of baseline correction."""
        metrics = result.quality_metrics

        # Combine metrics into single score (lower is better)
        score = (
            metrics.get("baseline_rmse", 0)
            + 0.1 * metrics.get("smoothness", 0)
            + 0.01 * metrics.get("total_correction", 0)
        )

        return float(score)

    def _validate_data(
        self, temperature: NDArray[np.float64], heat_flow: NDArray[np.float64]
    ) -> None:
        """Validate input data arrays."""
        if len(temperature) != len(heat_flow):
            raise ValueError("Temperature and heat flow arrays must have same length")

        if len(temperature) < self.smoothing_window:
            raise ValueError(
                f"Data length must be at least {self.smoothing_window} points"
            )
