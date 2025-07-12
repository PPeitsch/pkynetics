"""Baseline correction methods for DSC data."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize, signal
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull

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

        self.methods = {
            "linear": self._fit_linear_baseline,
            "polynomial": self._fit_polynomial_baseline,
            "spline": self._fit_spline_baseline,
            "asymmetric": self._fit_asymmetric_baseline,
            "rubberband": self._fit_rubberband_baseline,
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
        """
        if method != "auto" and method not in self.methods:
            raise ValueError(f"Unknown baseline method: {method}")

        self._validate_data(temperature, heat_flow)

        heat_flow_smooth = signal.savgol_filter(
            heat_flow, self.smoothing_window, self.smoothing_order
        )

        if method == "auto":
            baseline_result = self._auto_baseline(
                temperature, heat_flow_smooth, regions, **kwargs
            )
            # The auto method already returns a complete BaselineResult
            return baseline_result

        correction_func = self.methods[method]
        baseline, params = correction_func(
            temperature, heat_flow_smooth, regions, **kwargs
        )

        corrected_data = heat_flow - baseline

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
        """
        regions = self._find_quiet_regions(temperature, heat_flow, n_regions)
        return self.correct(temperature, heat_flow, method, regions)

    def _auto_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> BaselineResult:
        """Automatically select best baseline method and return its result."""
        best_score = float("inf")
        best_result: Optional[BaselineResult] = None

        # Exclude rubberband as it is often unstable for auto-selection
        methods_to_try = [m for m in self.methods if m != "rubberband"]

        for method_name in methods_to_try:
            try:
                result = self.correct(
                    temperature, heat_flow, method_name, regions, **kwargs
                )
                score = self._evaluate_baseline_quality(result)
                if score < best_score:
                    best_score = score
                    best_result = result
            except ValueError:
                continue

        if best_result is None:
            raise ValueError("Could not determine a valid automatic baseline.")

        # Update method to 'auto' but keep the parameters from the best method
        best_result.method = "auto"
        best_result.parameters["best_method"] = best_result.parameters.get(
            "method", "unknown"
        )
        return best_result

    def _fit_linear_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        temp_points, heat_points = self._get_points_in_regions(
            temperature, heat_flow, regions
        )
        if len(temp_points) < 2:
            raise ValueError("Insufficient data points in regions for linear fit.")
        coeffs = np.polyfit(temp_points, heat_points, 1)
        return np.polyval(coeffs, temperature), {
            "slope": coeffs[0],
            "intercept": coeffs[1],
        }

    def _fit_polynomial_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions,
        degree=3,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        temp_points, heat_points = self._get_points_in_regions(
            temperature, heat_flow, regions
        )
        if len(temp_points) <= degree:
            raise ValueError(
                f"Insufficient data points for polynomial degree {degree}."
            )
        coeffs = np.polyfit(temp_points, heat_points, degree)
        return np.polyval(coeffs, temperature), {
            "coefficients": coeffs.tolist(),
            "degree": degree,
        }

    def _fit_spline_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions,
        smoothing=1.0,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        temp_points, heat_points = self._get_points_in_regions(
            temperature, heat_flow, regions
        )
        if len(temp_points) < 4:
            raise ValueError("Insufficient data points for spline fit.")
        spline = UnivariateSpline(temp_points, heat_points, s=smoothing, k=3)
        return spline(temperature), {
            "smoothing": smoothing,
            "n_knots": len(spline.get_knots()),
        }

    def _fit_asymmetric_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions,
        lam=1e5,
        p=0.001,
        niter=10,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        L = len(heat_flow)
        D = np.diff(np.eye(L), 2)
        w = np.ones(L)
        D_T_D = D.T @ D
        for _ in range(niter):
            W = np.diag(w)
            try:
                C = linalg.cholesky(W + lam * D_T_D)
                z = linalg.solve_triangular(
                    C, linalg.solve_triangular(C.T, w * heat_flow, lower=True)
                )
                w = p * (heat_flow > z) + (1 - p) * (heat_flow < z)
            except linalg.LinAlgError:
                # Fallback to a simpler method if Cholesky fails
                z = np.linalg.solve(W + lam * D_T_D, w * heat_flow)
                w = p * (heat_flow > z) + (1 - p) * (heat_flow < z)
        return z, {"lambda": lam, "p": p}

    def _fit_rubberband_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        points = np.column_stack((temperature, heat_flow))
        if len(points) < 3:
            raise ValueError("Insufficient data for Convex Hull.")
        hull = ConvexHull(points)
        lower_hull_indices = hull.vertices
        # A robust way to get the lower envelope
        lower_hull = [points[lower_hull_indices[0]]]
        for i in range(1, len(lower_hull_indices)):
            p1 = points[lower_hull_indices[i - 1]]
            p2 = points[lower_hull_indices[i]]
            if p2[0] > p1[0]:
                # Keep points that form the lower part of the hull
                if (
                    len(lower_hull) < 2
                    or np.cross(p2 - lower_hull[-2], p1 - lower_hull[-2]) > 0
                ):
                    lower_hull.append(p2)
        lower_hull = np.array(sorted(lower_hull, key=lambda p: p[0]))
        baseline = np.interp(temperature, lower_hull[:, 0], lower_hull[:, 1])
        return baseline, {"n_hull_points": len(lower_hull)}

    @staticmethod
    def _get_points_in_regions(temperature, heat_flow, regions):
        if regions is None:
            n_points = max(2, len(temperature) // 10)
            regions = [
                (temperature[0], temperature[n_points - 1]),
                (temperature[-n_points], temperature[-1]),
            ]
        mask = np.zeros_like(temperature, dtype=bool)
        for start_temp, end_temp in regions:
            mask |= (temperature >= start_temp) & (temperature <= end_temp)
        return temperature[mask], heat_flow[mask]

    def _find_quiet_regions(self, temperature, heat_flow, n_regions=4, window=20):
        if len(heat_flow) <= window:
            return [(temperature[0], temperature[-1])]
        rolling_var = np.array(
            [
                np.var(heat_flow[i : i + window])
                for i in range(len(heat_flow) - window + 1)
            ]
        )
        rolling_var = np.pad(
            rolling_var, (window // 2, window - 1 - window // 2), mode="edge"
        )
        var_minima = signal.argrelmin(rolling_var, order=window)[0]
        if len(var_minima) == 0:
            return [(temperature[0], temperature[-1])]
        sorted_minima = var_minima[np.argsort(rolling_var[var_minima])][:n_regions]
        regions = []
        for point in sorted(sorted_minima):
            start = max(0, point - window // 2)
            end = min(len(temperature) - 1, point + window // 2)
            if start < end:
                regions.append((temperature[start], temperature[end]))
        return regions if regions else [(temperature[0], temperature[-1])]

    def _calculate_quality_metrics(self, temperature, heat_flow, baseline, regions):
        metrics = {}
        temp_points, heat_points_in_region = self._get_points_in_regions(
            temperature, heat_flow, regions
        )
        _, baseline_in_region = self._get_points_in_regions(
            temperature, baseline, regions
        )
        if len(temp_points) > 0:
            residuals = heat_points_in_region - baseline_in_region
            metrics["baseline_rmse"] = float(np.sqrt(np.mean(residuals**2)))
            metrics["baseline_max_deviation"] = float(np.max(np.abs(residuals)))
        metrics["total_correction"] = float(np.sum(np.abs(heat_flow - baseline)))
        if len(baseline) > 2:
            metrics["smoothness"] = float(np.mean(np.abs(np.diff(baseline, 2))))
        return metrics

    def _evaluate_baseline_quality(self, result):
        metrics = result.quality_metrics
        score = metrics.get("baseline_rmse", 1e6) + 0.1 * metrics.get("smoothness", 1e6)
        return float(score)

    def _validate_data(self, temperature, heat_flow):
        if len(temperature) != len(heat_flow):
            raise ValueError(
                "Temperature and heat flow arrays must have the same length."
            )
        if len(temperature) < self.smoothing_window:
            raise ValueError(
                f"Data length ({len(temperature)}) must be at least the smoothing window size ({self.smoothing_window})."
            )
