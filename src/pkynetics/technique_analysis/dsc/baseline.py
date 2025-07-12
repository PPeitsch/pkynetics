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

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            method: Correction method to use
            regions: Optional list of (start_temp, end_temp) for baseline regions
            **kwargs: Additional parameters for specific correction methods

        Returns:
            BaselineResult object containing correction results
        """
        if method != "auto" and method not in self.methods:
            raise ValueError(f"Unknown baseline method: {method}")

        self._validate_data(temperature, heat_flow)

        heat_flow_smooth = signal.savgol_filter(
            heat_flow, self.smoothing_window, self.smoothing_order
        )

        if method == "auto":
            return self._auto_baseline(temperature, heat_flow_smooth, regions, **kwargs)

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

        for method_name in self.methods:
            try:
                result = self.correct(
                    temperature, heat_flow, method_name, regions, **kwargs
                )
                score = self._evaluate_baseline_quality(result)

                if score < best_score:
                    best_score = score
                    best_result = result
            except (ValueError, np.linalg.LinAlgError):
                # This method failed, continue to the next one
                continue

        if best_result is None:
            raise ValueError("Could not determine a valid automatic baseline.")

        # Return the complete result from the best method
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
            # Use first and last 10% of data as default regions
            n_points = len(temperature) // 10
            regions = [
                (float(temperature[0]), float(temperature[n_points])),
                (float(temperature[-n_points]), float(temperature[-1])),
            ]

        temp_points, heat_points = self._get_points_in_regions(
            temperature, heat_flow, regions
        )

        if len(temp_points) < 2:
            raise ValueError("Not enough points in specified regions for linear fit.")

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
                (float(temperature[0]), float(temperature[n_points])),
                (float(temperature[-n_points]), float(temperature[-1])),
            ]

        temp_points, heat_points = self._get_points_in_regions(
            temperature, heat_flow, regions
        )

        if len(temp_points) <= degree:
            raise ValueError(
                f"Not enough points in regions ({len(temp_points)}) for polynomial degree {degree}"
            )

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

        temp_points, heat_points = self._get_points_in_regions(
            temperature, heat_flow, regions
        )

        if len(temp_points) < 4:  # Spline needs at least k+1 points, k=3 default
            raise ValueError("Not enough points for spline fit.")

        spline = UnivariateSpline(temp_points, heat_points, s=smoothing, k=3)
        baseline = spline(temperature)

        params = {"smoothing": smoothing, "n_knots": len(spline.get_knots())}
        return baseline, params

    def _fit_asymmetric_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        lam: float = 1e5,
        p: float = 0.001,
        niter: int = 10,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Fit asymmetric least squares baseline."""
        L = len(heat_flow)
        D = signal.cspline2d(np.ones(L), 8.0)
        D = np.diff(np.diff(np.eye(L)))
        w = np.ones(L)

        # Cholesky decomposition for stability and speed
        D_T_D = D.T @ D
        for i in range(niter):
            W = np.diag(w)
            C = linalg.cholesky(W + lam * D_T_D)
            z = linalg.solve_triangular(
                C, linalg.solve_triangular(C.T, w * heat_flow, lower=True)
            )
            w = p * (heat_flow > z) + (1 - p) * (heat_flow < z)

        return z, {"lambda": lam, "p": p}

    def _fit_rubberband_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Fit rubberband baseline using convex hull."""
        points = np.column_stack((temperature, heat_flow))

        if len(points) < 3:
            raise ValueError("Not enough points for Convex Hull.")

        try:
            hull = ConvexHull(points)
        except Exception as e:
            raise ValueError(f"Could not compute Convex Hull: {e}")

        # The lower hull is the set of vertices from the start to the end point
        # of the hull, in increasing order of x-coordinates.
        hull_points = points[hull.vertices]

        # Sort hull vertices by temperature
        sorted_hull_points = hull_points[np.argsort(hull_points[:, 0])]

        # Find the indices of the points with min and max temperature
        min_temp_idx = np.argmin(sorted_hull_points[:, 0])
        max_temp_idx = np.argmax(sorted_hull_points[:, 0])

        # Walk along the hull from min_temp_idx to max_temp_idx
        lower_hull_indices = []
        current_idx = min_temp_idx
        while True:
            lower_hull_indices.append(current_idx)
            if current_idx == max_temp_idx:
                break
            current_idx = (current_idx + 1) % len(sorted_hull_points)

        lower_hull = sorted_hull_points[lower_hull_indices]

        baseline = np.interp(temperature, lower_hull[:, 0], lower_hull[:, 1])

        return baseline, {"n_hull_points": len(lower_hull)}

    @staticmethod
    def _get_points_in_regions(
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: List[Tuple[float, float]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Collects all data points within the specified list of regions."""
        mask = np.zeros_like(temperature, dtype=bool)
        for start_temp, end_temp in regions:
            mask |= (temperature >= float(start_temp)) & (
                temperature <= float(end_temp)
            )
        return temperature[mask], heat_flow[mask]

    def _find_quiet_regions(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        n_regions: int = 4,
        window: int = 20,
    ) -> List[Tuple[float, float]]:
        """Find quiet (low variance) regions in the data."""
        if len(heat_flow) < window:
            return [(temperature[0], temperature[-1])]

        rolling_var = np.array(
            [
                np.var(heat_flow[i : i + window])
                for i in range(len(heat_flow) - window + 1)
            ]
        )
        # Pad rolling_var to match temperature length for indexing
        rolling_var = np.pad(
            rolling_var, (window // 2, window - 1 - window // 2), mode="edge"
        )

        var_minima_indices = signal.argrelmin(rolling_var, order=window)[0]
        if len(var_minima_indices) == 0:
            return [(temperature[0], temperature[-1])]

        sorted_minima = var_minima_indices[np.argsort(rolling_var[var_minima_indices])]
        selected_points = sorted_minima[:n_regions]

        regions = []
        for point in sorted(selected_points):
            start_idx = max(0, point - window // 2)
            end_idx = min(len(temperature) - 1, point + window // 2)
            if start_idx < end_idx:
                regions.append(
                    (float(temperature[start_idx]), float(temperature[end_idx]))
                )
        return regions if regions else [(temperature[0], temperature[-1])]

    def _calculate_quality_metrics(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict:
        """Calculate quality metrics for baseline fit."""
        metrics = {}
        if regions:
            temp_points, heat_points = self._get_points_in_regions(
                temperature, heat_flow, regions
            )
            _, baseline_points = self._get_points_in_regions(
                temperature, baseline, regions
            )
            if len(temp_points) > 0:
                residuals = heat_points - baseline_points
                metrics["baseline_rmse"] = float(np.sqrt(np.mean(residuals**2)))
                metrics["baseline_max_deviation"] = float(np.max(np.abs(residuals)))

        metrics["total_correction"] = float(np.sum(np.abs(heat_flow - baseline)))
        if len(baseline) > 2:
            metrics["smoothness"] = float(np.mean(np.abs(np.diff(baseline, 2))))
        else:
            metrics["smoothness"] = 0.0
        return metrics

    def _evaluate_baseline_quality(self, result: BaselineResult) -> float:
        """Evaluate overall quality of baseline correction."""
        metrics = result.quality_metrics
        score = (
            metrics.get("baseline_rmse", 1e6)
            + 0.1 * metrics.get("smoothness", 1e6)
            + 0.01 * metrics.get("total_correction", 1e6)
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
                f"Data length ({len(temperature)}) must be at least the smoothing window size ({self.smoothing_window})"
            )
