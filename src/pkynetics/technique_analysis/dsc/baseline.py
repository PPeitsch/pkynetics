"""Baseline correction methods for DSC data."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal, sparse
from scipy.interpolate import UnivariateSpline
from scipy.sparse.linalg import spsolve

from .types import BaselineResult
from .utilities import safe_savgol_filter


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
        step_regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> BaselineResult:
        """
        Apply baseline correction with specified method.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            method: Correction method to use
            regions: Optional list of (start_temp, end_temp) for baseline regions
            step_regions: Optional list of (start_temp, end_temp) where the heat
                flow steps to a new level, such as a glass transition. The
                baseline is then fitted on each side separately and joined
                across the step, instead of one fit crossing it
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
        if step_regions:
            baseline, params = self._fit_stepped_baseline(
                temperature, heat_flow_smooth, method, regions, step_regions, **kwargs
            )
            method = params["method"]
        else:
            correction_func = self.methods[method]
            baseline, params = correction_func(
                temperature, heat_flow_smooth, regions, **kwargs
            )

            # Report the method actually used when selected automatically
            if method == "auto":
                method = params.pop("method")

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

        if not regions:
            raise ValueError("No quiet regions found for baseline optimization")

        # Try different region combinations
        best_result: Optional[BaselineResult] = None
        best_score = float("inf")

        for i in range(min(10, len(regions))):  # Limit number of combinations
            test_regions = regions[i : i + n_regions]
            result = self.correct(temperature, heat_flow, method, test_regions)
            score = self._evaluate_baseline_quality(result)

            if score < best_score:
                best_score = score
                best_result = result

        assert best_result is not None
        return best_result

    def _fit_linear_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """
        Fit linear baseline through specified regions.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            regions: List of (start_temp, end_temp) tuples for baseline regions
            **kwargs: Additional parameters

        Returns:
            Tuple[NDArray[np.float64], Dict]: Baseline array and parameters
        """
        if regions is None:
            regions = self._default_regions(temperature)

        temp_points, heat_points = self._collect_region_points(
            temperature, heat_flow, regions
        )
        if len(temp_points) < 2:
            raise ValueError("Not enough points in baseline regions for linear fit")

        coeffs = np.polyfit(temp_points, heat_points, 1)
        baseline = np.asarray(np.polyval(coeffs, temperature), dtype=np.float64)

        params = {"slope": float(coeffs[0]), "intercept": float(coeffs[1])}
        return baseline, params

    def _fit_polynomial_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        degree: int = 3,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """
        Fit polynomial baseline of specified degree.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            regions: List of (start_temp, end_temp) tuples for baseline regions
            degree: Polynomial degree
            **kwargs: Additional parameters

        Returns:
            Tuple[NDArray[np.float64], Dict]: Baseline array and parameters
        """
        if regions is None:
            regions = self._default_regions(temperature)

        temp_points, heat_points = self._collect_region_points(
            temperature, heat_flow, regions
        )

        # Fit polynomial
        if len(temp_points) <= degree:
            raise ValueError(f"Not enough points for polynomial degree {degree}")

        coeffs = np.polyfit(temp_points, heat_points, degree)
        baseline = np.asarray(np.polyval(coeffs, temperature), dtype=np.float64)

        params = {"coefficients": coeffs.tolist(), "degree": degree}
        return baseline, params

    def _fit_spline_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        smoothing: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """
        Fit spline baseline with automatic knot selection.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            regions: List of (start_temp, end_temp) baseline regions
            smoothing: Smoothing factor **relative to the noise**, not an
                absolute residual budget. UnivariateSpline's ``s`` bounds the
                sum of squared residuals, so a fixed value means something
                different for every signal: for heat flow in mW the same
                ``s=1.0`` is a loose fit on a microwatt-level baseline and an
                interpolation on a milliwatt-level one, and it also tightens
                as points are added. Here ``s = smoothing * n * sigma**2``,
                with sigma estimated robustly from the point-to-point
                differences of the baseline points, which is the residual a
                correct fit is expected to leave. So ``smoothing=1.0`` means
                "follow the trend, not the noise" on any signal and any
                sampling rate; lower values fit tighter.
            **kwargs: Additional parameters

        Returns:
            Tuple[NDArray[np.float64], Dict]: Baseline array and parameters
        """
        if regions is None:
            regions = self._find_quiet_regions(temperature, heat_flow)

        temp_points, heat_points = self._collect_region_points(
            temperature, heat_flow, regions
        )

        # On noise-free (synthetic) data the estimate is 0, and s=0 forces
        # the spline to interpolate every point: it then adds knots to chase
        # the curvature the peak tails leave in the baseline points, and
        # swings between regions. Floor sigma at 1e-2 of the spread so clean
        # data gets the simplest smooth curve that fits, not an interpolant.
        sigma = np.sqrt(self._noise_variance(heat_points))
        sigma = max(sigma, 1e-2 * float(np.ptp(heat_points)))
        s = smoothing * len(heat_points) * sigma**2

        # Fit univariate spline
        spline = UnivariateSpline(temp_points, heat_points, s=s)
        baseline = spline(temperature)

        params = {
            "smoothing": smoothing,
            "s": s,
            "n_knots": len(spline.get_knots()),
        }

        return baseline, params

    @staticmethod
    def _noise_variance(values: NDArray[np.float64]) -> float:
        """
        Estimate the noise variance of a signal robustly.

        Uses the MAD of the successive differences: differencing removes any
        smooth trend, the median absolute deviation ignores the few large
        jumps an event or an outlier leaves, and the sqrt(2) divides out the
        variance the difference itself adds.
        """
        if len(values) < 2:
            return 0.0
        diffs = np.diff(values)
        sigma = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs)))) / np.sqrt(2)
        return float(sigma**2)

    def _fit_asymmetric_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """
        Fit asymmetric least squares baseline.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            regions: Optional baseline regions
            **kwargs: lam (smoothness, for 1000 points; default 1e7) and
                p (asymmetry; default 0.001, for upward peaks)

        Returns:
            Tuple[NDArray[np.float64], Dict]: Baseline and parameters
        """

        def als_baseline(
            y: NDArray[np.float64], lam: float, p: float, niter: int = 10
        ) -> NDArray[np.float64]:
            # Eilers & Boelens (2005), with sparse matrices: O(n) memory
            L = len(y)
            D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
            penalty = lam * D.dot(D.T)
            w = np.ones(L)
            z = y
            for _ in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                z = spsolve(sparse.csc_matrix(W + penalty), w * y)
                w = p * (y > z) + (1 - p) * (y < z)
            return np.asarray(z, dtype=np.float64)

        lam = kwargs.get("lam", 1e7)
        p = kwargs.get("p", 0.001)

        # The penalty acts on index differences; scaling by (N/1000)^4 keeps
        # the stiffness independent of the number of points
        lam_scaled = lam * (len(heat_flow) / 1000) ** 4
        baseline = als_baseline(heat_flow, lam_scaled, p)
        params = {"lambda": lam, "p": p}

        return baseline, params

    def _auto_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """
        Select the baseline model by BIC on event-free regions.

        Linear and polynomial (degree 2 and 3) baselines are fitted to the
        baseline regions (detected quiet regions if none are given) and the
        model with the lowest Bayesian information criterion is kept, so a
        higher degree is only chosen when it clearly improves the fit. Spline
        and asymmetric baselines are not candidates: they are not constrained
        to event-free regions and can absorb broad peaks.
        """
        if regions is None:
            regions = self._find_quiet_regions(temperature, heat_flow)

        temp_points, heat_points = self._collect_region_points(
            temperature, heat_flow, regions
        )
        n = len(temp_points)

        best_bic = float("inf")
        best: Optional[Tuple[NDArray[np.float64], Dict]] = None
        for degree in (1, 2, 3):
            n_params = degree + 1
            if n <= n_params + 1:
                break
            coeffs = np.polyfit(temp_points, heat_points, degree)
            rss = float(np.sum((heat_points - np.polyval(coeffs, temp_points)) ** 2))
            # Floor avoids log(0) on noise-free data
            rss = max(rss, n * float(np.finfo(float).eps))
            bic = n * np.log(rss / n) + n_params * np.log(n)
            if bic < best_bic:
                best_bic = bic
                baseline = np.asarray(np.polyval(coeffs, temperature), dtype=np.float64)
                if degree == 1:
                    params: Dict = {
                        "method": "linear",
                        "slope": float(coeffs[0]),
                        "intercept": float(coeffs[1]),
                    }
                else:
                    params = {
                        "method": "polynomial",
                        "coefficients": coeffs.tolist(),
                        "degree": degree,
                    }
                best = (baseline, params)

        if best is None:
            raise ValueError("Not enough points in baseline regions")

        return best

    def _fit_rubberband_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """Fit rubberband baseline using the lower convex hull.

        Assumes events point upwards (positive peaks); invert the heat flow
        to use it for downward peaks.
        """
        from scipy.spatial import ConvexHull

        points = np.column_stack((temperature, heat_flow))
        hull = ConvexHull(points)

        # 2D hull vertices are counterclockwise: walking from the leftmost
        # to the rightmost vertex traverses the lower hull
        vertices = hull.vertices
        start = int(np.argmin(points[vertices, 0]))
        vertices = np.roll(vertices, -start)
        end = int(np.argmax(points[vertices, 0]))
        lower_points = points[vertices[: end + 1]]

        # Interpolate baseline
        baseline = np.interp(temperature, lower_points[:, 0], lower_points[:, 1])

        return baseline, {"n_hull_points": len(lower_points)}

    def _fit_stepped_baseline(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        method: str,
        regions: Optional[List[Tuple[float, float]]],
        step_regions: List[Tuple[float, float]],
        **kwargs: Any,
    ) -> Tuple[NDArray[np.float64], Dict]:
        """
        Fit the baseline on each side of a step and join it across the step.

        A glass transition is a step in the heat flow, not a transient
        deviation: the sample stays at a new level afterwards. A single fit
        across it passes through the middle of the step, which leaves a
        residual bump where the transition was (detected as a spurious peak)
        and shifts the level of the real peaks on both sides. Each segment
        between steps is therefore fitted on its own, and inside the step the
        baseline interpolates linearly between the neighbouring fits.
        """
        segments = self._segments_between(temperature, step_regions)
        if not segments:
            raise ValueError("Step regions leave no data to fit the baseline on")

        correction_func = self.methods[method]
        baseline = np.full(len(temperature), np.nan, dtype=np.float64)
        segment_params: List[Dict] = []
        methods_used = []

        for low, high in segments:
            mask = (temperature >= low) & (temperature <= high)
            segment_regions = self._clip_regions(regions, low, high)
            segment_baseline, params = correction_func(
                temperature[mask], heat_flow[mask], segment_regions, **kwargs
            )
            baseline[mask] = segment_baseline
            methods_used.append(params.pop("method") if method == "auto" else method)
            segment_params.append(
                {"range": (float(low), float(high)), "parameters": params}
            )

        self._interpolate_gaps(temperature, baseline)

        used = sorted(set(methods_used))
        return baseline, {
            "method": f"stepped ({'+'.join(used)})",
            "base_method": method,
            "steps": [(float(low), float(high)) for low, high in step_regions],
            "segments": segment_params,
        }

    @staticmethod
    def _segments_between(
        temperature: NDArray[np.float64], step_regions: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Temperature ranges outside the steps, in order, ignoring empty ones."""
        steps = sorted(
            (min(float(a), float(b)), max(float(a), float(b))) for a, b in step_regions
        )
        segments = []
        start = float(np.min(temperature))
        end = float(np.max(temperature))
        for low, high in steps:
            if low > start:
                segments.append((start, min(low, end)))
            start = max(start, high)
        if start < end:
            segments.append((start, end))
        # A fit needs at least a few points; drop slivers
        return [
            (low, high)
            for low, high in segments
            if np.count_nonzero((temperature >= low) & (temperature <= high)) >= 10
        ]

    @staticmethod
    def _clip_regions(
        regions: Optional[List[Tuple[float, float]]], low: float, high: float
    ) -> Optional[List[Tuple[float, float]]]:
        """Regions restricted to a segment; None to let the method pick its own."""
        if regions is None:
            return None
        clipped = []
        for start, end in regions:
            region_low, region_high = sorted((float(start), float(end)))
            if region_high > low and region_low < high:
                clipped.append((max(region_low, low), min(region_high, high)))
        return clipped or None

    @staticmethod
    def _interpolate_gaps(
        temperature: NDArray[np.float64], baseline: NDArray[np.float64]
    ) -> None:
        """
        Fill the steps in place, interpolating between the fitted segments.

        There is always at least one fitted segment: _fit_stepped_baseline
        rejects step regions that leave no data to fit.
        """
        fitted = ~np.isnan(baseline)
        baseline[~fitted] = np.interp(
            temperature[~fitted], temperature[fitted], baseline[fitted]
        )

    @staticmethod
    def _default_regions(
        temperature: NDArray[np.float64], fraction: float = 0.1
    ) -> List[Tuple[float, float]]:
        """Baseline regions at both ends of the data (first/last fraction)."""
        n_points = max(2, int(len(temperature) * fraction))
        first, last = temperature[:n_points], temperature[-n_points:]
        return [
            (float(np.min(first)), float(np.max(first))),
            (float(np.min(last)), float(np.max(last))),
        ]

    @staticmethod
    def _collect_region_points(
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        regions: List[Tuple[float, float]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Select the data points falling inside any of the regions."""
        mask = np.zeros(len(temperature), dtype=bool)
        for start_temp, end_temp in regions:
            low, high = sorted((float(start_temp), float(end_temp)))
            mask |= (temperature >= low) & (temperature <= high)
        return temperature[mask], heat_flow[mask]

    def _find_quiet_regions(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        n_regions: int = 4,
        window: int = 20,
    ) -> List[Tuple[float, float]]:
        """
        Find event-free regions: low slope and low curvature.

        Low variance alone is not enough, since peak apexes are locally flat.
        Each point is scored by its first and second derivative, each
        normalized by its median magnitude, and the lowest-scoring windows
        are selected at least len/(2*n_regions) points apart.
        """
        window = min(window, len(heat_flow))
        smooth = safe_savgol_filter(
            heat_flow, self.smoothing_window, self.smoothing_order
        )
        d1 = np.gradient(smooth, temperature)
        d2 = np.gradient(d1, temperature)

        def _normalized(x: NDArray[np.float64]) -> NDArray[np.float64]:
            scale = float(np.median(np.abs(x)))
            return np.abs(x) / scale if scale > 0 else np.abs(x)

        activity = _normalized(d1) + _normalized(d2)
        score = np.convolve(activity, np.ones(window) / window, mode="valid")

        min_separation = max(window, len(heat_flow) // (2 * n_regions))
        selected: List[int] = []
        for idx in np.argsort(score):
            if all(abs(int(idx) - j) >= min_separation for j in selected):
                selected.append(int(idx))
            if len(selected) == n_regions:
                break

        regions: List[Tuple[float, float]] = []
        for i in sorted(selected):
            first, last = float(temperature[i]), float(temperature[i + window - 1])
            regions.append((min(first, last), max(first, last)))
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
            residuals: List[float] = []
            for start_temp, end_temp in regions:
                low, high = sorted((float(start_temp), float(end_temp)))
                mask = (temperature >= low) & (temperature <= high)
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

        # Fit in baseline regions plus a smoothness penalty (lower is better).
        # total_correction is deliberately excluded: it rewards baselines that
        # follow the data through the peaks.
        score = metrics.get("baseline_rmse", 0) + 0.1 * metrics.get("smoothness", 0)

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
