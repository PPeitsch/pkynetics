"""Peak analysis implementation for DSC data."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, signal
from scipy.integrate import trapezoid

from .types import DSCPeak
from .utilities import find_intersection_point, safe_savgol_filter, validate_window_size

logger = logging.getLogger(__name__)


class PeakAnalyzer:
    """Class for DSC peak analysis."""

    def __init__(
        self,
        smoothing_window: int = 21,
        smoothing_order: int = 3,
        peak_prominence: float = 0.1,
        height_threshold: float = 0.05,
    ):
        """
        Initialize peak analyzer.

        Args:
            smoothing_window: Window size for Savitzky-Golay smoothing
            smoothing_order: Order of polynomial for smoothing
            peak_prominence: Minimum prominence for peak detection
            height_threshold: Minimum height threshold for peak detection
        """
        self.smoothing_window = smoothing_window
        self.smoothing_order = smoothing_order
        self.peak_prominence = peak_prominence
        self.height_threshold = height_threshold
        self.limit_noise_factor = 3.0

    def find_peaks(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
        heating_rate: Optional[float] = None,
        sample_mass: Optional[float] = None,
    ) -> List[DSCPeak]:
        """
        Find and analyze (upward) peaks in DSC data.

        Args:
            temperature: Temperature array in K
            heat_flow: Heat flow array in mW
            baseline: Optional baseline array in mW
            heating_rate: Heating rate in K/min, needed for the enthalpy
            sample_mass: Sample mass in mg, needed for the enthalpy

        Returns:
            List of DSCPeak objects containing peak information. The enthalpy
            is NaN unless heating_rate and sample_mass are given.

        Raises:
            ValueError: If the arrays are empty or have different lengths
        """
        if len(temperature) != len(heat_flow):
            raise ValueError("Temperature and heat flow arrays must have same length")
        if len(temperature) < 3:
            raise ValueError("At least 3 data points are required")

        # Apply signal smoothing with safe window size
        smooth_heat_flow = safe_savgol_filter(
            heat_flow, self.smoothing_window, self.smoothing_order
        )

        # Apply baseline correction if provided
        signal_to_analyze = smooth_heat_flow.copy()
        if baseline is not None:
            signal_to_analyze = smooth_heat_flow - baseline

        # Find peaks with enhanced criteria
        peaks, properties = signal.find_peaks(
            signal_to_analyze,
            prominence=self.peak_prominence,
            height=self.height_threshold,
            width=validate_window_size(len(signal_to_analyze), self.smoothing_window)
            // 2,
            distance=validate_window_size(
                len(signal_to_analyze), self.smoothing_window
            ),
        )

        limits = self.integration_limits(
            signal_to_analyze,
            peaks,
            properties,
            self.noise_level(heat_flow if baseline is None else heat_flow - baseline),
        )

        peak_list = []
        for peak_idx, (lo, hi) in zip(peaks, limits):
            if hi - lo < 3:
                continue
            peak_info = self.analyze_peak_region(
                temperature[lo:hi],
                heat_flow[lo:hi],
                int(peak_idx) - lo,
                baseline[lo:hi] if baseline is not None else None,
                heating_rate=heating_rate,
                sample_mass=sample_mass,
            )
            peak_info.peak_indices = (lo, hi - 1)
            peak_list.append(peak_info)

        return peak_list

    def noise_level(self, data: NDArray[np.float64]) -> float:
        """Estimate noise as the residual of Savitzky-Golay smoothing."""
        smooth = safe_savgol_filter(data, self.smoothing_window, self.smoothing_order)
        return float(np.std(data - smooth))

    def integration_limits(
        self,
        smooth: NDArray[np.float64],
        peak_indices: NDArray[np.intp],
        properties: Dict[str, Any],
        noise: float,
    ) -> List[Tuple[int, int]]:
        """
        Integration limits (start, stop) of peaks found by scipy find_peaks.

        The limits are where the (smoothed) peak has decayed to 0.1% of its
        prominence above its reference base, but not below
        limit_noise_factor times the noise remaining after smoothing, where
        the limits would wander. Unlike the find_peaks bases, they do not
        extend into neighbouring events.

        Args:
            smooth: Smoothed signal the peaks were found in
            peak_indices: Peak indices from find_peaks
            properties: find_peaks properties (with prominences)
            noise: Noise level of the unsmoothed signal

        Returns:
            List of (start, stop) index pairs, stop exclusive
        """
        window = validate_window_size(len(smooth), self.smoothing_window)
        coeffs = signal.savgol_coeffs(window, min(self.smoothing_order, window - 1))
        smooth_noise = noise * float(np.sqrt(np.sum(coeffs**2)))

        limits = []
        for j, peak_idx in enumerate(peak_indices):
            prom = float(properties["prominences"][j])
            level = max(1e-3 * prom, self.limit_noise_factor * smooth_noise)
            _, _, left, right = signal.peak_widths(
                smooth,
                [peak_idx],
                rel_height=1 - min(level / prom, 0.5),
                prominence_data=(
                    properties["prominences"][j : j + 1],
                    properties["left_bases"][j : j + 1],
                    properties["right_bases"][j : j + 1],
                ),
            )
            limits.append((int(np.floor(left[0])), int(np.ceil(right[0])) + 1))
        return limits

    def _calculate_onset(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """
        Calculate the extrapolated onset temperature (ISO 11357-1).

        Intersection of the baseline with the tangent at the inflection point
        (steepest slope) of the leading edge.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peak_idx: Index of peak maximum
            baseline: Optional baseline array (zero if not given)

        Returns:
            Onset temperature
        """
        return self._extrapolated_edge(
            temperature, heat_flow, peak_idx, baseline, leading=True
        )

    def _calculate_endset(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """
        Calculate the extrapolated endset temperature (ISO 11357-1).

        Intersection of the baseline with the tangent at the inflection point
        (steepest slope) of the trailing edge.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peak_idx: Index of peak maximum
            baseline: Optional baseline array (zero if not given)

        Returns:
            Endset temperature
        """
        return self._extrapolated_edge(
            temperature, heat_flow, peak_idx, baseline, leading=False
        )

    def _extrapolated_edge(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
        baseline: Optional[NDArray[np.float64]],
        leading: bool,
    ) -> float:
        """Tangent at the steepest point of one peak edge, intersected with
        the baseline. Works on the baseline-corrected signal, so the
        intersection is with zero and may be extrapolated beyond the data."""
        self._validate_peak_index(peak_idx, len(temperature))

        corrected = heat_flow if baseline is None else heat_flow - baseline
        smooth = safe_savgol_filter(
            corrected, self.smoothing_window, self.smoothing_order
        )
        dydx = np.gradient(smooth, temperature)

        # Steepest rise before the maximum, or steepest fall after it
        if leading:
            if peak_idx < 1:
                return float(temperature[peak_idx])
            edge_idx = int(np.argmax(dydx[:peak_idx]))
        else:
            if peak_idx >= len(temperature) - 1:
                return float(temperature[peak_idx])
            edge_idx = peak_idx + int(np.argmin(dydx[peak_idx:]))

        slope = dydx[edge_idx]
        if abs(slope) < 1e-12:
            return float(temperature[edge_idx])

        return float(temperature[edge_idx] - smooth[edge_idx] / slope)

    def _calculate_peak_width(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """
        Calculate peak width at half height.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peak_idx: Index of peak maximum
            baseline: Optional baseline array

        Returns:
            Peak width at half height
        """
        if baseline is None:
            baseline = np.zeros_like(heat_flow)

        # Calculate peak height from baseline
        peak_height = heat_flow[peak_idx] - baseline[peak_idx]
        half_height = peak_height / 2 + baseline[peak_idx]

        # Find intersection points at half height
        left_temp, _ = find_intersection_point(
            temperature,
            heat_flow,
            np.full_like(heat_flow, half_height),
            peak_idx,
            "backward",
        )

        right_temp, _ = find_intersection_point(
            temperature,
            heat_flow,
            np.full_like(heat_flow, half_height),
            peak_idx,
            "forward",
        )

        return right_temp - left_temp

    def deconvolute_peaks(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        n_peaks: int,
        peak_shape: str = "gaussian",
    ) -> Tuple[List[Dict], NDArray[np.float64]]:
        """
        Deconvolute overlapping peaks.

        Initial guesses come from local maxima, then from second-derivative
        minima (shoulders). Peaks closer than about their half width at half
        maximum cannot be separated reliably.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            n_peaks: Number of peaks to fit
            peak_shape: Peak function type ("gaussian" or "lorentzian")

        Returns:
            Tuple of (list of peak parameters, fitted curve)
        """

        def gaussian(
            x: NDArray[np.float64], amp: float, cen: float, wid: float
        ) -> NDArray[np.float64]:
            result: NDArray[np.float64] = amp * np.exp(-(((x - cen) / wid) ** 2))
            return result

        def lorentzian(
            x: NDArray[np.float64], amp: float, cen: float, wid: float
        ) -> NDArray[np.float64]:
            return amp * wid**2 / ((x - cen) ** 2 + wid**2)

        peak_func = gaussian if peak_shape == "gaussian" else lorentzian

        # Apply smoothing for better peak detection
        smooth_flow = safe_savgol_filter(
            heat_flow, validate_window_size(len(heat_flow), self.smoothing_window), 3
        )

        # Find all potential peaks; keep the most prominent n_peaks
        peaks, properties = signal.find_peaks(
            smooth_flow, prominence=np.max(smooth_flow) * 0.05
        )
        order = np.argsort(properties["prominences"])[::-1][:n_peaks]
        peak_indices = [int(i) for i in peaks[order]]

        if len(peak_indices) < n_peaks:
            # Hidden shoulders have no maximum of their own but show up as
            # minima of the second derivative
            # (from a more strongly smoothed signal: d2 amplifies noise),
            # restricted to where the signal is significant
            d2_smooth = safe_savgol_filter(heat_flow, max(len(heat_flow) // 20, 5), 3)
            d2 = np.gradient(np.gradient(d2_smooth, temperature), temperature)
            d2_min, d2_props = signal.find_peaks(-d2, height=0)
            d2_min = d2_min[d2_smooth[d2_min] > 0.1 * np.max(d2_smooth)]
            d2_props["peak_heights"] = -d2[d2_min]
            min_gap = validate_window_size(len(temperature), self.smoothing_window)
            for idx in d2_min[np.argsort(d2_props["peak_heights"])[::-1]]:
                if len(peak_indices) == n_peaks:
                    break
                if all(abs(int(idx) - j) > min_gap for j in peak_indices):
                    peak_indices.append(int(idx))

        if len(peak_indices) < n_peaks:
            # Last resort: evenly spaced guesses
            temp_min = temperature.min()
            region_size = (temperature.max() - temp_min) / n_peaks
            for i in range(n_peaks - len(peak_indices)):
                pos = temp_min + region_size * (i + 0.5)
                peak_indices.append(int(np.abs(temperature - pos).argmin()))

        # Generate initial parameters and bounds
        p0: List[float] = []
        bounds_low: List[float] = []
        bounds_high: List[float] = []

        temp_range = temperature.max() - temperature.min()
        min_width = temp_range * 0.005
        max_width = temp_range * 0.5
        max_amp = 2 * float(np.max(np.abs(smooth_flow)))

        for idx in peak_indices:
            amp = float(np.clip(smooth_flow[idx], max_amp * 1e-3, max_amp / 2))
            cen = float(temperature[idx])
            wid = temp_range * 0.05  # Initial width 5% of temperature range

            p0.extend([amp, cen, wid])
            bounds_low.extend([0.0, cen - temp_range * 0.15, min_width])
            bounds_high.extend([max_amp, cen + temp_range * 0.15, max_width])

        def fit_function(x: NDArray[np.float64], *params: Any) -> NDArray[np.float64]:
            result = np.zeros_like(x)
            for i in range(0, len(params), 3):
                result += peak_func(x, params[i], params[i + 1], params[i + 2])
            return result

        try:
            # Perform curve fitting with improved bounds
            popt, _ = optimize.curve_fit(
                fit_function,
                temperature,
                heat_flow,
                p0=p0,
                bounds=(bounds_low, bounds_high),
                maxfev=10000,
                ftol=1e-8,
                xtol=1e-8,
            )

            peak_params = []
            fitted_curve = np.zeros_like(temperature)

            for i in range(0, len(popt), 3):
                params = {
                    "amplitude": float(popt[i]),
                    "center": float(popt[i + 1]),
                    "width": float(popt[i + 2]),
                    "area": float(
                        trapezoid(
                            peak_func(temperature, popt[i], popt[i + 1], popt[i + 2]),
                            temperature,
                        )
                    ),
                }
                peak_params.append(params)
                fitted_curve += peak_func(temperature, *popt[i : i + 3])

            return peak_params, fitted_curve

        except RuntimeError as e:
            logger.warning(f"Peak deconvolution did not converge: {e}")
            return [], np.zeros_like(temperature)

    def _validate_peak_index(self, peak_idx: int, array_length: int) -> None:
        """
        Validate peak index is within array bounds.

        Args:
            peak_idx: Peak index to validate
            array_length: Length of data array

        Raises:
            IndexError: If peak_idx is out of bounds
        """
        if peak_idx < 0 or peak_idx >= array_length:
            raise IndexError(
                f"Peak index {peak_idx} out of bounds for array of length {array_length}"
            )

    def analyze_peak_region(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
        baseline: Optional[NDArray[np.float64]] = None,
        heating_rate: Optional[float] = None,
        sample_mass: Optional[float] = None,
    ) -> DSCPeak:
        """
        Characterize a single peak.

        The arrays should span the whole peak (from where the signal leaves
        the baseline to where it returns); the area is integrated over them.

        Args:
            temperature: Temperature array in K
            heat_flow: Heat flow array in mW
            peak_idx: Index of peak maximum
            baseline: Optional baseline array (zero if not given)
            heating_rate: Heating rate in K/min, needed for the enthalpy
            sample_mass: Sample mass in mg, needed for the enthalpy

        Returns:
            DSCPeak with onset/endset (extrapolated), height, FWHM, area in
            mW*K and enthalpy in J/g (NaN without heating rate and mass)
        """
        onset_temp = self._calculate_onset(temperature, heat_flow, peak_idx, baseline)
        endset_temp = self._calculate_endset(temperature, heat_flow, peak_idx, baseline)

        heat_flow_corr = heat_flow if baseline is None else heat_flow - baseline

        peak_height = float(heat_flow_corr[peak_idx])
        peak_width = self._calculate_peak_width(
            temperature, heat_flow, peak_idx, baseline
        )

        peak_area = float(trapezoid(heat_flow_corr, temperature))

        # dH = integral(q dT) / beta / m: mW*K / (K/s) = mJ, mJ / mg = J/g
        if heating_rate and sample_mass:
            # Magnitude: on cooling both the area and beta are negative
            enthalpy = abs(peak_area / (heating_rate / 60)) / sample_mass
        else:
            enthalpy = float("nan")

        return DSCPeak(
            onset_temperature=float(onset_temp),
            peak_temperature=float(temperature[peak_idx]),
            endset_temperature=float(endset_temp),
            enthalpy=float(enthalpy),
            peak_height=peak_height,
            peak_width=float(peak_width),
            peak_area=peak_area,
            baseline_type="none" if baseline is None else "provided",
            baseline_params={},
            peak_indices=(0, len(temperature) - 1),
        )
