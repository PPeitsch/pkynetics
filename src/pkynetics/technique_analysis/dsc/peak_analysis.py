"""Peak analysis implementation for DSC data."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, signal
from scipy.integrate import trapz

from .types import DSCPeak
from .utilities import find_intersection_point, safe_savgol_filter, validate_window_size


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

    def find_peaks(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> List[DSCPeak]:
        """
        Find and analyze peaks in DSC data.

        Args:
            temperature: Temperature array in K
            heat_flow: Heat flow array in mW
            baseline: Optional baseline array in mW

        Returns:
            List of DSCPeak objects containing peak information

        Raises:
            ValueError: If temperature and heat flow arrays have different lengths
        """
        if len(temperature) != len(heat_flow):
            raise ValueError("Temperature and heat flow arrays must have same length")

        # Apply signal smoothing with safe window size
        smooth_heat_flow = safe_savgol_filter(
            heat_flow, self.smoothing_window, self.smoothing_order
        )

        # Apply baseline correction if provided
        signal_to_analyze = smooth_heat_flow.copy()
        if baseline is not None:
            if len(baseline) != len(signal_to_analyze):
                raise ValueError("Baseline must have same length as heat flow data")
            signal_to_analyze -= baseline

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

        peak_list = []
        for i, peak_idx in enumerate(peaks):
            left_base_idx = int(properties["left_bases"][i])
            right_base_idx = int(properties["right_bases"][i])

            # Ensure indices are within bounds
            left_idx = max(0, left_base_idx - 10)
            right_idx = min(len(temperature) - 1, right_base_idx + 10)

            # Recalculate peak_idx relative to the sliced region
            relative_peak_idx = peak_idx - left_idx

            peak_info = self._analyze_peak_region(
                temperature[left_idx : right_idx + 1],
                heat_flow[left_idx : right_idx + 1],
                relative_peak_idx,
                baseline[left_idx : right_idx + 1] if baseline is not None else None,
            )

            # Store global indices
            peak_info.peak_indices = (left_idx, right_idx)
            peak_list.append(peak_info)

        return peak_list

    def _analyze_peak_region(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> DSCPeak:
        """
        Analyze a single peak region with enhanced characterization.

        Args:
            temperature: Temperature array for the peak region
            heat_flow: Heat flow array for the peak region
            peak_idx: Index of peak maximum relative to the region
            baseline: Optional baseline array for the region

        Returns:
            DSCPeak object with peak characteristics
        """
        self._validate_peak_index(peak_idx, len(temperature))

        # Calculate onset and endset
        onset_temp = self._calculate_onset(temperature, heat_flow, peak_idx, baseline)
        endset_temp = self._calculate_endset(temperature, heat_flow, peak_idx, baseline)

        # Get peak temperature
        peak_temp = temperature[peak_idx]

        # Calculate baseline-corrected heat flow
        heat_flow_corr = heat_flow.copy()
        if baseline is not None:
            heat_flow_corr -= baseline

        # Calculate peak characteristics
        peak_height = float(heat_flow_corr[peak_idx])
        peak_width = self._calculate_peak_width(
            temperature, heat_flow, peak_idx, baseline
        )

        # Calculate area and enthalpy
        peak_mask = (temperature >= onset_temp) & (temperature <= endset_temp)
        peak_area = float(trapz(heat_flow_corr[peak_mask], temperature[peak_mask]))
        enthalpy = abs(peak_area)

        return DSCPeak(
            onset_temperature=float(onset_temp),
            peak_temperature=float(peak_temp),
            endset_temperature=float(endset_temp),
            enthalpy=enthalpy,
            peak_height=peak_height,
            peak_width=float(peak_width),
            peak_area=peak_area,
            baseline_type="provided" if baseline is not None else "none",
            baseline_params={},
            peak_indices=(
                int(np.searchsorted(temperature, onset_temp)),
                int(np.searchsorted(temperature, endset_temp)),
            ),
        )

    def _calculate_onset(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """
        Calculate onset temperature with index validation.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peak_idx: Index of peak maximum
            baseline: Optional baseline array

        Returns:
            Onset temperature (float)
        """
        self._validate_peak_index(peak_idx, len(temperature))
        if peak_idx < 2:
            return temperature[0]

        effective_baseline = (
            baseline if baseline is not None else np.zeros_like(heat_flow)
        )

        # Use extended pre-peak region for derivative
        pre_peak_slice = slice(0, peak_idx)
        if len(temperature[pre_peak_slice]) < 3:
            return temperature[0]

        smooth_flow = safe_savgol_filter(
            heat_flow[pre_peak_slice],
            self.smoothing_window,
            self.smoothing_order,
        )
        dydx = np.gradient(smooth_flow, temperature[pre_peak_slice])
        if len(dydx) == 0:
            return temperature[0]

        max_slope_idx = np.argmax(np.abs(dydx))

        # Tangent line from point of max slope
        slope = dydx[max_slope_idx]
        intercept = heat_flow[max_slope_idx] - slope * temperature[max_slope_idx]
        tangent_line = slope * temperature + intercept

        # Intersection with the baseline
        intersection_temp, _ = find_intersection_point(
            temperature, tangent_line, effective_baseline, max_slope_idx, "backward"
        )
        return intersection_temp

    def _calculate_endset(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """
        Calculate endset temperature.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peak_idx: Index of peak maximum
            baseline: Optional baseline array

        Returns:
            Endset temperature
        """
        self._validate_peak_index(peak_idx, len(temperature))
        if peak_idx >= len(temperature) - 2:
            return temperature[-1]

        effective_baseline = (
            baseline if baseline is not None else np.zeros_like(heat_flow)
        )

        # Use post-peak region for derivative
        post_peak_slice = slice(peak_idx, len(temperature))
        if len(temperature[post_peak_slice]) < 3:
            return temperature[-1]

        smooth_flow = safe_savgol_filter(
            heat_flow[post_peak_slice],
            self.smoothing_window,
            self.smoothing_order,
        )
        dydx = np.gradient(smooth_flow, temperature[post_peak_slice])
        if len(dydx) == 0:
            return temperature[-1]

        max_slope_idx_local = np.argmin(dydx)
        max_slope_idx_global = peak_idx + max_slope_idx_local

        slope = dydx[max_slope_idx_local]
        intercept = (
            heat_flow[max_slope_idx_global] - slope * temperature[max_slope_idx_global]
        )
        tangent_line = slope * temperature + intercept

        # Intersection with the baseline
        intersection_temp, _ = find_intersection_point(
            temperature,
            tangent_line,
            effective_baseline,
            max_slope_idx_global,
            "forward",
        )
        return intersection_temp

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
        effective_baseline = (
            baseline if baseline is not None else np.zeros_like(heat_flow)
        )

        peak_height = heat_flow[peak_idx] - effective_baseline[peak_idx]
        half_height = peak_height / 2 + effective_baseline[peak_idx]
        half_height_line = np.full_like(heat_flow, half_height)

        left_temp, _ = find_intersection_point(
            temperature, heat_flow, half_height_line, peak_idx, "backward"
        )
        right_temp, _ = find_intersection_point(
            temperature, heat_flow, half_height_line, peak_idx, "forward"
        )

        if right_temp > left_temp:
            return right_temp - left_temp
        return 0.0

    def deconvolute_peaks(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        n_peaks: int,
        peak_shape: str = "gaussian",
    ) -> Tuple[List[Dict], NDArray[np.float64]]:
        """
        Deconvolute overlapping peaks.

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
            return amp * np.exp(-(((x - cen) / wid) ** 2))

        def lorentzian(
            x: NDArray[np.float64], amp: float, cen: float, wid: float
        ) -> NDArray[np.float64]:
            return amp * wid**2 / ((x - cen) ** 2 + wid**2)

        peak_func = gaussian if peak_shape == "gaussian" else lorentzian

        smooth_flow = safe_savgol_filter(
            heat_flow, validate_window_size(len(heat_flow), self.smoothing_window), 3
        )

        peaks, properties = signal.find_peaks(
            smooth_flow,
            prominence=np.max(smooth_flow) * 0.1,
            width=5,
            distance=len(temperature) // (n_peaks * 2) if n_peaks > 0 else 1,
        )

        if len(peaks) < n_peaks:
            peak_indices = np.linspace(0, len(temperature) - 1, n_peaks, dtype=int)
        else:
            prominences = properties["prominences"]
            peak_indices = peaks[np.argsort(prominences)[-n_peaks:]]

        p0, bounds_low, bounds_high = [], [], []
        temp_range = temperature.max() - temperature.min()
        min_width, max_width = temp_range * 0.01, temp_range * 0.5

        for idx in sorted(peak_indices):
            amp = smooth_flow[idx]
            cen = temperature[idx]
            wid = temp_range * 0.05

            p0.extend([amp, cen, wid])
            bounds_low.extend([0, cen - temp_range * 0.2, min_width])
            bounds_high.extend([amp * 2, cen + temp_range * 0.2, max_width])

        def fit_function(x: NDArray[np.float64], *params) -> NDArray[np.float64]:
            result = np.zeros_like(x)
            for i in range(0, len(params), 3):
                result += peak_func(x, params[i], params[i + 1], params[i + 2])
            return result

        try:
            popt, _ = optimize.curve_fit(
                fit_function,
                temperature,
                heat_flow,
                p0=p0,
                bounds=(bounds_low, bounds_high),
                maxfev=20000,
            )
            peak_params = []
            fitted_curve = np.zeros_like(temperature)
            for i in range(0, len(popt), 3):
                peak_component = peak_func(temperature, *popt[i : i + 3])
                params = {
                    "amplitude": float(popt[i]),
                    "center": float(popt[i + 1]),
                    "width": float(popt[i + 2]),
                    "area": float(trapz(peak_component, temperature)),
                }
                peak_params.append(params)
                fitted_curve += peak_component
            return peak_params, fitted_curve
        except (optimize.OptimizeWarning, RuntimeError, ValueError):
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
        if not 0 <= peak_idx < array_length:
            raise IndexError(
                f"Peak index {peak_idx} out of bounds for array of length {array_length}"
            )
