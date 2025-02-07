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

        peak_list = []
        for i, peak_idx in enumerate(peaks):
            left_idx = int(properties["left_bases"][i])
            right_idx = int(properties["right_bases"][i])

            peak_info = self._analyze_peak_region(
                temperature[left_idx : right_idx + 1],
                heat_flow[left_idx : right_idx + 1],
                peak_idx - left_idx,
                baseline[left_idx : right_idx + 1] if baseline is not None else None,
            )

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
        Analyze a single peak region.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peak_idx: Index of peak maximum
            baseline: Optional baseline array

        Returns:
            DSCPeak object with peak characteristics
        """
        # Calculate onset and endset using existing methods
        onset_temp = self._calculate_onset(temperature, heat_flow, peak_idx, baseline)
        endset_temp = self._calculate_endset(temperature, heat_flow, peak_idx, baseline)

        # Get peak characteristics
        peak_temp = temperature[peak_idx]
        peak_height = heat_flow[peak_idx]
        if baseline is not None:
            peak_height -= baseline[peak_idx]

        # Calculate peak width
        peak_width = self._calculate_peak_width(
            temperature, heat_flow, peak_idx, baseline
        )

        # Calculate peak area
        signal_to_integrate = heat_flow
        if baseline is not None:
            signal_to_integrate = heat_flow - baseline
        peak_area = trapz(signal_to_integrate, temperature)

        return DSCPeak(
            onset_temperature=float(onset_temp),
            peak_temperature=float(peak_temp),
            endset_temperature=float(endset_temp),
            enthalpy=float(abs(peak_area)),
            peak_height=float(peak_height),
            peak_width=float(peak_width),
            peak_area=float(peak_area),
            baseline_type="none" if baseline is None else "provided",
            baseline_params={},
            peak_indices=(0, len(temperature) - 1),
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

        if baseline is None:
            baseline = np.zeros_like(heat_flow)

        # Use extended pre-peak region
        pre_peak = slice(0, peak_idx)

        # Smooth data for derivative calculation
        smooth_flow = safe_savgol_filter(
            heat_flow[pre_peak],
            min(len(heat_flow[pre_peak]), self.smoothing_window),
            self.smoothing_order,
        )

        # Calculate derivative
        dydx = np.gradient(smooth_flow, temperature[pre_peak])

        # Find point of maximum slope
        max_slope_idx = np.argmax(np.abs(dydx))

        # Calculate tangent line parameters
        slope = dydx[max_slope_idx]
        intercept = heat_flow[max_slope_idx] - slope * temperature[max_slope_idx]

        # Calculate tangent line for intersection search
        search_temp = temperature[:peak_idx]  # Solo hasta el pico
        tangent = slope * search_temp + intercept

        # Find intersection using multiple points for improved precision
        window = min(
            validate_window_size(len(temperature), self.smoothing_window), max_slope_idx
        )

        onset_temps = []
        for i in range(max(0, max_slope_idx - window), peak_idx - 1):
            y_current = tangent[i] - baseline[i]
            y_next = tangent[i + 1] - baseline[i + 1]

            if y_current * y_next <= 0:
                x1, x2 = temperature[i], temperature[i + 1]
                y1, y2 = y_current, y_next

                if abs(y2 - y1) > 1e-10:
                    x_int = x1 - y1 * (x2 - x1) / (y2 - y1)
                    onset_temps.append(x_int)

        if not onset_temps:
            return temperature[max_slope_idx]

        return float(np.median(onset_temps))

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
        if baseline is None:
            baseline = np.zeros_like(heat_flow)

        # Use post-peak region
        post_peak = slice(peak_idx, len(temperature))

        # Smooth data for derivative calculation
        smooth_flow = safe_savgol_filter(
            heat_flow[post_peak],
            min(len(heat_flow[post_peak]), self.smoothing_window),
            self.smoothing_order,
        )

        # Calculate derivative in post-peak region
        dydx = np.gradient(smooth_flow, temperature[post_peak])

        # Find point of maximum negative slope
        max_slope_idx = peak_idx + np.argmin(dydx)

        # Calculate tangent line parameters
        slope = np.min(dydx)
        intercept = heat_flow[max_slope_idx] - slope * temperature[max_slope_idx]

        # Calculate tangent line for intersection search
        search_temp = temperature[peak_idx:]  # Solo después del pico
        tangent = slope * search_temp + intercept

        search_baseline = baseline[peak_idx:]

        endset_temps = []
        for i in range(len(search_temp) - 1):
            y_current = tangent[i] - search_baseline[i]
            y_next = tangent[i + 1] - search_baseline[i + 1]

            # Verificar cambio de signo
            if y_current * y_next <= 0:
                x1, x2 = search_temp[i], search_temp[i + 1]
                y1, y2 = y_current, y_next

                if abs(y2 - y1) > 1e-10:
                    x_int = x1 - y1 * (x2 - x1) / (y2 - y1)
                    endset_temps.append(x_int)

        if not endset_temps:
            return temperature[max_slope_idx]

        return float(np.median(endset_temps))

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

        # Apply smoothing for better peak detection
        smooth_flow = safe_savgol_filter(
            heat_flow, validate_window_size(len(heat_flow), self.smoothing_window), 3
        )

        # Find all potential peaks
        peaks, properties = signal.find_peaks(
            smooth_flow,
            prominence=np.max(smooth_flow) * 0.1,
            width=5,
            distance=len(temperature) // (n_peaks * 2),
        )

        if len(peaks) < n_peaks:
            # Divide temperature range into n_peaks regions
            temp_range = temperature.max() - temperature.min()
            region_size = temp_range / n_peaks
            peak_positions = [
                temperature.min() + region_size * (i + 0.5) for i in range(n_peaks)
            ]
            peak_indices = [
                np.abs(temperature - pos).argmin() for pos in peak_positions
            ]
        else:
            # Select peaks with highest prominence
            prominences = properties["prominences"]
            peak_indices = peaks[np.argsort(prominences)[-n_peaks:]]

        # Generate initial parameters and bounds
        p0 = []
        bounds_low = []
        bounds_high = []

        temp_range = temperature.max() - temperature.min()
        min_width = temp_range * 0.02  # 2% of temperature range
        max_width = temp_range * 0.2  # 20% of temperature range

        for idx in peak_indices:
            amp = smooth_flow[idx]
            cen = temperature[idx]
            wid = temp_range * 0.05  # Initial width 5% of temperature range

            p0.extend([amp, cen, wid])
            bounds_low.extend([0, cen - temp_range * 0.15, min_width])
            bounds_high.extend([amp * 2, cen + temp_range * 0.15, max_width])

        def fit_function(x: NDArray[np.float64], *params) -> NDArray[np.float64]:
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
                        trapz(
                            peak_func(temperature, popt[i], popt[i + 1], popt[i + 2]),
                            temperature,
                        )
                    ),
                }
                peak_params.append(params)
                fitted_curve += peak_func(temperature, *popt[i : i + 3])

            return peak_params, fitted_curve

        except (optimize.OptimizeWarning, RuntimeError):
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
    ) -> DSCPeak:
        """
        Analyze a single peak region with enhanced characterization.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peak_idx: Index of peak maximum
            baseline: Optional baseline array

        Returns:
            DSCPeak object with peak characteristics
        """
        # Calculate onset and endset
        onset_temp = self._calculate_onset(temperature, heat_flow, peak_idx, baseline)
        endset_temp = self._calculate_endset(temperature, heat_flow, peak_idx, baseline)

        # Get peak temperature
        peak_temp = temperature[peak_idx]

        # Calculate baseline-corrected heat flow
        if baseline is None:
            baseline = np.zeros_like(heat_flow)
        heat_flow_corr = heat_flow - baseline

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
