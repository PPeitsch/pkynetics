"""Peak analysis implementation for DSC data."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, signal
from scipy.integrate import trapz

from .types import DSCPeak


def find_intersection_point(
    x: NDArray[np.float64],
    y1: NDArray[np.float64],
    y2: NDArray[np.float64],
    start_idx: int,
    direction: str = "forward",
) -> Tuple[float, int]:
    """
    Find the intersection point between two curves.

    Args:
        x: x-axis values (temperature)
        y1: First curve values
        y2: Second curve values
        start_idx: Starting index for search
        direction: Search direction ("forward" or "backward")

    Returns:
        Tuple of (intersection x value, index)
    """
    if direction == "forward":
        search_range = range(start_idx, len(x) - 1)
    else:
        search_range = range(start_idx, 0, -1)

    for i in search_range:
        if direction == "forward":
            if (y1[i] <= y2[i] and y1[i + 1] >= y2[i + 1]) or (
                y1[i] >= y2[i] and y1[i + 1] <= y2[i + 1]
            ):
                break
        else:
            if (y1[i] <= y2[i] and y1[i - 1] >= y2[i - 1]) or (
                y1[i] >= y2[i] and y1[i - 1] <= y2[i - 1]
            ):
                break
    else:
        return x[start_idx], start_idx

    # Linear interpolation to find precise intersection
    if direction == "forward":
        idx1, idx2 = i, i + 1
    else:
        idx1, idx2 = i - 1, i

    x1, x2 = x[idx1], x[idx2]
    y1_1, y1_2 = y1[idx1], y1[idx2]
    y2_1, y2_2 = y2[idx1], y2[idx2]

    # Intersection point by linear interpolation
    dx = x2 - x1
    dy1 = y1_2 - y1_1
    dy2 = y2_2 - y2_1

    if abs(dy1 - dy2) < 1e-10:  # Parallel lines
        return x[i], i

    x_int = x1 + (y2_1 - y1_1) * dx / (dy1 - dy2)
    return float(x_int), i


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

        # Apply signal smoothing to reduce noise
        smooth_heat_flow = signal.savgol_filter(
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
            width=self.smoothing_window // 2,
            distance=self.smoothing_window,
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

        # Calculate peak width using existing method
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
            peak_indices=(0, len(temperature) - 1),  # Will be updated in find_peaks
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
        smooth_flow = signal.savgol_filter(
            heat_flow[pre_peak], self.smoothing_window, self.smoothing_order
        )

        # Calculate derivative
        dydx = np.gradient(smooth_flow, temperature[pre_peak])

        # Find point of maximum slope
        max_slope_idx = np.argmax(np.abs(dydx))

        # Calculate tangent line
        slope = dydx[max_slope_idx]
        intercept = heat_flow[max_slope_idx] - slope * temperature[max_slope_idx]
        tangent = slope * temperature + intercept

        # Find intersection using multiple points for improved precision
        window = self.smoothing_window
        onset_temps = []
        for i in range(
            max(0, max_slope_idx - window),
            min(max_slope_idx + window, len(temperature)),
        ):
            if (tangent[i] <= baseline[i] and tangent[i + 1] >= baseline[i + 1]) or (
                tangent[i] >= baseline[i] and tangent[i + 1] <= baseline[i + 1]
            ):
                x1, x2 = temperature[i], temperature[i + 1]
                y1 = tangent[i] - baseline[i]
                y2 = tangent[i + 1] - baseline[i + 1]

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
        Calculate endset temperature using tangent method with enhanced precision.

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

        # Use extended post-peak region
        post_peak = slice(peak_idx, len(temperature))

        # Smooth data for derivative calculation
        smooth_flow = signal.savgol_filter(
            heat_flow[post_peak], self.smoothing_window, self.smoothing_order
        )

        # Calculate derivative in post-peak region
        dydx = np.gradient(smooth_flow, temperature[post_peak])

        # Find point of maximum negative slope
        max_slope_idx = peak_idx + np.argmin(dydx)

        # Calculate tangent line
        slope = np.min(dydx)  # Use the maximum negative slope
        intercept = heat_flow[max_slope_idx] - slope * temperature[max_slope_idx]
        tangent = slope * temperature + intercept

        # Find intersection using multiple points
        window = self.smoothing_window
        endset_temps = []
        for i in range(
            max_slope_idx, min(max_slope_idx + window, len(temperature) - 1)
        ):
            if (tangent[i] <= baseline[i] and tangent[i + 1] >= baseline[i + 1]) or (
                tangent[i] >= baseline[i] and tangent[i + 1] <= baseline[i + 1]
            ):
                x1, x2 = temperature[i], temperature[i + 1]
                y1 = tangent[i] - baseline[i]
                y2 = tangent[i + 1] - baseline[i + 1]

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
        Deconvolute overlapping peaks with improved parameter estimation.

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

        # Find initial peak positions
        smooth_flow = signal.savgol_filter(heat_flow, self.smoothing_window, 3)
        peaks, properties = signal.find_peaks(
            smooth_flow, distance=len(temperature) // (n_peaks + 1)
        )

        if len(peaks) < n_peaks:
            # Create evenly spaced initial guesses if not enough peaks found
            peak_indices = np.linspace(0, len(temperature) - 1, n_peaks, dtype=int)
        else:
            # Use the n_peaks highest peaks
            peak_heights = smooth_flow[peaks]
            peak_indices = peaks[np.argsort(peak_heights)[-n_peaks:]]

        # Generate initial parameters
        p0 = []
        bounds_low = []
        bounds_high = []
        for idx in peak_indices:
            amp = heat_flow[idx]
            cen = temperature[idx]
            wid = (temperature[1] - temperature[0]) * 10

            p0.extend([amp, cen, wid])
            bounds_low.extend(
                [0, temperature.min(), 0]
            )  # Amplitude and width must be positive
            bounds_high.extend([amp * 2, temperature.max(), wid * 5])

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
                maxfev=10000,
            )

            # Extract results
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
