"""Peak analysis implementation for DSC data."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, signal
from scipy.integrate import trapz
from scipy.signal import find_peaks

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
    ):
        """
        Initialize peak analyzer.

        Args:
            smoothing_window: Window size for Savitzky-Golay smoothing
            smoothing_order: Order of polynomial for smoothing
            peak_prominence: Minimum prominence for peak detection
        """
        self.smoothing_window = smoothing_window
        self.smoothing_order = smoothing_order
        self.peak_prominence = peak_prominence

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
        """
        # Validate inputs
        if len(temperature) != len(heat_flow):
            raise ValueError("Temperature and heat flow arrays must have same length")

        # Apply baseline correction if provided
        signal_to_analyze = heat_flow.copy()
        if baseline is not None:
            signal_to_analyze = heat_flow - baseline

        # Find peaks using scipy.signal
        peaks, properties = signal.find_peaks(
            signal_to_analyze,
            prominence=self.peak_prominence,
            width=self.smoothing_window // 2,
        )

        peak_list = []
        for i, peak_idx in enumerate(peaks):
            # Find peak boundaries
            left_idx = int(properties["left_bases"][i])
            right_idx = int(properties["right_bases"][i])

            # Calculate peak characteristics
            peak_info = self._analyze_peak_region(
                temperature[left_idx : right_idx + 1],
                signal_to_analyze[left_idx : right_idx + 1],
                peak_idx - left_idx,
                baseline[left_idx : right_idx + 1] if baseline is not None else None,
            )

            # Update peak indices to global coordinates
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
        Calculate onset temperature using tangent method.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peak_idx: Index of peak maximum
            baseline: Optional baseline array

        Returns:
            Onset temperature
        """
        # Use linear baseline if none provided
        if baseline is None:
            baseline = np.zeros_like(heat_flow)

        # Find region before peak for tangent calculation
        pre_peak = slice(0, peak_idx)

        # Calculate derivative in pre-peak region
        dydx = np.gradient(heat_flow[pre_peak], temperature[pre_peak])

        # Find point of maximum slope
        max_slope_idx = np.argmax(np.abs(dydx))

        # Calculate tangent line
        slope = dydx[max_slope_idx]
        intercept = heat_flow[max_slope_idx] - slope * temperature[max_slope_idx]
        tangent = slope * temperature + intercept

        # Find intersection with baseline
        onset_temp, _ = find_intersection_point(
            temperature, tangent, baseline, max_slope_idx, "backward"
        )

        return onset_temp

    def _calculate_endset(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """
        Calculate endset temperature using tangent method.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peak_idx: Index of peak maximum
            baseline: Optional baseline array

        Returns:
            Endset temperature
        """
        # Use linear baseline if none provided
        if baseline is None:
            baseline = np.zeros_like(heat_flow)

        # Find region after peak for tangent calculation
        post_peak = slice(peak_idx, len(temperature))

        # Calculate derivative in post-peak region
        dydx = np.gradient(heat_flow[post_peak], temperature[post_peak])

        # Find point of maximum slope
        max_slope_idx = peak_idx + np.argmin(dydx)

        # Calculate tangent line
        slope = dydx[max_slope_idx - peak_idx]
        intercept = heat_flow[max_slope_idx] - slope * temperature[max_slope_idx]
        tangent = slope * temperature + intercept

        # Find intersection with baseline
        endset_temp, _ = find_intersection_point(
            temperature, tangent, baseline, max_slope_idx, "forward"
        )

        return endset_temp

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
            peak_shape: Shape function ("gaussian" or "lorentzian")

        Returns:
            Tuple of (list of peak parameters, fitted curve)
        """

        def gaussian(
            x: NDArray[np.float64], amp: float, cen: float, wid: float
        ) -> NDArray[np.float64]:
            """Gaussian peak function."""
            return amp * np.exp(-(((x - cen) / wid) ** 2))

        def lorentzian(
            x: NDArray[np.float64], amp: float, cen: float, wid: float
        ) -> NDArray[np.float64]:
            """Lorentzian peak function."""
            return amp * wid**2 / ((x - cen) ** 2 + wid**2)

        peak_func = gaussian if peak_shape == "gaussian" else lorentzian

        # Initial guess for peak parameters
        peaks, properties = find_peaks(heat_flow, distance=len(temperature) // n_peaks)
        if len(peaks) < n_peaks:
            peaks = np.array([i * len(temperature) // n_peaks for i in range(n_peaks)])

        p0 = []
        for idx in peaks[:n_peaks]:
            p0.extend(
                [
                    heat_flow[idx],  # amplitude
                    temperature[idx],  # center
                    (temperature[1] - temperature[0]) * 10,  # width
                ]
            )

        def fit_function(x: NDArray[np.float64], *params) -> NDArray[np.float64]:
            """Combined peak function for fitting."""
            result = np.zeros_like(x)
            for i in range(0, len(params), 3):
                result += peak_func(x, params[i], params[i + 1], params[i + 2])
            return result

        try:
            # Fit peaks
            popt, _ = optimize.curve_fit(fit_function, temperature, heat_flow, p0=p0)

            # Extract individual peak parameters
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

        except optimize.OptimizeWarning:
            # Return empty results if fitting fails
            return [], np.zeros_like(temperature)

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
