"""Peak analysis implementation for DSC data."""

from typing import Any, Dict, List, Optional, Tuple, cast

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
            ValueError: If temperature and heat flow arrays have different lengths or are empty.
        """
        if len(temperature) != len(heat_flow):
            raise ValueError("Temperature and heat flow arrays must have same length")
        if not len(temperature):
            raise ValueError("Input arrays cannot be empty.")

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

        height = np.ptp(signal_to_analyze) * self.height_threshold
        prominence = np.ptp(signal_to_analyze) * self.peak_prominence

        peaks, properties = signal.find_peaks(
            signal_to_analyze,
            prominence=prominence,
            height=height,
            width=validate_window_size(len(signal_to_analyze), self.smoothing_window)
            // 4,
            distance=validate_window_size(
                len(signal_to_analyze), self.smoothing_window
            ),
        )

        peak_list = []
        for i, peak_idx in enumerate(peaks):
            # Use properties from find_peaks which are generally robust
            left_base_idx = int(properties["left_bases"][i])
            right_base_idx = int(properties["right_bases"][i])

            onset_temp = temperature[left_base_idx]
            endset_temp = temperature[right_base_idx]
            peak_temp = temperature[peak_idx]

            heat_flow_corr = heat_flow.copy()
            if baseline is not None:
                heat_flow_corr -= baseline

            peak_mask = slice(left_base_idx, right_base_idx + 1)
            peak_area = float(trapz(heat_flow_corr[peak_mask], temperature[peak_mask]))
            enthalpy = abs(peak_area)
            peak_height = float(properties["prominences"][i])

            # Width calculation
            width_properties = signal.peak_widths(
                signal_to_analyze, [peak_idx], rel_height=0.5
            )
            peak_width: float
            if not width_properties[0].size > 0:
                peak_width = 0.0
            else:
                w_left_temp = np.interp(
                    width_properties[2][0], np.arange(len(temperature)), temperature
                )
                w_right_temp = np.interp(
                    width_properties[3][0], np.arange(len(temperature)), temperature
                )
                peak_width = float(w_right_temp) - float(w_left_temp)

            peak_info = DSCPeak(
                onset_temperature=float(onset_temp),
                peak_temperature=float(peak_temp),
                endset_temperature=float(endset_temp),
                enthalpy=enthalpy,
                peak_height=peak_height,
                peak_width=peak_width,
                peak_area=peak_area,
                baseline_type="provided" if baseline is not None else "none",
                baseline_params={},
                peak_indices=(left_base_idx, right_base_idx),
            )
            peak_list.append(peak_info)

        return peak_list

    def deconvolute_peaks(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        n_peaks: int,
        peak_shape: str = "gaussian",
    ) -> Tuple[List[Dict[str, Any]], NDArray[np.float64]]:
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
            result = amp * np.exp(-(((x - cen) / wid) ** 2))
            return cast(NDArray[np.float64], result)

        def lorentzian(
            x: NDArray[np.float64], amp: float, cen: float, wid: float
        ) -> NDArray[np.float64]:
            result = amp * wid**2 / ((x - cen) ** 2 + wid**2)
            return cast(NDArray[np.float64], result)

        peak_func = gaussian if peak_shape == "gaussian" else lorentzian

        smooth_flow = safe_savgol_filter(
            heat_flow, validate_window_size(len(heat_flow), self.smoothing_window), 3
        )

        min_prominence = np.max(smooth_flow) * 0.05
        min_width = 5
        min_distance = len(temperature) // (n_peaks * 4) if n_peaks > 0 else 1

        peaks, properties = signal.find_peaks(
            smooth_flow,
            prominence=min_prominence,
            width=min_width,
            distance=min_distance,
        )

        prominences = properties.get("prominences")
        if prominences is None:
            prominences = np.array([])

        if len(peaks) < n_peaks:
            if len(prominences) > 0:
                peak_indices = peaks[np.argsort(prominences)[-len(peaks) :]]
            else:
                peak_indices = peaks

            if len(peak_indices) < n_peaks:
                current_peaks = set(peak_indices)
                additional_indices = np.linspace(
                    0, len(temperature) - 1, n_peaks + 2, dtype=int
                )[1:-1]
                for idx in additional_indices:
                    if len(current_peaks) < n_peaks and idx not in current_peaks:
                        current_peaks.add(idx)
                peak_indices = np.array(list(current_peaks))
        else:
            peak_indices = peaks[np.argsort(prominences)[-n_peaks:]]

        p0: List[float] = []
        bounds_low: List[float] = []
        bounds_high: List[float] = []
        temp_range = temperature.max() - temperature.min()
        min_w, max_w = temp_range * 0.01, temp_range * 0.5

        for idx in sorted(peak_indices):
            amp = smooth_flow[idx]
            cen = temperature[idx]
            wid = temp_range * 0.05

            p0.extend([amp, cen, wid])
            bounds_low.extend([0, cen - temp_range * 0.2, min_w])
            bounds_high.extend([amp * 2, cen + temp_range * 0.2, max_w])

        def fit_function(x: NDArray[np.float64], *params: float) -> NDArray[np.float64]:
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
            peak_params: List[Dict[str, Any]] = []
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
