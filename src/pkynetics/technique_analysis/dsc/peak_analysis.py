"""Peak analysis for DSC data."""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapz
from scipy.signal import find_peaks, savgol_filter

from .core import DSCPeak


class PeakAnalyzer:
    """Class for DSC peak analysis."""

    def __init__(self):
        """Initialize peak analyzer."""
        pass

    def find_peaks(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        prominence: float = 0.1,
        width: Optional[int] = None,
        height: Optional[float] = None,
    ) -> List[DSCPeak]:
        """Find and analyze peaks in DSC data."""
        # Smooth data for better peak detection
        heat_flow_smooth = savgol_filter(heat_flow, window_length=21, polyorder=3)

        # Find peaks
        peaks, properties = find_peaks(
            heat_flow_smooth, prominence=prominence, width=width, height=height
        )

        peak_list = []
        for i, peak_idx in enumerate(peaks):
            # Get peak boundaries
            left_idx = int(properties["left_bases"][i])
            right_idx = int(properties["right_bases"][i])

            # Analyze peak region
            peak_info = self._analyze_peak_region(
                temperature[left_idx : right_idx + 1],
                heat_flow[left_idx : right_idx + 1],
                peak_idx - left_idx,
            )
            peak_list.append(peak_info)

        return peak_list

    def _analyze_peak_region(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
    ) -> DSCPeak:
        """Analyze a single peak region."""
        # Calculate onset and endset
        onset_temp = self._calculate_onset(temperature, heat_flow, peak_idx)
        endset_temp = self._calculate_endset(temperature, heat_flow, peak_idx)

        # Calculate peak characteristics
        peak_temp = temperature[peak_idx]
        peak_height = heat_flow[peak_idx]
        peak_width = self._calculate_peak_width(temperature, heat_flow, peak_idx)

        # Calculate area and enthalpy
        peak_area = trapz(heat_flow, temperature)
        enthalpy = abs(peak_area)

        return DSCPeak(
            onset_temperature=onset_temp,
            peak_temperature=peak_temp,
            endset_temperature=endset_temp,
            enthalpy=enthalpy,
            peak_height=peak_height,
            peak_width=peak_width,
            peak_area=peak_area,
            baseline_type="linear",  # To be updated with actual baseline info
            baseline_params={},  # To be updated with actual baseline params
            peak_indices=(0, len(temperature) - 1),
        )

    def _calculate_onset(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
    ) -> float:
        """Calculate onset temperature."""
        # Implementation pending
        return float(temperature[0])

    def _calculate_endset(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
    ) -> float:
        """Calculate endset temperature."""
        # Implementation pending
        return float(temperature[-1])

    def _calculate_peak_width(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
    ) -> float:
        """Calculate peak width at half height."""
        # Implementation pending
        return float(temperature[-1] - temperature[0])
