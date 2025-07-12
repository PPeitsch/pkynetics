"""Thermal event detection and analysis for DSC data."""

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.optimize import curve_fit

from .types import (
    CrystallizationEvent,
    DSCPeak,
    GlassTransition,
    MeltingEvent,
    PhaseTransition,
)


class ThermalEventDetector:
    """Class for detecting and analyzing thermal events in DSC data."""

    def __init__(
        self,
        smoothing_window: int = 21,
        smoothing_order: int = 3,
        peak_prominence: float = 0.05,
        noise_threshold: float = 0.01,
    ):
        """
        Initialize thermal event detector.

        Args:
            smoothing_window: Window size for Savitzky-Golay smoothing
            smoothing_order: Order for Savitzky-Golay filter
            peak_prominence: Minimum prominence for peak detection
            noise_threshold: Threshold for noise filtering
        """
        self.smoothing_window = smoothing_window
        self.smoothing_order = smoothing_order
        self.peak_prominence = peak_prominence
        self.noise_threshold = noise_threshold

    def detect_events(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peaks: List[DSCPeak],
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> Dict:
        """
        Detect and analyze all thermal events.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            peaks: List of detected peaks
            baseline: Optional baseline array

        Returns:
            Dictionary containing different types of thermal events
        """
        events = {}

        # Detect glass transitions
        gt = self.detect_glass_transition(temperature, heat_flow, baseline)
        if gt:
            events["glass_transitions"] = [gt]

        # Detect crystallization events
        events["crystallization"] = self.detect_crystallization(
            temperature, heat_flow, baseline
        )

        # Detect melting events
        events["melting"] = self.detect_melting(temperature, heat_flow, baseline)

        # Detect other phase transitions
        events["phase_transitions"] = self.detect_phase_transitions(
            temperature, heat_flow, baseline
        )

        return events

    def detect_glass_transition(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> Optional[GlassTransition]:
        """
        Detect and analyze glass transition.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            baseline: Optional baseline array

        Returns:
            GlassTransition object if detected, None otherwise
        """
        if len(temperature) != len(heat_flow):
            raise ValueError(
                "Temperature and heat flow arrays must have the same length."
            )
        if temperature.size < self.smoothing_window:
            return None

        dHf_dt = np.gradient(heat_flow, temperature)
        dHf_smooth = signal.savgol_filter(
            dHf_dt, self.smoothing_window, self.smoothing_order
        )

        prominence = np.ptp(dHf_smooth) * self.peak_prominence
        peaks, props = signal.find_peaks(
            dHf_smooth, prominence=prominence, width=self.smoothing_window // 2
        )

        if not len(peaks):
            return None

        # A glass transition should be a broad peak in the derivative
        # Filter out sharp peaks that are more likely part of melting/crystallization
        widths = signal.peak_widths(dHf_smooth, peaks)[0]
        # Heuristic: a glass transition derivative peak should be reasonably wide
        possible_gt_indices = np.where(widths > 5)[0]
        if not len(possible_gt_indices):
            return None

        best_peak_idx_in_possible = np.argmax(props["prominences"][possible_gt_indices])
        mid_idx = peaks[possible_gt_indices[best_peak_idx_in_possible]]

        width_info = signal.peak_widths(dHf_smooth, [mid_idx], rel_height=0.8)
        start_idx = int(np.floor(width_info[2][0]))
        end_idx = int(np.ceil(width_info[3][0]))

        onset_temp, end_temp = temperature[start_idx], temperature[end_idx]
        mid_temp = temperature[mid_idx]

        delta_cp = np.nan
        if baseline is not None:
            delta_cp = self._calculate_delta_cp(
                temperature, heat_flow, baseline, start_idx, end_idx
            )

        return GlassTransition(
            onset_temperature=float(onset_temp),
            midpoint_temperature=float(mid_temp),
            endpoint_temperature=float(end_temp),
            delta_cp=float(delta_cp),
            width=float(end_temp - onset_temp),
            quality_metrics={},
            baseline_subtracted=baseline is not None,
        )

    def detect_crystallization(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> List[CrystallizationEvent]:
        """
        Detect and analyze crystallization events.
        """
        if len(temperature) == 0:
            raise ValueError("Input arrays cannot be empty.")
        if temperature.size < self.smoothing_window:
            return []

        exothermic_signal = -heat_flow
        prominence = max(
            np.ptp(exothermic_signal) * self.peak_prominence, self.noise_threshold
        )
        peaks, props = signal.find_peaks(
            exothermic_signal, prominence=prominence, distance=self.smoothing_window
        )

        events = []
        for i, peak_idx in enumerate(peaks):
            widths = signal.peak_widths(exothermic_signal, [peak_idx], rel_height=0.5)
            start_idx, end_idx = int(np.floor(widths[2][0])), int(np.ceil(widths[3][0]))

            heat_flow_corr = heat_flow - baseline if baseline is not None else heat_flow
            enthalpy = self._calculate_peak_enthalpy(
                temperature[start_idx:end_idx], heat_flow_corr[start_idx:end_idx]
            )

            events.append(
                CrystallizationEvent(
                    onset_temperature=temperature[start_idx],
                    peak_temperature=temperature[peak_idx],
                    endpoint_temperature=temperature[end_idx],
                    enthalpy=enthalpy,
                    peak_height=-props["peak_heights"][i],
                    width=temperature[end_idx] - temperature[start_idx],
                    quality_metrics={},
                    baseline_subtracted=baseline is not None,
                )
            )
        return events

    def detect_melting(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> List[MeltingEvent]:
        """
        Detect and analyze melting events.
        """
        if len(temperature) == 0:
            raise ValueError("Input arrays cannot be empty.")
        if temperature.size < self.smoothing_window:
            return []

        prominence = max(np.ptp(heat_flow) * self.peak_prominence, self.noise_threshold)
        peaks, props = signal.find_peaks(
            heat_flow, prominence=prominence, distance=self.smoothing_window
        )

        events = []
        for i, peak_idx in enumerate(peaks):
            widths = signal.peak_widths(heat_flow, [peak_idx], rel_height=0.5)
            start_idx, end_idx = int(np.floor(widths[2][0])), int(np.ceil(widths[3][0]))

            heat_flow_corr = heat_flow - baseline if baseline is not None else heat_flow
            enthalpy = self._calculate_peak_enthalpy(
                temperature[start_idx:end_idx], heat_flow_corr[start_idx:end_idx]
            )

            events.append(
                MeltingEvent(
                    onset_temperature=temperature[start_idx],
                    peak_temperature=temperature[peak_idx],
                    endpoint_temperature=temperature[end_idx],
                    enthalpy=enthalpy,
                    peak_height=props["peak_heights"][i],
                    width=temperature[end_idx] - temperature[start_idx],
                    quality_metrics={},
                    baseline_subtracted=baseline is not None,
                )
            )
        return events

    def detect_phase_transitions(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> List[PhaseTransition]:
        """
        Detect and analyze phase transitions.
        """
        if len(temperature) == 0:
            raise ValueError("Input arrays cannot be empty.")

        transitions = []
        melting_events = self.detect_melting(temperature, heat_flow, baseline)
        for event in melting_events:
            transitions.append(
                PhaseTransition(
                    transition_type="first_order",
                    start_temperature=event.onset_temperature,
                    peak_temperature=event.peak_temperature,
                    end_temperature=event.endpoint_temperature,
                    enthalpy=event.enthalpy,
                )
            )

        gt_event = self.detect_glass_transition(temperature, heat_flow, baseline)
        if gt_event:
            transitions.append(
                PhaseTransition(
                    transition_type="second_order",
                    start_temperature=gt_event.onset_temperature,
                    peak_temperature=gt_event.midpoint_temperature,
                    end_temperature=gt_event.endpoint_temperature,
                )
            )

        transitions.sort(key=lambda t: t.start_temperature)
        return transitions

    def _calculate_delta_cp(self, temperature, heat_flow, baseline, start_idx, end_idx):
        pre_region = slice(max(0, start_idx - 20), start_idx)
        post_region = slice(end_idx, min(len(temperature), end_idx + 20))

        if pre_region.stop <= pre_region.start or post_region.stop <= post_region.start:
            return np.nan

        pre_cp = np.mean(heat_flow[pre_region] - baseline[pre_region])
        post_cp = np.mean(heat_flow[post_region] - baseline[post_region])
        return post_cp - pre_cp

    def _calculate_peak_enthalpy(self, temperature, heat_flow):
        return np.trapz(heat_flow, temperature)

    def _is_glass_transition_shape(self, data: NDArray[np.float64]) -> bool:
        # This check is now part of the main detection logic (broad peak in derivative)
        return True

    def _calculate_crystallization_rate(self, temperature, heat_flow):
        return None  # Placeholder

    def _calculate_gt_quality(self, temperature, heat_flow, d2Hf):
        return {}  # Placeholder

    def _calculate_peak_quality(self, temperature, heat_flow, d1=None, d2=None):
        return {}  # Placeholder

    def _calculate_transition_quality(self, temperature, heat_flow, d1, d2):
        return {}  # Placeholder
