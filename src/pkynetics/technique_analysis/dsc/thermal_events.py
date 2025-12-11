"""Thermal event detection and analysis for DSC data."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

from .types import CrystallizationEvent, DSCPeak, GlassTransition, MeltingEvent, PhaseTransition


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
    ) -> Dict[str, Any]:
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
        events: Dict[str, Any] = {}

        # Detect glass transitions
        gt = self.detect_glass_transition(temperature, heat_flow, baseline)
        if gt:
            events["glass_transitions"] = [gt]

        # Detect crystallization events
        cryst_events = self.detect_crystallization(temperature, heat_flow, baseline)
        if cryst_events:
            events["crystallization"] = cryst_events

        # Detect melting events
        melt_events = self.detect_melting(temperature, heat_flow, baseline)
        if melt_events:
            events["melting"] = melt_events

        # Detect other phase transitions
        phase_transitions = self.detect_phase_transitions(
            temperature, heat_flow, baseline
        )
        if phase_transitions:
            events["phase_transitions"] = phase_transitions

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

        # --- 1. Identify and locate sharp first-order peaks to create exclusion zones ---
        signal_abs = np.abs(heat_flow - np.mean(heat_flow))
        signal_prominence = np.ptp(signal_abs) * 0.1
        sharp_peaks, _ = signal.find_peaks(signal_abs, prominence=signal_prominence)

        # --- 2. Find potential glass transitions (peaks in the derivative) ---
        dHf_dt = np.gradient(heat_flow, temperature)
        dHf_smooth = signal.savgol_filter(
            dHf_dt, self.smoothing_window, self.smoothing_order
        )

        # Use a combination of relative and absolute prominence to avoid noise detection
        relative_prominence = np.ptp(dHf_smooth) * self.peak_prominence
        absolute_prominence_threshold = (
            1e-4  # A hard threshold for significant slope changes
        )

        prominence = max(relative_prominence, absolute_prominence_threshold)

        derivative_peaks, props = signal.find_peaks(
            dHf_smooth, prominence=prominence, width=self.smoothing_window // 4
        )

        # --- 3. Validate potential Tg peaks against exclusion zones ---
        for i, peak_idx in enumerate(derivative_peaks):
            is_near_sharp_peak = False
            for sharp_peak_idx in sharp_peaks:
                if abs(peak_idx - sharp_peak_idx) < self.smoothing_window * 2:
                    is_near_sharp_peak = True
                    break

            if not is_near_sharp_peak:
                mid_idx = peak_idx
                width_info = signal.peak_widths(dHf_smooth, [mid_idx], rel_height=0.8)
                if not width_info[0].size > 0:
                    continue
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

        return None  # No valid glass transition found

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
            exothermic_signal,
            prominence=prominence,
            distance=self.smoothing_window,
            height=0,
        )

        events = []
        for i, peak_idx in enumerate(peaks):
            widths = signal.peak_widths(exothermic_signal, [peak_idx], rel_height=0.5)
            if not widths[0].size > 0:
                continue
            start_idx, end_idx = int(np.floor(widths[2][0])), int(np.ceil(widths[3][0]))

            heat_flow_corr = heat_flow - baseline if baseline is not None else heat_flow
            enthalpy = self._calculate_peak_enthalpy(
                temperature[start_idx : end_idx + 1],
                heat_flow_corr[start_idx : end_idx + 1],
            )

            peak_height = -props["prominences"][i]

            events.append(
                CrystallizationEvent(
                    onset_temperature=temperature[start_idx],
                    peak_temperature=temperature[peak_idx],
                    endpoint_temperature=temperature[end_idx],
                    enthalpy=enthalpy,
                    peak_height=peak_height,
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

        # --- Calculate derivative once for all checks ---
        dHf_dt = np.gradient(heat_flow, temperature)
        dHf_smooth = signal.savgol_filter(
            dHf_dt, self.smoothing_window, self.smoothing_order
        )

        # --- Find all potential positive peaks ---
        prominence = max(np.ptp(heat_flow) * self.peak_prominence, self.noise_threshold)
        peaks, props = signal.find_peaks(
            heat_flow, prominence=prominence, distance=self.smoothing_window, height=0
        )

        # --- Filter peaks based on derivative amplitude ---
        filtered_peaks = []
        filtered_props_indices = []
        # A true first-order peak must have a significant derivative swing
        min_derivative_amplitude = np.ptp(dHf_smooth) * 0.1

        for i, peak_idx in enumerate(peaks):
            window_radius = self.smoothing_window
            start_window = max(0, peak_idx - window_radius)
            end_window = min(len(dHf_smooth), peak_idx + window_radius)

            local_derivative_amplitude = np.ptp(dHf_smooth[start_window:end_window])

            if local_derivative_amplitude > min_derivative_amplitude:
                filtered_peaks.append(peak_idx)
                filtered_props_indices.append(i)

        # --- Process only the filtered, valid peaks ---
        events = []
        for i_filtered, peak_idx in enumerate(filtered_peaks):
            original_prop_idx = filtered_props_indices[i_filtered]
            widths = signal.peak_widths(heat_flow, [peak_idx], rel_height=0.5)
            if not widths[0].size > 0:
                continue
            start_idx, end_idx = int(np.floor(widths[2][0])), int(np.ceil(widths[3][0]))

            heat_flow_corr = heat_flow - baseline if baseline is not None else heat_flow
            enthalpy = self._calculate_peak_enthalpy(
                temperature[start_idx : end_idx + 1],
                heat_flow_corr[start_idx : end_idx + 1],
            )

            peak_height = props["prominences"][original_prop_idx]

            events.append(
                MeltingEvent(
                    onset_temperature=temperature[start_idx],
                    peak_temperature=temperature[peak_idx],
                    endpoint_temperature=temperature[end_idx],
                    enthalpy=enthalpy,
                    peak_height=peak_height,
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

        transitions: List[PhaseTransition] = []
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

    def _calculate_delta_cp(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: NDArray[np.float64],
        start_idx: int,
        end_idx: int,
    ) -> float:
        """Calculate change in heat capacity across glass transition."""
        pre_region = slice(max(0, start_idx - 20), start_idx)
        post_region = slice(end_idx + 1, min(len(temperature), end_idx + 21))

        if pre_region.stop <= pre_region.start or post_region.stop <= post_region.start:
            return np.nan

        pre_cp = np.mean((heat_flow - baseline)[pre_region])
        post_cp = np.mean((heat_flow - baseline)[post_region])
        return float(post_cp - pre_cp)

    def _calculate_peak_enthalpy(
        self, temperature: NDArray[np.float64], heat_flow: NDArray[np.float64]
    ) -> float:
        """Calculate enthalpy from peak area."""
        return float(trapezoid(heat_flow, temperature))

    def _is_glass_transition_shape(self, data: NDArray[np.float64]) -> bool:
        """Check if data has characteristic glass transition shape."""
        # This check is now part of the main detection logic
        return True

    def _calculate_crystallization_rate(
        self, temperature: NDArray[np.float64], heat_flow: NDArray[np.float64]
    ) -> Optional[float]:
        """Calculate crystallization rate from peak shape."""
        # Placeholder for future implementation
        return None

    def _calculate_gt_quality(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        d2Hf: NDArray[np.float64],
    ) -> Dict[str, float]:
        """Calculate quality metrics for glass transition."""
        # Placeholder for future implementation
        return {}

    def _calculate_peak_quality(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        d1: Optional[NDArray[np.float64]] = None,
        d2: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, float]:
        """Calculate quality metrics with size validation."""
        # Placeholder for future implementation
        return {}

    def _calculate_transition_quality(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        d1: NDArray[np.float64],
        d2: NDArray[np.float64],
    ) -> Dict[str, float]:
        """Calculate quality metrics for phase transitions."""
        # Placeholder for future implementation
        return {}
