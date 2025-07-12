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

        # Calculate first derivative of heat flow w.r.t. temperature
        dHf_dt = np.gradient(heat_flow, temperature)
        dHf_smooth = signal.savgol_filter(
            dHf_dt, self.smoothing_window, self.smoothing_order
        )

        # A glass transition is a peak in the first derivative
        prominence = np.ptp(dHf_smooth) * self.peak_prominence
        peaks, properties = signal.find_peaks(
            dHf_smooth, prominence=prominence, height=self.noise_threshold
        )
        neg_peaks, neg_properties = signal.find_peaks(
            -dHf_smooth, prominence=prominence, height=self.noise_threshold
        )

        all_peaks = np.concatenate((peaks, neg_peaks))
        all_prominences = np.concatenate(
            (properties.get("prominences", []), neg_properties.get("prominences", []))
        )

        if len(all_peaks) == 0:
            return None

        # Select the most prominent transition
        mid_idx = all_peaks[np.argmax(all_prominences)]

        width_info = signal.peak_widths(dHf_smooth, [mid_idx], rel_height=0.8)
        if len(width_info[0]) == 0:
            return None

        start_idx = int(np.floor(width_info[2][0]))
        end_idx = int(np.ceil(width_info[3][0]))

        onset_temp = temperature[start_idx]
        end_temp = temperature[end_idx]
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
            quality_metrics=self._calculate_gt_quality(
                temperature[start_idx:end_idx],
                heat_flow[start_idx:end_idx],
                dHf_smooth[start_idx:end_idx],
            ),
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

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            baseline: Optional baseline array

        Returns:
            List of CrystallizationEvent objects
        """
        if temperature.size < self.smoothing_window:
            return []

        exothermic_signal = -heat_flow
        prominence = max(
            np.ptp(exothermic_signal) * self.peak_prominence, self.noise_threshold
        )

        peaks, properties = signal.find_peaks(
            exothermic_signal, prominence=prominence, distance=self.smoothing_window
        )

        events = []
        for i, peak_idx in enumerate(peaks):
            width_info = signal.peak_widths(
                exothermic_signal, [peak_idx], rel_height=0.5
            )
            if len(width_info[0]) == 0:
                continue

            onset_idx = int(np.floor(width_info[2][0]))
            end_idx = int(np.ceil(width_info[3][0]))

            if baseline is not None:
                heat_flow_corr = heat_flow - baseline
            else:
                heat_flow_corr = heat_flow

            enthalpy = self._calculate_peak_enthalpy(
                temperature[onset_idx : end_idx + 1],
                heat_flow_corr[onset_idx : end_idx + 1],
            )

            rate = self._calculate_crystallization_rate(
                temperature[onset_idx : end_idx + 1],
                heat_flow_corr[onset_idx : end_idx + 1],
            )

            events.append(
                CrystallizationEvent(
                    onset_temperature=float(temperature[onset_idx]),
                    peak_temperature=float(temperature[peak_idx]),
                    endpoint_temperature=float(temperature[end_idx]),
                    enthalpy=float(enthalpy),
                    peak_height=float(-properties["prominences"][i]),
                    width=float(temperature[end_idx] - temperature[onset_idx]),
                    crystallization_rate=float(rate) if rate is not None else None,
                    quality_metrics=self._calculate_peak_quality(
                        temperature[onset_idx : end_idx + 1],
                        heat_flow_corr[onset_idx : end_idx + 1],
                    ),
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

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            baseline: Optional baseline array

        Returns:
            List of MeltingEvent objects
        """
        if len(temperature) != len(heat_flow):
            raise ValueError(
                "Temperature and heat flow arrays must have the same length."
            )
        if temperature.size < self.smoothing_window:
            return []

        prominence = max(np.ptp(heat_flow) * self.peak_prominence, self.noise_threshold)
        peaks, properties = signal.find_peaks(
            heat_flow, prominence=prominence, distance=self.smoothing_window
        )

        events = []
        for i, peak_idx in enumerate(peaks):
            width_info = signal.peak_widths(heat_flow, [peak_idx], rel_height=0.5)
            if len(width_info[0]) == 0:
                continue

            onset_idx = int(np.floor(width_info[2][0]))
            end_idx = int(np.ceil(width_info[3][0]))

            if baseline is not None:
                heat_flow_corr = heat_flow - baseline
            else:
                heat_flow_corr = heat_flow

            enthalpy = self._calculate_peak_enthalpy(
                temperature[onset_idx : end_idx + 1],
                heat_flow_corr[onset_idx : end_idx + 1],
            )

            events.append(
                MeltingEvent(
                    onset_temperature=float(temperature[onset_idx]),
                    peak_temperature=float(temperature[peak_idx]),
                    endpoint_temperature=float(temperature[end_idx]),
                    enthalpy=float(enthalpy),
                    peak_height=float(properties["prominences"][i]),
                    width=float(temperature[end_idx] - temperature[onset_idx]),
                    quality_metrics=self._calculate_peak_quality(
                        temperature[onset_idx : end_idx + 1],
                        heat_flow_corr[onset_idx : end_idx + 1],
                    ),
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

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            baseline: Optional baseline array

        Returns:
            List of PhaseTransition objects
        """
        if temperature.size < self.smoothing_window:
            return []

        transitions = []

        # Detect first-order transitions (melting/crystallization)
        melting_events = self.detect_melting(temperature, heat_flow, baseline)
        for event in melting_events:
            transitions.append(
                PhaseTransition(
                    transition_type="first_order_endothermic",
                    start_temperature=event.onset_temperature,
                    peak_temperature=event.peak_temperature,
                    end_temperature=event.endpoint_temperature,
                    enthalpy=event.enthalpy,
                    transition_width=event.width,
                    quality_metrics=event.quality_metrics,
                )
            )

        cryst_events = self.detect_crystallization(temperature, heat_flow, baseline)
        for event in cryst_events:
            transitions.append(
                PhaseTransition(
                    transition_type="first_order_exothermic",
                    start_temperature=event.onset_temperature,
                    peak_temperature=event.peak_temperature,
                    end_temperature=event.endpoint_temperature,
                    enthalpy=event.enthalpy,
                    transition_width=event.width,
                    quality_metrics=event.quality_metrics,
                )
            )

        # Detect second-order transitions (glass transitions)
        gt_event = self.detect_glass_transition(temperature, heat_flow, baseline)
        if gt_event:
            transitions.append(
                PhaseTransition(
                    transition_type="second_order",
                    start_temperature=gt_event.onset_temperature,
                    peak_temperature=gt_event.midpoint_temperature,
                    end_temperature=gt_event.endpoint_temperature,
                    enthalpy=None,
                    transition_width=gt_event.width,
                    quality_metrics=gt_event.quality_metrics,
                )
            )

        # Sort transitions by peak temperature
        transitions.sort(key=lambda t: t.peak_temperature)

        return transitions

    def _is_glass_transition_shape(self, data: NDArray[np.float64]) -> bool:
        """Check if data has characteristic glass transition shape."""
        # Normalize data
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Fit sigmoid function
        def sigmoid(x, a, b, c, d):
            return a + (b - a) / (1 + np.exp(-c * (x - d)))

        try:
            x = np.linspace(0, 1, len(data))
            popt, _ = curve_fit(sigmoid, x, norm_data)
            fitted = sigmoid(x, *popt)

            # Calculate fit quality
            r2 = 1 - np.sum((norm_data - fitted) ** 2) / np.sum(
                (norm_data - np.mean(norm_data)) ** 2
            )

            # Check if fit is good enough and shape parameters are reasonable
            return r2 > 0.95 and popt[2] > 0  # Positive slope parameter

        except RuntimeError:
            return False

    def _calculate_delta_cp(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: NDArray[np.float64],
        start_idx: int,
        end_idx: int,
    ) -> float:
        """Calculate change in heat capacity across glass transition."""
        # Use regions before and after the transition to calculate Cp
        pre_region_end = max(0, start_idx - 10)
        post_region_start = min(len(temperature), end_idx + 10)

        pre_cp = np.mean(
            heat_flow[pre_region_end:start_idx] - baseline[pre_region_end:start_idx]
        )
        post_cp = np.mean(
            heat_flow[end_idx:post_region_start] - baseline[end_idx:post_region_start]
        )

        if np.isnan(pre_cp) or np.isnan(post_cp):
            return np.nan

        return post_cp - pre_cp

    def _calculate_crystallization_rate(
        self, temperature: NDArray[np.float64], heat_flow: NDArray[np.float64]
    ) -> Optional[float]:
        """Calculate crystallization rate from peak shape."""
        try:
            # Calculate rate as maximum slope of cumulative heat flow
            cumulative = np.cumsum(heat_flow)
            rate = float(np.max(np.gradient(cumulative, temperature)))
            return rate
        except (ValueError, RuntimeError):
            return None

    def _find_onset_index(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
    ) -> int:
        """Find onset point index using tangent method."""
        # Search in region before peak
        search_region = slice(max(0, peak_idx - 100), peak_idx)

        # Calculate derivatives
        dHf = np.gradient(heat_flow[search_region], temperature[search_region])

        # Find point of maximum rate change
        d2Hf = np.gradient(dHf)
        onset_idx = search_region.start + np.argmax(np.abs(d2Hf))

        return onset_idx

    def _find_endpoint_index(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peak_idx: int,
    ) -> int:
        """Find endpoint index using tangent method."""
        # Search in region after peak
        search_region = slice(peak_idx, min(len(heat_flow), peak_idx + 100))

        # Calculate derivatives
        dHf = np.gradient(heat_flow[search_region], temperature[search_region])

        # Find point of maximum rate change
        d2Hf = np.gradient(dHf)
        end_idx = search_region.start + np.argmax(np.abs(d2Hf))

        return end_idx

    def _calculate_peak_enthalpy(
        self, temperature: NDArray[np.float64], heat_flow: NDArray[np.float64]
    ) -> float:
        """Calculate enthalpy from peak area."""
        # Integrate heat flow with respect to temperature
        enthalpy = np.trapz(heat_flow, temperature)
        return float(abs(enthalpy))

    def _calculate_gt_quality(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        d2Hf: NDArray[np.float64],
    ) -> Dict[str, float]:
        """Calculate quality metrics for glass transition."""
        metrics = {}

        # Calculate signal-to-noise ratio
        noise = np.std(d2Hf[:20])  # Use start of region for noise estimate
        signal = np.max(np.abs(d2Hf))
        metrics["snr"] = float(signal / noise if noise > 0 else 0)

        # Calculate symmetry
        mid_idx = len(heat_flow) // 2
        left_half = heat_flow[:mid_idx]
        right_half = heat_flow[mid_idx:]
        metrics["symmetry"] = float(np.corrcoef(left_half, right_half[::-1])[0, 1])

        # Calculate smoothness
        metrics["smoothness"] = float(1 / (1 + np.std(np.gradient(heat_flow))))

        return metrics

    def _calculate_peak_quality(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        d1: Optional[NDArray[np.float64]] = None,
        d2: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, float]:
        """Calculate quality metrics with size validation."""
        metrics = {}

        # Calculate peak sharpness only if arrays are large enough
        peak_idx = np.argmax(np.abs(heat_flow))
        left_half = heat_flow[:peak_idx]
        right_half = heat_flow[peak_idx:]

        if len(left_half) > 1 and len(right_half) > 1:
            left_grad = np.gradient(left_half) if len(left_half) > 2 else [0]
            right_grad = np.gradient(right_half) if len(right_half) > 2 else [0]
            metrics["sharpness"] = float(np.max(left_grad) - np.min(right_grad))
        else:
            metrics["sharpness"] = 0.0

        # Basic signal metrics don't require gradients
        noise = np.std(heat_flow[: min(20, len(heat_flow))])
        signal = np.max(np.abs(heat_flow))
        metrics["peak_to_noise"] = float(signal / noise if noise > 0 else 0)

        # Calculate baseline stability
        start_region = heat_flow[: min(20, len(heat_flow))]
        metrics["baseline_stability"] = float(1 / (1 + np.std(start_region)))

        return metrics

    def _calculate_transition_quality(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        d1: NDArray[np.float64],
        d2: NDArray[np.float64],
    ) -> Dict[str, float]:
        """Calculate quality metrics for phase transitions."""
        metrics = {}

        # Calculate transition sharpness
        metrics["sharpness"] = float(np.max(np.abs(d1)))

        # Calculate transition clarity
        metrics["clarity"] = float(np.max(np.abs(d2)))

        # Calculate signal quality
        noise = np.std(heat_flow[:20])  # Use start of region for noise
        signal = np.max(np.abs(heat_flow)) - np.min(np.abs(heat_flow))
        metrics["signal_quality"] = float(signal / noise if noise > 0 else 0)

        # Calculate reproducibility indicator
        # (based on smoothness of the transition)
        metrics["reproducibility"] = float(1 / (1 + np.std(np.gradient(d1))))

        return metrics
