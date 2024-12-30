"""Thermal event detection and analysis for DSC data."""

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.optimize import curve_fit

from .types import CrystallizationEvent, GlassTransition, MeltingEvent, PhaseTransition


class ThermalEventDetector:
    """Class for detecting and analyzing thermal events in DSC data."""

    def __init__(
        self,
        smoothing_window: int = 21,
        smoothing_order: int = 3,
        peak_prominence: float = 0.1,
        noise_threshold: float = 0.05,
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
        # Calculate derivatives
        dT = np.gradient(temperature)
        dHf = np.gradient(heat_flow)
        d2Hf = np.gradient(dHf)

        # Smooth second derivative
        d2Hf_smooth = signal.savgol_filter(
            d2Hf, self.smoothing_window, self.smoothing_order
        )

        # Find inflection points
        inflection_points = signal.find_peaks(
            np.abs(d2Hf_smooth), prominence=self.peak_prominence
        )[0]

        if len(inflection_points) < 2:
            return None

        # Find the region with characteristic glass transition shape
        for i in range(len(inflection_points) - 1):
            start_idx = inflection_points[i]
            end_idx = inflection_points[i + 1]

            # Check if region has characteristic sigmoid shape
            region = heat_flow[start_idx:end_idx]
            if self._is_glass_transition_shape(region):
                # Calculate glass transition parameters
                onset_temp = temperature[start_idx]
                end_temp = temperature[end_idx]
                mid_temp = temperature[
                    start_idx + np.argmax(np.abs(dHf[start_idx:end_idx]))
                ]

                # Calculate change in heat capacity
                if baseline is not None:
                    delta_cp = self._calculate_delta_cp(
                        temperature[start_idx:end_idx],
                        heat_flow[start_idx:end_idx],
                        baseline[start_idx:end_idx],
                    )
                else:
                    delta_cp = np.nan

                return GlassTransition(
                    onset_temperature=float(onset_temp),
                    midpoint_temperature=float(mid_temp),
                    endpoint_temperature=float(end_temp),
                    delta_cp=float(delta_cp),
                    width=float(end_temp - onset_temp),
                    quality_metrics=self._calculate_gt_quality(
                        temperature[start_idx:end_idx],
                        heat_flow[start_idx:end_idx],
                        d2Hf_smooth[start_idx:end_idx],
                    ),
                )

        return None

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
        # Adjust heat flow direction (crystallization is exothermic)
        heat_flow_adj = -heat_flow if np.mean(heat_flow) > 0 else heat_flow

        # Find peaks
        peaks, properties = signal.find_peaks(
            heat_flow_adj, prominence=self.peak_prominence
        )

        events = []
        for i, peak_idx in enumerate(peaks):
            # Find onset and endpoint
            onset_idx = self._find_onset_index(temperature, heat_flow_adj, peak_idx)
            end_idx = self._find_endpoint_index(temperature, heat_flow_adj, peak_idx)

            # Calculate baseline-corrected heat flow
            if baseline is not None:
                heat_flow_corr = heat_flow_adj - baseline
            else:
                heat_flow_corr = heat_flow_adj

            # Calculate enthalpy
            enthalpy = self._calculate_peak_enthalpy(
                temperature[onset_idx : end_idx + 1],
                heat_flow_corr[onset_idx : end_idx + 1],
            )

            # Calculate crystallization rate
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
                    peak_height=float(properties["prominences"][i]),
                    width=float(temperature[end_idx] - temperature[onset_idx]),
                    crystallization_rate=float(rate) if rate is not None else None,
                    quality_metrics=self._calculate_peak_quality(
                        temperature[onset_idx : end_idx + 1],
                        heat_flow_corr[onset_idx : end_idx + 1],
                    ),
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
        # Find endothermic peaks
        peaks, properties = signal.find_peaks(
            heat_flow, prominence=self.peak_prominence
        )

        events = []
        for i, peak_idx in enumerate(peaks):
            # Find onset and endpoint
            onset_idx = self._find_onset_index(temperature, heat_flow, peak_idx)
            end_idx = self._find_endpoint_index(temperature, heat_flow, peak_idx)

            # Calculate baseline-corrected heat flow
            if baseline is not None:
                heat_flow_corr = heat_flow - baseline
            else:
                heat_flow_corr = heat_flow

            # Calculate enthalpy
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
        # Calculate derivatives
        d1 = np.gradient(heat_flow, temperature)
        d2 = np.gradient(d1, temperature)

        # Smooth derivatives
        d1_smooth = signal.savgol_filter(
            d1, self.smoothing_window, self.smoothing_order
        )
        d2_smooth = signal.savgol_filter(
            d2, self.smoothing_window, self.smoothing_order
        )

        transitions = []

        # Find first-order transitions (peaks in heat flow)
        peaks, _ = signal.find_peaks(np.abs(heat_flow), prominence=self.peak_prominence)
        for peak_idx in peaks:
            # Find transition boundaries
            start_idx = self._find_onset_index(temperature, heat_flow, peak_idx)
            end_idx = self._find_endpoint_index(temperature, heat_flow, peak_idx)

            # Calculate enthalpy if baseline is provided
            if baseline is not None:
                heat_flow_corr = heat_flow - baseline
                enthalpy = self._calculate_peak_enthalpy(
                    temperature[start_idx : end_idx + 1],
                    heat_flow_corr[start_idx : end_idx + 1],
                )
            else:
                enthalpy = None

            transitions.append(
                PhaseTransition(
                    transition_type="first_order",
                    start_temperature=float(temperature[start_idx]),
                    peak_temperature=float(temperature[peak_idx]),
                    end_temperature=float(temperature[end_idx]),
                    enthalpy=float(enthalpy) if enthalpy is not None else None,
                    transition_width=float(
                        temperature[end_idx] - temperature[start_idx]
                    ),
                    quality_metrics=self._calculate_transition_quality(
                        temperature[start_idx : end_idx + 1],
                        heat_flow[start_idx : end_idx + 1],
                        d1_smooth[start_idx : end_idx + 1],
                        d2_smooth[start_idx : end_idx + 1],
                    ),
                )
            )

        # Find second-order transitions (steps in heat flow)
        steps = signal.find_peaks(np.abs(d1_smooth), prominence=self.peak_prominence)[0]
        for step_idx in steps:
            if not any(
                abs(step_idx - peak_idx) < len(temperature) // 20 for peak_idx in peaks
            ):
                start_idx = max(0, step_idx - len(temperature) // 40)
                end_idx = min(len(temperature) - 1, step_idx + len(temperature) // 40)

                transitions.append(
                    PhaseTransition(
                        transition_type="second_order",
                        start_temperature=float(temperature[start_idx]),
                        peak_temperature=float(temperature[step_idx]),
                        end_temperature=float(temperature[end_idx]),
                        transition_width=float(
                            temperature[end_idx] - temperature[start_idx]
                        ),
                        quality_metrics=self._calculate_transition_quality(
                            temperature[start_idx : end_idx + 1],
                            heat_flow[start_idx : end_idx + 1],
                            d1_smooth[start_idx : end_idx + 1],
                            d2_smooth[start_idx : end_idx + 1],
                        ),
                    )
                )

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
    ) -> float:
        """Calculate change in heat capacity across glass transition."""
        # Use first and last 20% of points to calculate pre and post Cp
        n_points = len(temperature) // 5

        pre_cp = np.mean(heat_flow[:n_points] - baseline[:n_points])
        post_cp = np.mean(heat_flow[-n_points:] - baseline[-n_points:])

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
        self, temperature: NDArray[np.float64], heat_flow: NDArray[np.float64]
    ) -> Dict[str, float]:
        """Calculate quality metrics for peak events."""
        metrics = {}

        # Calculate peak sharpness
        peak_idx = np.argmax(np.abs(heat_flow))
        left_half = heat_flow[:peak_idx]
        right_half = heat_flow[peak_idx:]

        if len(left_half) > 0 and len(right_half) > 0:
            metrics["sharpness"] = float(
                np.max(np.gradient(left_half)) - np.min(np.gradient(right_half))
            )
        else:
            metrics["sharpness"] = 0.0

        # Calculate peak-to-noise ratio
        noise = np.std(heat_flow[:20])  # Use start of region for noise
        signal = np.max(np.abs(heat_flow))
        metrics["peak_to_noise"] = float(signal / noise if noise > 0 else 0)

        # Calculate baseline stability
        metrics["baseline_stability"] = float(
            1 / (1 + np.std(heat_flow[:20]))  # Use start of region
        )

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
