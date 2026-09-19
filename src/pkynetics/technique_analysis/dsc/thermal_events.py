"""Thermal event detection and analysis for DSC data.

Sign convention: by default endothermic events point up (exo_up=False).
Enthalpies are positive for endothermic and negative for exothermic events,
in J/g when the heating rate and sample mass are given (NaN otherwise).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.optimize import curve_fit

from .peak_analysis import PeakAnalyzer
from .types import (
    CrystallizationEvent,
    DSCPeak,
    GlassTransition,
    MeltingEvent,
    PhaseTransition,
)
from .utilities import safe_savgol_filter

logger = logging.getLogger(__name__)


def _logistic(
    x: NDArray[np.float64], a: float, b: float, c: float, s: float
) -> NDArray[np.float64]:
    """Sigmoidal step from a to a + b centred at c with width parameter s."""
    result: NDArray[np.float64] = a + b / (1 + np.exp(-np.clip((x - c) / s, -500, 500)))
    return result


class ThermalEventDetector:
    """Class for detecting and analyzing thermal events in DSC data."""

    def __init__(
        self,
        smoothing_window: int = 21,
        smoothing_order: int = 3,
        peak_prominence: float = 0.1,
        noise_threshold: float = 5.0,
        exo_up: bool = False,
    ):
        """
        Initialize thermal event detector.

        Args:
            smoothing_window: Window size for Savitzky-Golay smoothing
            smoothing_order: Order for Savitzky-Golay filter
            peak_prominence: Minimum peak prominence, in heat flow units
            noise_threshold: Minimum peak prominence as a multiple of the
                estimated noise level (the larger of both limits applies)
            exo_up: True if exothermic events point up in the data
        """
        self.smoothing_window = smoothing_window
        self.smoothing_order = smoothing_order
        self.peak_prominence = peak_prominence
        self.noise_threshold = noise_threshold
        self.exo_up = exo_up
        self.limit_noise_factor = 3.0
        self.noisy_derivative_fraction = 30
        self.peak_analyzer = PeakAnalyzer(
            smoothing_window=smoothing_window, smoothing_order=smoothing_order
        )

    def detect_events(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        peaks: Optional[List[DSCPeak]] = None,
        baseline: Optional[NDArray[np.float64]] = None,
        heating_rate: Optional[float] = None,
        sample_mass: Optional[float] = None,
    ) -> Dict[str, List[Any]]:
        """
        Detect and analyze all thermal events.

        Args:
            temperature: Temperature array in K
            heat_flow: Heat flow array in mW
            peaks: Previously detected peaks (unused, kept for compatibility)
            baseline: Optional baseline array
            heating_rate: Heating rate in K/min, for enthalpies and delta Cp
            sample_mass: Sample mass in mg, for enthalpies and delta Cp

        Returns:
            Dictionary with lists of glass_transitions, crystallization,
            melting and phase_transitions
        """
        kwargs: Dict[str, Any] = {
            "baseline": baseline,
            "heating_rate": heating_rate,
            "sample_mass": sample_mass,
        }
        gt = self.detect_glass_transition(temperature, heat_flow, **kwargs)
        return {
            "glass_transitions": [gt] if gt is not None else [],
            "crystallization": self.detect_crystallization(
                temperature, heat_flow, **kwargs
            ),
            "melting": self.detect_melting(temperature, heat_flow, **kwargs),
            "phase_transitions": self.detect_phase_transitions(
                temperature, heat_flow, **kwargs
            ),
        }

    def detect_glass_transition(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
        heating_rate: Optional[float] = None,
        sample_mass: Optional[float] = None,
    ) -> Optional[GlassTransition]:
        """
        Detect and analyze the most pronounced glass transition.

        The glass transition is a step in the endothermic direction.
        Characteristic temperatures follow ISO 11357-2: onset and endpoint
        are the intersections of the inflectional tangent with the extrapolated
        pre- and post-transition baselines, the midpoint is at half step height.

        Args:
            temperature: Temperature array in K
            heat_flow: Heat flow array in mW
            baseline: Optional baseline array
            heating_rate: Heating rate in K/min, needed for delta Cp
            sample_mass: Sample mass in mg, needed for delta Cp

        Returns:
            GlassTransition object if detected, None otherwise. delta_cp is in
            J/(g*K), NaN without heating rate and sample mass.
        """
        self._validate(temperature, heat_flow, baseline)
        y = self._endo_up(heat_flow, baseline)

        steps = self._find_steps(temperature, y)
        if not steps:
            return None

        step = steps[0]
        return GlassTransition(
            onset_temperature=step["onset"],
            midpoint_temperature=step["midpoint"],
            endpoint_temperature=step["endpoint"],
            delta_cp=self._specific(step["height"], heating_rate, sample_mass),
            width=step["endpoint"] - step["onset"],
            quality_metrics=step["quality"],
            baseline_subtracted=baseline is not None,
        )

    def detect_crystallization(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
        heating_rate: Optional[float] = None,
        sample_mass: Optional[float] = None,
    ) -> List[CrystallizationEvent]:
        """
        Detect and analyze crystallization (exothermic) peaks.

        Args:
            temperature: Temperature array in K
            heat_flow: Heat flow array in mW
            baseline: Optional baseline array
            heating_rate: Heating rate in K/min, for enthalpy and rate
            sample_mass: Sample mass in mg, for enthalpy

        Returns:
            List of CrystallizationEvent objects, ordered by temperature. The
            crystallization rate is the maximum of d(alpha)/dt in 1/s (None
            without heating rate).
        """
        events = []
        for peak, quality in self._find_peaks(
            temperature, heat_flow, baseline, heating_rate, sample_mass, endo=False
        ):
            rate = None
            if heating_rate and peak.peak_area > 0:
                # max d(alpha)/dT = height / area; d(alpha)/dt = that * beta
                rate = peak.peak_height / peak.peak_area * heating_rate / 60
            events.append(
                CrystallizationEvent(
                    onset_temperature=peak.onset_temperature,
                    peak_temperature=peak.peak_temperature,
                    endpoint_temperature=peak.endset_temperature,
                    enthalpy=-peak.enthalpy,
                    peak_height=peak.peak_height,
                    width=peak.peak_width,
                    crystallization_rate=rate,
                    quality_metrics=quality,
                    baseline_subtracted=baseline is not None,
                )
            )
        return events

    def detect_melting(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
        heating_rate: Optional[float] = None,
        sample_mass: Optional[float] = None,
    ) -> List[MeltingEvent]:
        """
        Detect and analyze melting (endothermic) peaks.

        Args:
            temperature: Temperature array in K
            heat_flow: Heat flow array in mW
            baseline: Optional baseline array
            heating_rate: Heating rate in K/min, for enthalpy
            sample_mass: Sample mass in mg, for enthalpy

        Returns:
            List of MeltingEvent objects, ordered by temperature
        """
        return [
            MeltingEvent(
                onset_temperature=peak.onset_temperature,
                peak_temperature=peak.peak_temperature,
                endpoint_temperature=peak.endset_temperature,
                enthalpy=peak.enthalpy,
                peak_height=peak.peak_height,
                width=peak.peak_width,
                quality_metrics=quality,
                baseline_subtracted=baseline is not None,
            )
            for peak, quality in self._find_peaks(
                temperature, heat_flow, baseline, heating_rate, sample_mass, endo=True
            )
        ]

    def detect_phase_transitions(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
        heating_rate: Optional[float] = None,
        sample_mass: Optional[float] = None,
    ) -> List[PhaseTransition]:
        """
        Detect first-order (peaks) and second-order (steps) transitions.

        Args:
            temperature: Temperature array in K
            heat_flow: Heat flow array in mW
            baseline: Optional baseline array
            heating_rate: Heating rate in K/min, for enthalpies
            sample_mass: Sample mass in mg, for enthalpies

        Returns:
            List of PhaseTransition objects: first-order transitions ordered
            by temperature, then second-order transitions ordered by
            temperature. First-order enthalpies are signed (endo > 0).
        """
        self._validate(temperature, heat_flow, baseline)

        first_order = []
        for endo in (True, False):
            sign = 1.0 if endo else -1.0
            for peak, quality in self._find_peaks(
                temperature, heat_flow, baseline, heating_rate, sample_mass, endo
            ):
                first_order.append(
                    PhaseTransition(
                        transition_type="first_order",
                        start_temperature=peak.onset_temperature,
                        peak_temperature=peak.peak_temperature,
                        end_temperature=peak.endset_temperature,
                        enthalpy=sign * peak.enthalpy,
                        transition_width=peak.endset_temperature
                        - peak.onset_temperature,
                        quality_metrics=quality,
                        baseline_subtracted=baseline is not None,
                    )
                )

        second_order = [
            PhaseTransition(
                transition_type="second_order",
                start_temperature=step["onset"],
                peak_temperature=step["midpoint"],
                end_temperature=step["endpoint"],
                transition_width=step["endpoint"] - step["onset"],
                quality_metrics=step["quality"],
                baseline_subtracted=baseline is not None,
            )
            for step in self._find_steps(
                temperature, self._endo_up(heat_flow, baseline)
            )
        ]

        first_order.sort(key=lambda t: t.peak_temperature)
        second_order.sort(key=lambda t: t.peak_temperature)
        return first_order + second_order

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Validate input array lengths."""
        if len(temperature) != len(heat_flow):
            raise ValueError("Temperature and heat flow arrays must have same length")
        if baseline is not None and len(baseline) != len(heat_flow):
            raise ValueError("Baseline must have the same length as heat flow")
        if len(temperature) < 3:
            raise ValueError("At least 3 data points are required")

    def _endo_up(
        self,
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Baseline-corrected heat flow with endothermic direction up."""
        corrected = heat_flow if baseline is None else heat_flow - baseline
        return -corrected if self.exo_up else corrected

    @staticmethod
    def _specific(
        value: float, heating_rate: Optional[float], sample_mass: Optional[float]
    ) -> float:
        """Convert a heat flow quantity (mW, mW*K) per K/s and mg (NaN if
        heating rate or mass are missing)."""
        if not heating_rate or not sample_mass:
            return float("nan")
        return float(value / (heating_rate / 60) / sample_mass)

    def _noise_level(self, y: NDArray[np.float64]) -> float:
        """Estimate noise as the residual of Savitzky-Golay smoothing."""
        return self.peak_analyzer.noise_level(y)

    def _find_peaks(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        baseline: Optional[NDArray[np.float64]],
        heating_rate: Optional[float],
        sample_mass: Optional[float],
        endo: bool,
    ) -> List[Tuple[DSCPeak, Dict[str, float]]]:
        """
        Find peaks in the endothermic or exothermic direction.

        Detected steps (glass / second-order transitions) are subtracted
        first: they change the heat capacity, i.e. the baseline, and the
        plateau between a step and a peak must not be taken as a peak.
        Integration limits are where the peak has decayed to 0.1% of its
        prominence (or to limit_noise_factor times the smoothed noise, if
        higher). Without a baseline the peak is integrated against its
        reference base (horizontal); pass a baseline for sloped data. Without a baseline, a linear baseline between the limits
        is used, so peaks sitting on slopes are measured correctly.
        """
        self._validate(temperature, heat_flow, baseline)
        y = self._endo_up(heat_flow, baseline)
        for step in self._find_steps(temperature, y):
            y = y - _logistic(temperature, 0.0, *step["fit"])
        if not endo:
            y = -y

        smooth = safe_savgol_filter(y, self.smoothing_window, self.smoothing_order)
        noise = self._noise_level(y)
        prominence = max(self.peak_prominence, self.noise_threshold * noise)

        idx, props = signal.find_peaks(smooth, prominence=prominence)
        if len(idx) == 0:
            return []
        self.peak_analyzer.limit_noise_factor = self.limit_noise_factor
        limits = self.peak_analyzer.integration_limits(smooth, idx, props, noise)

        results = []
        for j, (peak_idx, (lo, hi)) in enumerate(zip(idx, limits)):
            prom = float(props["prominences"][j])
            if hi - lo < 3:
                continue

            t_region = temperature[lo:hi]
            y_region = y[lo:hi]
            # Integrate against the peak's reference base (apex - prominence):
            # the limits lie `level` above it, so the signal at the limits
            # must not be used as the baseline. With an explicit baseline the
            # corrected signal is already referenced to zero.
            if baseline is None:
                local = np.full_like(y_region, smooth[peak_idx] - prom)
            else:
                local = np.zeros_like(y_region)

            peak = self.peak_analyzer.analyze_peak_region(
                t_region,
                y_region,
                int(peak_idx) - lo,
                local,
                heating_rate=heating_rate,
                sample_mass=sample_mass,
            )
            results.append((peak, self._peak_quality(peak, noise)))

        return results

    @staticmethod
    def _peak_quality(peak: DSCPeak, noise: float) -> Dict[str, float]:
        """Quality metrics for a peak (all positive, larger is better)."""
        height = abs(peak.peak_height)
        noise = max(noise, float(np.finfo(float).eps) * max(height, 1.0))
        return {
            "peak_to_noise": float(height / noise),
            "sharpness": float(height / peak.peak_width) if peak.peak_width else 0.0,
            "baseline_stability": (
                float(1.0 / (1.0 + noise / height)) if height else 0.0
            ),
        }

    def _find_steps(
        self, temperature: NDArray[np.float64], y: NDArray[np.float64]
    ) -> List[Dict[str, Any]]:
        """
        Find upward sigmoidal steps (glass / second-order transitions).

        Candidates are maxima of dy/dT. A logistic step is fitted around each
        one, bounded by neighbouring events (clearly negative slope); peak
        flanks and noise do not fit a step and are rejected. Returns steps
        sorted by height, largest first.
        """
        n = len(y)
        noise = self._noise_level(y)

        # Derivatives amplify noise: smooth more strongly for noisy data
        window = self.smoothing_window
        if noise >= 1e-3 * np.ptp(y):
            window = max(window, n // self.noisy_derivative_fraction)
        smooth = safe_savgol_filter(y, window, self.smoothing_order)
        d1 = np.gradient(smooth, temperature)

        candidates, props = signal.find_peaks(d1, prominence=0)
        candidates = candidates[np.argsort(props["prominences"])[::-1]][:10]

        steps: List[Dict[str, Any]] = []
        for i in candidates:
            k = max(int(signal.peak_widths(d1, [i], rel_height=0.5)[0][0]), 3)

            # Up to 2k each side (plateaus), stopping at a neighbouring event:
            # a clearly negative slope sustained over k/4 samples (noise on
            # the plateaus crosses the threshold only briefly)
            falling = (d1 < -0.1 * d1[i]).astype(float)
            run = max(3, k // 4)
            sustained = (
                np.convolve(falling, np.ones(run), mode="full")[run - 1 :] >= run
            )
            stop_right = np.flatnonzero(sustained[i:])  # run starts at index
            stop_left = np.flatnonzero(
                np.convolve(falling, np.ones(run), mode="full")[:n][: i + 1][::-1]
                >= run
            )
            right = min(2 * k, stop_right[0] if len(stop_right) else n - 1 - i)
            left = min(2 * k, stop_left[0] if len(stop_left) else i)
            if left < 0.8 * k or right < 0.8 * k:
                continue  # plateaus not (sufficiently) within the data

            lo, hi = i - left, i + right + 1
            x, yy = temperature[lo:hi], y[lo:hi]
            s0 = k * float(np.mean(np.diff(x))) / 3.5
            try:
                popt, _ = curve_fit(
                    _logistic,
                    x,
                    yy,
                    p0=[smooth[lo], smooth[hi - 1] - smooth[lo], temperature[i], s0],
                    maxfev=5000,
                )
            except RuntimeError:
                continue

            a, b, c, s = (float(v) for v in popt)
            residual = yy - _logistic(x, *popt)
            rms = float(np.sqrt(np.mean(residual**2)))
            # Accept only complete steps: both tangent intersections inside
            # the fitted data, followed by a plateau of at least s on each
            # side (a peak flank turns back at the apex without a plateau),
            # wider than a few samples (not a noise jump),
            # clearly above the noise and well described by the model
            spacing = float(np.mean(np.diff(x)))
            if not (
                b > max(2 * noise, 0.01 * float(np.ptp(smooth)))
                and 4 * s > 3 * spacing
                and x[0] + s <= c - 2 * s
                and c + 2 * s <= x[-1] - s
                and rms < max(1.5 * noise, 0.03 * b)
            ):
                continue

            # A step changes the level for good: away from the step the data
            # must stay on the respective side of the half-step level. A peak
            # flank fails this, since the signal returns to where it started
            # (the median ignores intervening peaks, e.g. cold crystallization)
            half_level = a + b / 2
            before, after = smooth[:lo], smooth[hi:]
            if len(after) >= k and np.median(after) < half_level:
                continue
            if len(before) >= k and np.median(before) > half_level:
                continue

            # Tangent construction on the fitted step: the inflectional
            # tangent (slope b / 4s at c) meets the plateaus at c -/+ 2s
            r2 = 1 - np.sum(residual**2) / np.sum((yy - yy.mean()) ** 2)
            steps.append(
                {
                    "onset": c - 2 * s,
                    "midpoint": c,
                    "endpoint": c + 2 * s,
                    "height": b,
                    "fit": (b, c, s),
                    "quality": {
                        "snr": b / max(noise, float(np.finfo(float).eps) * b),
                        "fit_r2": float(max(r2, 0.0)),
                    },
                }
            )

        # Different candidates can converge on the same step
        unique: List[Dict[str, Any]] = []
        for step in sorted(steps, key=lambda st: st["height"], reverse=True):
            if all(
                abs(step["midpoint"] - u["midpoint"]) > (u["endpoint"] - u["onset"]) / 2
                for u in unique
            ):
                unique.append(step)
        return unique
