"""Core DSC analysis functionality."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .baseline import BaselineCorrector
from .peak_analysis import PeakAnalyzer
from .thermal_events import ThermalEventDetector
from .types import BaselineResult, DSCExperiment, DSCPeak, GlassTransition


def _step_region(
    transition: Optional[GlassTransition],
) -> Optional[Tuple[float, float]]:
    """The (onset, endpoint) range of a transition, if it is usable."""
    if transition is None:
        return None
    onset = float(transition.onset_temperature)
    endpoint = float(transition.endpoint_temperature)
    if not np.isfinite(onset) or not np.isfinite(endpoint) or endpoint <= onset:
        return None
    return (onset, endpoint)


class DSCAnalyzer:
    """Complete DSC analysis: baseline correction, peaks and thermal events."""

    def __init__(
        self,
        experiment: DSCExperiment,
        baseline_corrector: Optional[BaselineCorrector] = None,
        peak_analyzer: Optional[PeakAnalyzer] = None,
        event_detector: Optional[ThermalEventDetector] = None,
    ):
        """
        Initialize DSC analyzer with experiment data.

        Args:
            experiment: DSC experiment (temperature in K, heat flow in mW,
                time in s, mass in mg, heating rate in K/min)
            baseline_corrector: Baseline corrector (default settings if None)
            peak_analyzer: Peak analyzer (default settings if None)
            event_detector: Thermal event detector (default settings, i.e.
                endothermic up, if None)
        """
        self.experiment = experiment
        self.baseline_corrector = baseline_corrector or BaselineCorrector()
        self.peak_analyzer = peak_analyzer or PeakAnalyzer()
        self.event_detector = event_detector or ThermalEventDetector()

        self.baseline_result: Optional[BaselineResult] = None
        self.baseline: Optional[NDArray[np.float64]] = None
        self.corrected_heat_flow: Optional[NDArray[np.float64]] = None
        self.peaks: List[DSCPeak] = []
        self.events: Dict[str, List[Any]] = {}
        self.glass_transition: Optional[GlassTransition] = None

    def analyze(
        self,
        baseline_method: str = "auto",
        detect_steps: bool = True,
        **baseline_kwargs: Any,
    ) -> Dict:
        """
        Perform complete DSC analysis.

        Args:
            baseline_method: Baseline correction method (see BaselineCorrector)
            detect_steps: Look for a glass transition on the raw curve first
                and fit the baseline on each side of it (see Note). Pass
                False, or an explicit ``step_regions``, to skip the search
            **baseline_kwargs: Additional baseline parameters, e.g. regions

        Returns:
            Dictionary with 'peaks' (endothermic and exothermic peaks of the
            corrected signal, by temperature, with type 'endothermic' or
            'exothermic' and enthalpy magnitudes), 'events' (thermal events
            by type) and 'baseline' (method, parameters and quality
            metrics). Enthalpies are in J/g, using the experiment's heating
            rate and mass. The sign convention is the event detector's
            (exo_up).

        Note:
            A glass transition is a step in the heat flow: the sample stays at
            a new level afterwards. One fit across it passes through the
            middle of the step, which leaves a residual bump (detected as a
            spurious peak) and biases the enthalpies of the real peaks. The
            transition is therefore located on the raw curve first, and the
            baseline is fitted on each side of it and joined across it, as
            the standards draw it. ``step_regions`` overrides the search and
            ``detect_steps=False`` disables it.
        """
        exp = self.experiment
        heating_rate = abs(exp.heating_rate) if exp.heating_rate else None

        self.glass_transition = None
        if detect_steps and "step_regions" not in baseline_kwargs:
            self.glass_transition = self._detect_glass_transition(
                heating_rate, exp.mass
            )
            step = _step_region(self.glass_transition)
            if step is not None:
                baseline_kwargs["step_regions"] = [step]

        self.baseline_result = self.baseline_corrector.correct(
            exp.temperature, exp.heat_flow, method=baseline_method, **baseline_kwargs
        )
        self.baseline = self.baseline_result.baseline
        self.corrected_heat_flow = exp.heat_flow - self.baseline

        # Peaks point up in find_peaks: search both orientations
        endo_up = (
            -self.corrected_heat_flow
            if self.event_detector.exo_up
            else self.corrected_heat_flow
        )
        self.peaks = []
        for kind, oriented in (("endothermic", endo_up), ("exothermic", -endo_up)):
            for peak in self.peak_analyzer.find_peaks(
                exp.temperature,
                oriented,
                heating_rate=heating_rate,
                sample_mass=exp.mass,
            ):
                peak.type = kind
                self.peaks.append(peak)
        self.peaks.sort(key=lambda peak: peak.peak_temperature)

        self.events = self.event_detector.detect_events(
            exp.temperature,
            exp.heat_flow,
            baseline=self.baseline,
            heating_rate=heating_rate,
            sample_mass=exp.mass,
        )
        # A stepped baseline removes the step it was fitted around, so the
        # detector cannot find it again: report the transition measured on
        # the raw curve, with its own local baseline
        if self.glass_transition is not None and not self.events["glass_transitions"]:
            self.events["glass_transitions"] = [self.glass_transition]

        return {
            "peaks": self.peaks,
            "events": self.events,
            "baseline": {
                "type": self.baseline_result.method,
                "parameters": self.baseline_result.parameters,
                "quality_metrics": self.baseline_result.quality_metrics,
            },
        }

    def _detect_glass_transition(
        self, heating_rate: Optional[float], sample_mass: Optional[float]
    ) -> Optional[GlassTransition]:
        """
        Glass transition measured on the raw curve.

        The detector makes its own local baselines per event, so it sees the
        step before any whole-curve baseline has distorted it.
        """
        exp = self.experiment
        return self.event_detector.detect_glass_transition(
            exp.temperature,
            exp.heat_flow,
            heating_rate=heating_rate,
            sample_mass=sample_mass,
        )
