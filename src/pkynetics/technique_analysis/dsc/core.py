"""Core DSC analysis functionality."""

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .baseline import BaselineCorrector
from .peak_analysis import PeakAnalyzer
from .thermal_events import ThermalEventDetector
from .types import BaselineResult, DSCExperiment, DSCPeak


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

    def analyze(self, baseline_method: str = "auto", **baseline_kwargs: Any) -> Dict:
        """
        Perform complete DSC analysis.

        Args:
            baseline_method: Baseline correction method (see BaselineCorrector)
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
            The baseline is fitted to the whole curve. A glass transition
            (a step in the heat flow) lies on the baseline itself: a single
            linear or polynomial fit across it distorts the step. Restrict
            the data or pass baseline regions on one side of it, or use
            ThermalEventDetector on the raw curve, to characterize a Tg.
        """
        exp = self.experiment
        heating_rate = abs(exp.heating_rate) if exp.heating_rate else None

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

        return {
            "peaks": self.peaks,
            "events": self.events,
            "baseline": {
                "type": self.baseline_result.method,
                "parameters": self.baseline_result.parameters,
                "quality_metrics": self.baseline_result.quality_metrics,
            },
        }
