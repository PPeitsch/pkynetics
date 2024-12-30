"""Core DSC analysis functionality."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .baseline import BaselineCorrector
from .peak_analysis import PeakAnalyzer
from .thermal_events import ThermalEventDetector


@dataclass
class DSCPeak:
    """Class to store DSC peak information."""

    onset_temperature: float
    peak_temperature: float
    endset_temperature: float
    enthalpy: float
    peak_height: float
    peak_width: float
    peak_area: float
    baseline_type: str
    baseline_params: Dict
    peak_indices: Tuple[int, int]  # Start and end indices


@dataclass
class DSCExperiment:
    """Class to hold DSC experiment data and metadata."""

    temperature: NDArray[np.float64]  # K
    heat_flow: NDArray[np.float64]  # mW
    time: NDArray[np.float64]  # s
    mass: float  # mg
    heating_rate: Optional[float] = None  # K/min
    name: str = "sample"
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Validate data and calculate heating rate if not provided."""
        if self.heating_rate is None:
            self.heating_rate = float(
                np.mean(np.gradient(self.temperature, self.time)) * 60
            )
        self.metadata = self.metadata or {}


class DSCAnalyzer:
    """Main class for DSC analysis."""

    def __init__(self,
                 experiment: DSCExperiment,
                 baseline_corrector: Optional['BaselineCorrector'] = None,
                 peak_analyzer: Optional['PeakAnalyzer'] = None,
                 event_detector: Optional['ThermalEventDetector'] = None):
        """Initialize DSC analyzer with experiment data."""
        self.experiment = experiment
        self.baseline_corrector = baseline_corrector or BaselineCorrector()
        self.peak_analyzer = peak_analyzer or PeakAnalyzer()
        self.event_detector = event_detector or ThermalEventDetector()

        self.baseline: Optional[NDArray[np.float64]] = None
        self.corrected_heat_flow: Optional[NDArray[np.float64]] = None
        self.peaks: List[DSCPeak] = []
        self.events: Dict = {}

    def analyze(self) -> Dict:
        """Perform complete DSC analysis."""
        # Correct baseline
        self.baseline, baseline_params = self.baseline_corrector.correct(
            self.experiment.temperature,
            self.experiment.heat_flow
        )
        self.corrected_heat_flow = self.experiment.heat_flow - self.baseline

        # Analyze peaks
        self.peaks = self.peak_analyzer.find_peaks(
            self.experiment.temperature,
            self.corrected_heat_flow
        )

        # Detect thermal events
        self.events = self.event_detector.detect_events(
            self.experiment.temperature,
            self.corrected_heat_flow,
            self.peaks
        )

        return {
            "peaks": self.peaks,
            "events": self.events,
            "baseline": {
                "type": baseline_params["type"],
                "parameters": baseline_params
            }
        }
