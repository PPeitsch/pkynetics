"""Type definitions for DSC analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


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


@dataclass
class BaselineResult:
    """Container for baseline correction results."""

    baseline: NDArray[np.float64]
    corrected_data: NDArray[np.float64]
    method: str
    parameters: Dict
    quality_metrics: Dict
    regions: Optional[List[Tuple[float, float]]] = None


@dataclass
class GlassTransition:
    """Glass transition characteristics."""

    onset_temperature: float
    midpoint_temperature: float
    endpoint_temperature: float
    delta_cp: float  # Change in heat capacity
    width: float  # Temperature range of transition
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CrystallizationEvent:
    """Crystallization event characteristics."""

    onset_temperature: float
    peak_temperature: float
    endpoint_temperature: float
    enthalpy: float
    peak_height: float
    width: float
    crystallization_rate: Optional[float] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MeltingEvent:
    """Melting event characteristics."""

    onset_temperature: float
    peak_temperature: float
    endpoint_temperature: float
    enthalpy: float
    peak_height: float
    width: float
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PhaseTransition:
    """Generic phase transition characteristics."""

    transition_type: str  # 'first_order', 'second_order', etc.
    start_temperature: float
    peak_temperature: float
    end_temperature: float
    enthalpy: Optional[float] = None
    transition_width: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
