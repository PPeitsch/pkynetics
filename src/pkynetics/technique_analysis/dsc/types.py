"""Type definitions for DSC analysis."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class CpMethod(Enum):
    """Enumeration of heat capacity calculation methods."""

    THREE_STEP = "three_step"   # Method using three separate measurements (sample, reference, blank)
    SINGLE_STEP = "single_step" # Method using a single measurement (with or without calibration)
    MODULATED = "modulated"     # Method using temperature modulation (MDSC)


class OperationMode(Enum):
    """Measurement operation modes that can be applied to Cp methods."""

    CONTINUOUS = "continuous"  # Continuous heating/cooling
    STEPPED = "stepped"       # Step-wise heating/cooling with isothermal segments


class StabilityMethod(Enum):
    """Methods for detecting stable regions in stepped mode."""

    BASIC = "basic"          # Simple dT/dt threshold method
    STATISTICAL = "stats"    # Statistical analysis method
    CLUSTERING = "cluster"   # Clustering-based method


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


@dataclass
class CpResult:
    """Container for specific heat capacity results."""

    temperature: NDArray[np.float64]
    specific_heat: NDArray[np.float64]
    method: CpMethod
    uncertainty: NDArray[np.float64]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Union[float, str, NDArray[np.float64]]]
    operation_mode: OperationMode = field(default=OperationMode.CONTINUOUS)
    stable_regions: Optional[List[Tuple[int, int]]] = None


@dataclass
class CalibrationData:
    """Container for calibration data."""

    reference_material: str
    temperature: NDArray[np.float64]
    measured_cp: NDArray[np.float64]
    reference_cp: NDArray[np.float64]
    calibration_factors: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    valid_range: Tuple[float, float]
    timestamp: str = field(default_factory=lambda: np.datetime64("now").astype(str))
