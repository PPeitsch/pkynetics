"""Type definitions for DSC analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class CpMethod(Enum):
    """Heat capacity calculation methods."""

    THREE_STEP = "three_step"  # Three separate measurements (sample, reference, blank)
    SINGLE_STEP = "single_step"  # Single measurement with calibration curve
    MODULATED = "modulated"  # Temperature modulation method (MDSC)


class OperationMode(Enum):
    """Measurement operation modes."""

    CONTINUOUS = "continuous"  # Continuous heating/cooling
    STEPPED = "stepped"  # Step-wise heating/cooling with isothermal segments


class StabilityMethod(Enum):
    """Methods for detecting stable regions."""

    BASIC = "basic"  # Simple dT/dt and signal thresholds
    STATISTICAL = "statistical"  # Statistical analysis of signal
    LINEAR_FIT = "linear_fit"  # Linear regression analysis
    ADAPTIVE = "adaptive"  # Adaptive segmentation
    WAVELET = "wavelet"  # Wavelet-based analysis


@dataclass
class DSCPeak:
    """DSC peak information."""

    onset_temperature: float  # K
    peak_temperature: float  # K
    endset_temperature: float  # K
    enthalpy: float  # J/g
    peak_height: float  # mW/mg
    peak_width: float  # K
    peak_area: float  # mJ/mg
    baseline_type: str
    baseline_params: Dict
    peak_indices: Tuple[int, int]
    type: Optional[str] = None  # 'melting', 'crystallization', 'transition'
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DSCExperiment:
    """DSC experiment data and metadata."""

    temperature: NDArray[np.float64]  # K
    heat_flow: NDArray[np.float64]  # mW
    time: NDArray[np.float64]  # s
    mass: float  # mg
    heating_rate: Optional[float] = None  # K/min
    reference_mass: Optional[float] = None  # mg
    reference_material: Optional[str] = None
    sample_name: str = "sample"
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate data and calculate heating rate."""
        if not isinstance(self.temperature, np.ndarray):
            self.temperature = np.array(self.temperature, dtype=np.float64)
        if not isinstance(self.heat_flow, np.ndarray):
            self.heat_flow = np.array(self.heat_flow, dtype=np.float64)
        if not isinstance(self.time, np.ndarray):
            self.time = np.array(self.time, dtype=np.float64)

        # Calculate heating rate if not provided
        if self.heating_rate is None:
            self.heating_rate = float(
                np.mean(np.gradient(self.temperature, self.time)) * 60  # K/min
            )

        # Validate data
        if len(self.temperature) != len(self.heat_flow) or len(self.temperature) != len(self.time):
            raise ValueError("Temperature, heat flow, and time arrays must have the same length")
        if self.mass < 0:
            raise ValueError("Sample mass must be positive")
        if self.heating_rate == 0:
            raise ValueError("Heating rate cannot be zero")


@dataclass
class BaselineResult:
    """Baseline correction results."""

    baseline: NDArray[np.float64]  # mW
    corrected_data: NDArray[np.float64]  # mW
    method: str
    parameters: Dict
    quality_metrics: Dict[str, float]
    regions: Optional[List[Tuple[float, float]]] = None  # K


@dataclass
class GlassTransition:
    """Glass transition characteristics."""

    onset_temperature: float  # K
    midpoint_temperature: float  # K
    endpoint_temperature: float  # K
    delta_cp: float  # J/(g·K)
    width: float  # K
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_subtracted: bool = False


@dataclass
class CrystallizationEvent:
    """Crystallization event characteristics."""

    onset_temperature: float  # K
    peak_temperature: float  # K
    endpoint_temperature: float  # K
    enthalpy: float  # J/g
    peak_height: float  # mW/mg
    width: float  # K
    crystallization_rate: Optional[float] = None  # 1/s
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_subtracted: bool = False


@dataclass
class MeltingEvent:
    """Melting event characteristics."""

    onset_temperature: float  # K
    peak_temperature: float  # K
    endpoint_temperature: float  # K
    enthalpy: float  # J/g
    peak_height: float  # mW/mg
    width: float  # K
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_subtracted: bool = False


@dataclass
class PhaseTransition:
    """Generic phase transition characteristics."""

    transition_type: str  # 'first_order', 'second_order', 'glass', etc.
    start_temperature: float  # K
    peak_temperature: float  # K
    end_temperature: float  # K
    enthalpy: Optional[float] = None  # J/g
    transition_width: float = 0.0  # K
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_subtracted: bool = False


@dataclass
class CpResult:
    """Specific heat capacity results."""

    temperature: NDArray[np.float64]  # K
    specific_heat: NDArray[np.float64]  # J/(g·K)
    uncertainty: NDArray[np.float64]  # J/(g·K)
    method: CpMethod
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Union[float, str, NDArray[np.float64]]]
    operation_mode: OperationMode = field(default=OperationMode.CONTINUOUS)
    stable_regions: Optional[List[Tuple[int, int]]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate arrays and calculations."""
        arrays = [self.temperature, self.specific_heat, self.uncertainty]
        if not all(isinstance(arr, np.ndarray) for arr in arrays):
            raise TypeError("temperature, specific_heat, and uncertainty must be numpy arrays")
        if not all(len(arr) == len(self.temperature) for arr in arrays):
            raise ValueError("All arrays must have the same length")
        if not all(np.all(np.isfinite(arr)) for arr in arrays):
            raise ValueError("Arrays contain invalid values (inf or nan)")


@dataclass
class CalibrationData:
    """Calibration data and parameters."""

    reference_material: str
    temperature: NDArray[np.float64]  # K
    measured_cp: NDArray[np.float64]  # J/(g·K)
    reference_cp: NDArray[np.float64]  # J/(g·K)
    calibration_factors: NDArray[np.float64]
    uncertainty: NDArray[np.float64]  # J/(g·K)
    valid_range: Tuple[float, float]  # K
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate calibration data."""
        arrays = [self.temperature, self.measured_cp, self.reference_cp,
                  self.calibration_factors, self.uncertainty]
        if not all(isinstance(arr, np.ndarray) for arr in arrays):
            raise TypeError("All data arrays must be numpy arrays")
        if not all(len(arr) == len(self.temperature) for arr in arrays):
            raise ValueError("All arrays must have the same length")
        if not self.valid_range[0] < self.valid_range[1]:
            raise ValueError("Invalid temperature range")
