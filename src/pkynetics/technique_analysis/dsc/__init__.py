"""DSC analysis module for Pkynetics.

This module provides functionality for differential scanning calorimetry (DSC)
data analysis, including baseline correction, peak detection, thermal event analysis,
and heat capacity calculations.
"""

from .baseline import BaselineCorrector
from .core import DSCAnalyzer
from .heat_capacity import CpCalculator
from .peak_analysis import PeakAnalyzer
from .signal_stability import SignalStabilityDetector
from .thermal_events import ThermalEventDetector
from .types import (
    BaselineResult,
    CalibrationData,
    CpMethod,
    CpResult,
    CrystallizationEvent,
    DSCExperiment,
    DSCPeak,
    GlassTransition,
    MeltingEvent,
    OperationMode,
    PhaseTransition,
    StabilityMethod,
)
from .utilities import (
    DSCUnits,
    DataValidator,
    SignalProcessor,
    UnitConverter,
    find_intersection_point,
    safe_savgol_filter,
    validate_window_size,
)

__all__ = [
    # Core components
    "DSCAnalyzer",
    "DSCExperiment",
    "DSCPeak",
    "BaselineCorrector",
    "PeakAnalyzer",
    "ThermalEventDetector",
    "CpCalculator",
    "SignalStabilityDetector",
    # Utility components
    "DSCUnits",
    "SignalProcessor",
    "UnitConverter",
    "DataValidator",
    "validate_window_size",
    "safe_savgol_filter",
    "find_intersection_point",
    # Types and Enums
    "CpMethod",
    "OperationMode",
    "StabilityMethod",
    "DSCPeak",
    "DSCExperiment",
    "BaselineResult",
    "GlassTransition",
    "CrystallizationEvent",
    "MeltingEvent",
    "PhaseTransition",
    "CpResult",
    "CalibrationData",
]