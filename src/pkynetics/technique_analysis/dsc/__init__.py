"""DSC analysis module for Pkynetics."""

from .baseline import BaselineCorrector
from .core import DSCAnalyzer
from .heat_capacity import CpCalculator
from .peak_analysis import PeakAnalyzer
from .thermal_events import ThermalEventDetector
from .types import DSCExperiment, DSCPeak
from .utilities import (
    DataValidator,
    DSCUnits,
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
    # Utility components
    "DSCUnits",
    "SignalProcessor",
    "UnitConverter",
    "DataValidator",
    "validate_window_size",
    "safe_savgol_filter",
    "find_intersection_point",
]
