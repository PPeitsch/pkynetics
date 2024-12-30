"""Type definitions for DSC analysis."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

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
