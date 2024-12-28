from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from pkynetics.data_preprocessing.common_preprocessing import smooth_data


@dataclass
class DSCExperiment:
    """Class to hold DSC experiment data and parameters."""

    temperature: np.ndarray
    heat_flow: np.ndarray
    time: np.ndarray
    mass: float
    heating_rate: Optional[float] = None
    name: str = "sample"

    def __post_init__(self):
        """Calculate heating rate if not provided."""
        if self.heating_rate is None:
            self.heating_rate = np.mean(np.gradient(self.temperature, self.time)) * 60


class SpecificHeatCalculator:
    """Class for specific heat calculations using DSC data."""

    def __init__(self, temperature_range: Optional[Tuple[float, float]] = None):
        """
        Initialize calculator with optional temperature range.

        Args:
            temperature_range: Optional tuple of (min_temp, max_temp) in Kelvin
        """
        self.temperature_range = temperature_range

    def _validate_data(self, *experiments: DSCExperiment) -> bool:
        """Validate experiment data."""
        lengths = [len(exp.temperature) for exp in experiments]
        if len(set(lengths)) != 1:
            raise ValueError("All experiments must have the same number of points")

        if self.temperature_range:
            min_temp, max_temp = self.temperature_range
            for exp in experiments:
                if exp.temperature.min() > min_temp or exp.temperature.max() < max_temp:
                    raise ValueError(
                        f"Temperature range {self.temperature_range} outside data range"
                    )
        return True

    def calculate_two_step(
        self, sample: DSCExperiment, baseline: DSCExperiment
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculate specific heat using two-step method.

        Args:
            sample: Sample experiment data
            baseline: Baseline experiment data

        Returns:
            Tuple of (specific heat array, metadata dictionary)
        """
        self._validate_data(sample, baseline)

        # Smooth heat flow signals
        sample_hf_smooth = smooth_data(sample.heat_flow)
        baseline_hf_smooth = smooth_data(baseline.heat_flow)

        # Calculate net heat flow
        net_heat_flow = sample_hf_smooth - baseline_hf_smooth

        # Convert units and calculate Cp (J/(g·K))
        specific_heat = (net_heat_flow * 60) / (sample.mass * sample.heating_rate)

        metadata = {
            "heating_rate": sample.heating_rate,
            "mean_cp": np.mean(specific_heat),
            "std_cp": np.std(specific_heat),
            "temperature_range": self.temperature_range,
        }

        return specific_heat, metadata

    def calculate_three_step(
        self,
        sample: DSCExperiment,
        reference: DSCExperiment,
        baseline: DSCExperiment,
        reference_cp: Union[np.ndarray, float],
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculate specific heat using three-step method.

        Args:
            sample: Sample experiment data
            reference: Reference material experiment data
            baseline: Baseline experiment data
            reference_cp: Known Cp values of reference material

        Returns:
            Tuple of (specific heat array, metadata dictionary)
        """
        self._validate_data(sample, reference, baseline)

        # Smooth heat flow signals
        sample_hf_smooth = smooth_data(sample.heat_flow)
        reference_hf_smooth = smooth_data(reference.heat_flow)
        baseline_hf_smooth = smooth_data(baseline.heat_flow)

        # Calculate net heat flows
        net_sample = sample_hf_smooth - baseline_hf_smooth
        net_reference = reference_hf_smooth - baseline_hf_smooth

        # Calculate specific heat
        specific_heat = (net_sample * reference.mass * reference_cp) / (
            net_reference * sample.mass
        )

        metadata = {
            "heating_rate": sample.heating_rate,
            "mean_cp": np.mean(specific_heat),
            "std_cp": np.std(specific_heat),
            "reference_ratio": np.mean(net_sample / net_reference),
            "temperature_range": self.temperature_range,
        }

        return specific_heat, metadata


def get_sapphire_cp(temperature: np.ndarray) -> np.ndarray:
    """Calculate sapphire specific heat capacity (273.15 K to 1000 K)."""
    if np.any(temperature < 273.15) or np.any(temperature > 1000):
        raise ValueError("Temperature must be between 273.15 K and 1000 K")

    # NIST SRM 720 coefficients
    a, b, c = 1.0289, 2.3506e-4, -1.6818e-7
    return a + b * temperature + c * temperature**2
