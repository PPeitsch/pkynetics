from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .common_preprocessing import calculate_derivatives, smooth_data


def preprocess_dilatometry_data(
    temperature: NDArray[np.float64], strain: NDArray[np.float64], smooth: bool = True
) -> Dict[str, Union[NDArray[np.float64], Dict[str, NDArray[np.float64]]]]:
    """
    Preprocess dilatometry data for analysis.

    Args:
        temperature: Temperature data in °C
        strain: Relative length change data
        smooth: Whether to smooth the strain data

    Returns:
        Dict containing processed data including:
        - temperature: Original or processed temperature data
        - strain: Original or smoothed strain data
        - derivatives: Dict containing first and second derivatives
    """
    processed_strain = smooth_data(strain) if smooth else strain
    derivatives = calculate_derivatives(temperature, processed_strain)

    return {
        "temperature": np.array(temperature, dtype=np.float64),
        "strain": np.array(processed_strain, dtype=np.float64),
        "derivatives": derivatives,
    }


def normalize_strain(
    strain: NDArray[np.float64], reference_temp_idx: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Normalize strain data relative to a reference point.

    Args:
        strain: Strain/relative length change data
        reference_temp_idx: Index of reference temperature point. If None, uses first point

    Returns:
        Normalized strain data
    """
    if reference_temp_idx is None:
        reference_temp_idx = 0

    reference_value = strain[reference_temp_idx]
    return np.array((strain - reference_value) / reference_value, dtype=np.float64)


def detect_noise_level(strain: NDArray[np.float64], window_size: int = 20) -> float:
    """
    Estimate noise level in strain data.

    Args:
        strain: Strain data
        window_size: Window size for local standard deviation calculation

    Returns:
        Estimated noise level (standard deviation)
    """
    local_std = np.array(
        [
            np.std(strain[i : i + window_size])
            for i in range(0, len(strain) - window_size, window_size)
        ]
    )

    return float(np.median(local_std))


def remove_outliers(
    temperature: np.ndarray, strain: np.ndarray, threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from strain data using z-score method.

    Args:
        temperature: Temperature data
        strain: Strain data
        threshold: Z-score threshold for outlier detection

    Returns:
        Tuple of cleaned (temperature, strain) arrays
    """
    z_scores = np.abs((strain - np.mean(strain)) / np.std(strain))
    mask = z_scores < threshold

    return temperature[mask], strain[mask]
