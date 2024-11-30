import numpy as np
from numpy.typing import NDArray

from .common_preprocessing import smooth_data


def calculate_tga_transformed_fraction(
    weight: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate the transformed fraction from TGA weight data.

    Args:
        weight (np.ndarray): Weight data from TGA.

    Returns:
        np.ndarray: Transformed fraction (normalized from 0 to 1).
    """
    smoothed_weight = smooth_data(weight)
    transformed_fraction = (smoothed_weight - smoothed_weight.min()) / (
        smoothed_weight.max() - smoothed_weight.min()
    )
    return np.array(
        1 - transformed_fraction, dtype=np.float64
    )  # Invert because weight typically decreases in TGA
