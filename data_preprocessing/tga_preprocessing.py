import numpy as np
from .common_preprocessing import smooth_data


def calculate_tga_transformed_fraction(weight: np.ndarray) -> np.ndarray:
    """
    Calculate the transformed fraction from TGA weight data.

    Args:
        weight (np.ndarray): Weight data from TGA.

    Returns:
        np.ndarray: Transformed fraction (normalized from 0 to 1).
    """
    smoothed_weight = smooth_data(weight)
    transformed_fraction = (smoothed_weight - smoothed_weight.min()) / (smoothed_weight.max() - smoothed_weight.min())
    return 1 - transformed_fraction  # Invert because weight typically decreases in TGA
