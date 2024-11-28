import numpy as np

from .common_preprocessing import smooth_data


def calculate_dsc_transformed_fraction(heat_flow: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Calculate the transformed fraction from DSC heat flow data.

    Args:
        heat_flow (np.ndarray): Heat flow data from DSC.
        time (np.ndarray): Corresponding time data.

    Returns:
        np.ndarray: Transformed fraction (normalized from 0 to 1).
    """
    smoothed_heat_flow = smooth_data(heat_flow)
    cumulative_heat = np.cumsum(smoothed_heat_flow)
    transformed_fraction = (cumulative_heat - cumulative_heat.min()) / (cumulative_heat.max() - cumulative_heat.min())
    return transformed_fraction
