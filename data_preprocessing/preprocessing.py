import numpy as np


def calculate_transformed_fraction(heat_flow: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Calculate the transformed fraction from DSC heat flow data.

    Args:
        heat_flow (np.ndarray): Heat flow data from DSC.
        time (np.ndarray): Corresponding time data.

    Returns:
        np.ndarray: Transformed fraction (normalized from 0 to 1).
    """
    cumulative_heat = np.cumsum(heat_flow)
    transformed_fraction = (cumulative_heat - cumulative_heat.min()) / (cumulative_heat.max() - cumulative_heat.min())
    return transformed_fraction
