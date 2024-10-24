import numpy as np
from scipy.signal import savgol_filter


def smooth_data(data: np.ndarray, window_length: int = 51, polyorder: int = 3) -> np.ndarray:
    """
    Smooth the data using Savitzky-Golay filter.

    Args:
        data (np.ndarray): Input data to be smoothed.
        window_length (int): Length of the filter window. Must be odd and greater than polyorder.
        polyorder (int): Order of the polynomial used to fit the samples.

    Returns:
        np.ndarray: Smoothed data.
    """
    return savgol_filter(data, window_length, polyorder)
