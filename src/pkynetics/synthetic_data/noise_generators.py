import numpy as np
from numpy.typing import NDArray


def add_gaussian_noise(
    data: NDArray[np.float64], std_dev: float
) -> NDArray[np.float64]:
    """
    Add Gaussian noise to the data.

    Args:
        data (np.ndarray): Input data
        std_dev (float): Standard deviation of the Gaussian noise

    Returns:
        np.ndarray: Data with added Gaussian noise

    Raises:
        ValueError: If std_dev is negative
    """
    if std_dev < 0:
        raise ValueError("Standard deviation must be non-negative")

    noise = np.random.normal(0, std_dev, data.shape)
    return np.clip(data + noise, 0, 1)


def add_outliers(
    data: np.ndarray, outlier_fraction: float, outlier_std_dev: float
) -> np.ndarray:
    """
    Add outliers to the data.

    Args:
        data (np.ndarray): Input data
        outlier_fraction (float): Fraction of data points to be outliers (0 to 1)
        outlier_std_dev (float): Standard deviation for generating outliers

    Returns:
        np.ndarray: Data with added outliers

    Raises:
        ValueError: If outlier_fraction is not between 0 and 1, or if outlier_std_dev is negative
    """
    if not 0 <= outlier_fraction <= 1:
        raise ValueError("Outlier fraction must be between 0 and 1")
    if outlier_std_dev < 0:
        raise ValueError("Outlier standard deviation must be non-negative")

    num_outliers = int(len(data) * outlier_fraction)
    if num_outliers > 0:
        outlier_indices = np.random.choice(len(data), num_outliers, replace=False)
        outliers = np.random.normal(0, outlier_std_dev, num_outliers)
        data[outlier_indices] += outliers
    return np.clip(data, 0, 1)
