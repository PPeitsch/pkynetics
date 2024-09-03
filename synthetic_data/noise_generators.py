"""Module for generating various types of noise for kinetic data."""

import numpy as np


def add_gaussian_noise(data: np.ndarray, std_dev: float) -> np.ndarray:
    """
    Add Gaussian noise to the data.
    
    Args:
        data (np.ndarray): Input data
        std_dev (float): Standard deviation of the Gaussian noise
    
    Returns:
        np.ndarray: Data with added Gaussian noise
    """
    noise = np.random.normal(0, std_dev, data.shape)
    return np.clip(data + noise, 0, 1)


def add_outliers(data: np.ndarray, outlier_fraction: float, outlier_std_dev: float) -> np.ndarray:
    """
    Add outliers to the data.
    
    Args:
        data (np.ndarray): Input data
        outlier_fraction (float): Fraction of data points to be outliers
        outlier_std_dev (float): Standard deviation for generating outliers
    
    Returns:
        np.ndarray: Data with added outliers
    """
    num_outliers = int(len(data) * outlier_fraction)
    outlier_indices = np.random.choice(len(data), num_outliers, replace=False)
    outliers = np.random.normal(0, outlier_std_dev, num_outliers)
    data[outlier_indices] += outliers
    return np.clip(data, 0, 1)
