"""
Data preprocessing module for Pkynetics.

This module provides functions for preprocessing thermal analysis data,
including DSC, TGA, and dilatometry.
"""

from .common_preprocessing import smooth_data
from .dilatometry_preprocessing import (
    detect_noise_level,
    normalize_strain,
    preprocess_dilatometry_data,
    remove_outliers,
)
from .dsc_preprocessing import calculate_dsc_transformed_fraction
from .tga_preprocessing import calculate_tga_transformed_fraction

__all__ = [
    "smooth_data",
    "calculate_dsc_transformed_fraction",  # move
    "calculate_tga_transformed_fraction",  # move
    "preprocess_dilatometry_data",
    "normalize_strain",
    "detect_noise_level",
    "remove_outliers",
]
