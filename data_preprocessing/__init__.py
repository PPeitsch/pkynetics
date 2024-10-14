"""
Data preprocessing module for Pkynetics.

This module provides functions for preprocessing thermal analysis data,
including calculation of transformed fraction from DSC data.
"""

from .preprocessing import calculate_transformed_fraction

__all__ = ['calculate_transformed_fraction']
