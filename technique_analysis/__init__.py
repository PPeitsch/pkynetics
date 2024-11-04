"""
Technique-specific analysis module.

This module provides analysis tools for specific thermal analysis techniques
including dilatometry, DSC, and TGA.
"""

from .dilatometry import (
    analyze_dilatometry_curve,
    find_inflection_points,
    extrapolate_linear_segments,
    calculate_transformed_fraction_lever,
    tangent_method
)

__all__ = [
    # Dilatometry analysis
    'analyze_dilatometry_curve',
    'find_inflection_points',
    'extrapolate_linear_segments',
    'calculate_transformed_fraction_lever',
    'tangent_method',
]
