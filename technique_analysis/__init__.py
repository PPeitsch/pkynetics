"""
Technique-specific analysis module for Pkynetics.

This module provides comprehensive analysis tools for various thermal analysis techniques:

1. Dilatometry Analysis:
   - Transformation point detection
   - Lever rule and tangent method implementations
   - Transformed fraction calculation

2. DSC Analysis (Planned)

3. TGA Analysis (Planned)
"""

from .dilatometry import (
    # Main analysis functions
    analyze_dilatometry_curve,

    # Core methods
    find_inflection_points,
    extrapolate_linear_segments,
    calculate_transformed_fraction_lever,
    find_optimal_margin,

    # Analysis methods
    lever_method,
    tangent_method,

    # Quality assessment
    calculate_fit_quality,
    calculate_r2
)

__all__ = [
    # Main analysis function
    'analyze_dilatometry_curve',

    # Core analysis functions
    'find_inflection_points',
    'extrapolate_linear_segments',
    'calculate_transformed_fraction_lever',
    'find_optimal_margin',

    # Analysis methods
    'lever_method',
    'tangent_method',

    # Quality assessment functions
    'calculate_fit_quality',
    'calculate_r2'
]
