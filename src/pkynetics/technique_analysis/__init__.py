"""
Technique-specific analysis module.

This module provides comprehensive analysis tools for various thermal analysis techniques:

1. Dilatometry Analysis:
   - Transformation point detection
   - Lever rule and tangent method implementations
   - Transformed fraction calculation

2. DSC Analysis (Planned)

3. TGA Analysis (Planned)
"""

from .dilatometry import (
    analyze_dilatometry_curve,
    calculate_fit_quality,
    calculate_r2,
    calculate_transformed_fraction_lever,
    extrapolate_linear_segments,
    find_inflection_points,
    find_optimal_margin,
    lever_method,
    tangent_method,
)

__all__ = [
    # Main analysis function
    "analyze_dilatometry_curve",
    # Core analysis functions
    "find_inflection_points",
    "extrapolate_linear_segments",
    "calculate_transformed_fraction_lever",
    "find_optimal_margin",
    # Analysis methods
    "lever_method",
    "tangent_method",
    # Quality assessment functions
    "calculate_fit_quality",
    "calculate_r2",
]
