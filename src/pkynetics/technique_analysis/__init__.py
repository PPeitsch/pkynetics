"""
Technique-specific analysis module.

This module provides comprehensive analysis tools for various thermal analysis techniques:

1. Dilatometry Analysis:
   - Transformation point detection
   - Lever rule and tangent method implementations
   - Complete workflow with DilatometryAnalyzer
   - Transformed fraction calculation

2. DSC Analysis (``technique_analysis.dsc``):
   - Baseline correction, peak analysis (ISO 11357 onset/endset, enthalpy)
   - Thermal events: glass transition, crystallization, melting
   - Specific heat capacity (single-step, three-step, stepped, modulated)
   - Complete workflow with DSCAnalyzer

3. TGA Analysis (Planned)
"""

from . import dsc
from .dilatometry import (
    DilatometryAnalyzer,
    analyze_dilatometry_curve,
    calculate_fit_quality,
    calculate_r2,
    calculate_transformed_fraction_lever,
    extrapolate_linear_segments,
    find_inflection_points,
    find_optimal_margin,
    find_transformation_limits,
    lever_method,
    tangent_method,
)

__all__ = [
    # Technique subpackages
    "dsc",
    # Main analysis entry points
    "analyze_dilatometry_curve",
    "DilatometryAnalyzer",
    # Core analysis functions
    "find_inflection_points",
    "find_transformation_limits",
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
