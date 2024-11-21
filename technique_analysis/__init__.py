"""
Technique-specific analysis module.

This module provides comprehensive analysis tools for various thermal analysis techniques:

1. Dilatometry Analysis:
   - Transformation point detection
   - Lever rule and tangent method implementations
   - Transformed fraction calculation

2. DSC Analysis:
   - Specific heat calculation (two-step and three-step methods)
   - More features planned

3. TGA Analysis (Planned)
"""

from technique_analysis.dilatometry import (
    analyze_dilatometry_curve,
    find_inflection_points,
    extrapolate_linear_segments,
    calculate_transformed_fraction_lever,
    find_optimal_margin,
    lever_method,
    tangent_method,
    calculate_fit_quality,
    calculate_r2
)

from technique_analysis.dsc import (
    DSCExperiment,
    SpecificHeatCalculator,
    get_sapphire_cp
)

__all__ = [
    # Dilatometry analysis
    'analyze_dilatometry_curve',
    'find_inflection_points',
    'extrapolate_linear_segments',
    'calculate_transformed_fraction_lever',
    'find_optimal_margin',
    'lever_method',
    'tangent_method',
    'calculate_fit_quality',
    'calculate_r2',

    # DSC analysis
    'DSCExperiment',
    'SpecificHeatCalculator',
    'get_sapphire_cp'
]