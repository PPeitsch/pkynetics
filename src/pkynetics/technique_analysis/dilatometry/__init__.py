"""Dilatometry analysis.

The module is a package of focused pieces, but its public surface is
unchanged: every function below is importable from here and from
``pkynetics.technique_analysis`` exactly as before.

- :mod:`~pkynetics.technique_analysis.dilatometry.core` -- :func:`analyze_dilatometry_curve` and
  :class:`DilatometryAnalyzer`, the two ways in
- :mod:`~pkynetics.technique_analysis.dilatometry.transformation_points` -- where a transformation starts and ends
- :mod:`~pkynetics.technique_analysis.dilatometry.linear_segments` -- the baselines on either side of it
- :mod:`~pkynetics.technique_analysis.dilatometry.transformed_fraction` -- how far it has gone
- :mod:`~pkynetics.technique_analysis.dilatometry.methods` -- the lever and tangent methods
- :mod:`~pkynetics.technique_analysis.dilatometry.curve_features` -- the derivative and the noise of the curve
- :mod:`~pkynetics.technique_analysis.dilatometry.utilities` -- small numerical helpers
- :mod:`~pkynetics.technique_analysis.dilatometry.types` -- type definitions
"""

from .core import DilatometryAnalyzer, analyze_dilatometry_curve
from .curve_features import _strain_derivative, detect_noise_level
from .linear_segments import (
    calculate_fit_quality,
    extrapolate_linear_segments,
    find_optimal_margin,
    fit_linear_segments,
    get_extrapolated_values,
    get_linear_segment_masks,
)
from .methods import lever_method, tangent_method
from .transformation_points import (
    find_inflection_points,
    find_midpoint_temperature,
    find_transformation_limits,
)
from .transformed_fraction import (
    calculate_transformed_fraction,
    calculate_transformed_fraction_lever,
)
from .types import ReturnDict
from .utilities import calculate_r2

__all__ = [
    # Entry points
    "analyze_dilatometry_curve",
    "DilatometryAnalyzer",
    # Transformation points
    "find_transformation_limits",
    "find_inflection_points",
    "find_midpoint_temperature",
    # Linear segments
    "extrapolate_linear_segments",
    "fit_linear_segments",
    "get_linear_segment_masks",
    "get_extrapolated_values",
    "find_optimal_margin",
    "calculate_fit_quality",
    # Transformed fraction
    "calculate_transformed_fraction",
    "calculate_transformed_fraction_lever",
    # Methods
    "lever_method",
    "tangent_method",
    # Curve features and helpers
    "detect_noise_level",
    "calculate_r2",
    # Types
    "ReturnDict",
]
