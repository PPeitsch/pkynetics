"""
Pkynetics: A comprehensive library for thermal analysis kinetic methods.

This library provides tools for data preprocessing, kinetic analysis using various
methods (model-fitting, model-free), technique-specific analysis, and result 
visualization for thermal analysis data.
"""

from . import (
    data_import,
    data_preprocessing,
    model_fitting_methods,
    model_free_methods,
    result_visualization,
    synthetic_data,
    technique_analysis,
)
from .__about__ import __version__

__all__ = [
    "data_import",
    "data_preprocessing",
    "model_fitting_methods",
    "model_free_methods",
    "result_visualization",
    "synthetic_data",
    "technique_analysis",
    "__version__",
]
