"""
Pkynetics: A comprehensive library for thermal analysis kinetic methods.

This library provides tools for data preprocessing, kinetic analysis using various
methods (model-fitting, model-free), technique-specific analysis, and result 
visualization for thermal analysis data.
"""

__version__ = "0.2.3"

from src.pkynetics import technique_analysis, model_free_methods

__all__ = [
    "data_import",
    "data_preprocessing",
    "model_fitting_methods",
    "model_free_methods",
    "result_visualization",
    "synthetic_data",
    "technique_analysis"
]
