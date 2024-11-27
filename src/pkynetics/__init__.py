"""
Pkynetics: A comprehensive library for thermal analysis kinetic methods.

This library provides tools for data preprocessing, kinetic analysis using various
methods (model-fitting, model-free, and advanced techniques), machine learning
approaches, and result visualization for thermal analysis data.
"""

from src.pkynetics import data_import, model_fitting_methods, model_free_methods, result_visualization

from src.pkynetics.data_import import tga_importer, dsc_importer
from src.pkynetics.model_fitting_methods import kissinger_method

__all__ = [
    "data_preprocessing",
    "data_import",
    "model_fitting_methods",
    "model_free_methods",
    "advanced_methods",
    "machine_learning_methods",
    "result_visualization",
    "statistical_analysis",
    "parallel_processing",
    "utility_functions",
    "smooth_data",
    "tga_importer",
    "dsc_importer",
    "kissinger_method",
]
