"""
Pkynetics: A comprehensive library for thermal analysis kinetic methods.

This library provides tools for data preprocessing, kinetic analysis using various
methods (model-fitting, model-free, and advanced techniques), machine learning
approaches, and result visualization for thermal analysis data.
"""

__version__ = "0.1.0"

from . import data_preprocessing
from . import data_import
from . import model_fitting_methods
from . import model_free_methods
from . import advanced_methods
from . import machine_learning_methods
from . import result_visualization
from . import statistical_analysis
from . import parallel_processing
from . import utility_functions

# Convenience imports for commonly used functions
from .data_preprocessing import smooth_data
from .data_import import tga_importer, dsc_importer
from .result_visualization import plot_activation_energy

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
    "synthetic_data",
    "plot_activation_energy",
]
