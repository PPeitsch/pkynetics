"""
Pkynetics: A comprehensive library for thermal analysis kinetic methods.

This library provides tools for data preprocessing, kinetic analysis using various
methods (model-fitting, model-free), technique-specific analysis, and result 
visualization for thermal analysis data.
"""

__version__ = "0.2.3"

from . import data_import
from . import data_preprocessing
from . import model_fitting_methods
from . import model_free_methods
from . import result_visualization
from . import synthetic_data
from . import technique_analysis

__all__ = [
    "data_import",
    "data_preprocessing",
    "model_fitting_methods",
    "model_free_methods",
    "result_visualization",
    "synthetic_data",
    "technique_analysis"
]
