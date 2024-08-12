"""
Model-fitting methods for thermal analysis kinetics.

This module provides various model-fitting methods for analyzing thermal analysis data.
"""

from .avrami import avrami_method, avrami_equation
#from .kissinger import kissinger_method
#from .coats_redfern import coats_redfern_method
#from .freeman_carroll import freeman_carroll_method
#from .horowitz_metzger import horowitz_metzger_method

__all__ = [
    "avrami_method",
    "avrami_equation",
#    "kissinger_method",
#    "coats_redfern_method",
#    "freeman_carroll_method",
#    "horowitz_metzger_method",
]
