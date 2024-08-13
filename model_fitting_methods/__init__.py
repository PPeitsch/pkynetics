"""Model fitting methods for kinetic analysis in Pkynetics."""

from .avrami import avrami_method, avrami_equation
from .kissinger import kissinger_method, kissinger_equation, calculate_t_p
from .coats_redfern import coats_redfern_method, coats_redfern_equation, coats_redfern_plot
#from .freeman_carroll import freeman_carroll_method, freeman_carroll_equation, freeman_carroll_plot
#from .horowitz_metzger import horowitz_metzger_method, horowitz_metzger_equation, horowitz_metzger_plot

__all__ = [
    "avrami_method",
    "avrami_equation",
    "kissinger_method",
    "kissinger_equation",
    "calculate_t_p",
    "coats_redfern_method",
    "coats_redfern_equation",
    "coats_redfern_plot",
#    "freeman_carroll_method",
#    "freeman_carroll_equation",
#    "freeman_carroll_plot",
#    "horowitz_metzger_method",
#    "horowitz_metzger_equation",
#    "horowitz_metzger_plot"
]
