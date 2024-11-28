"""Model fitting methods for kinetic analysis in Pkynetics."""

from .coats_redfern import coats_redfern_equation, coats_redfern_method
from .freeman_carroll import (
    freeman_carroll_equation,
    freeman_carroll_method,
    plot_diagnostic,
)
from .horowitz_metzger import horowitz_metzger_equation, horowitz_metzger_method
from .jmak import (
    fit_modified_jmak,
    jmak_equation,
    jmak_half_time,
    jmak_method,
    modified_jmak_equation,
)
from .kissinger import kissinger_equation, kissinger_method

__all__ = [
    "jmak_method",
    "jmak_equation",
    "jmak_half_time",
    "modified_jmak_equation",
    "fit_modified_jmak",
    "kissinger_method",
    "coats_redfern_method",
    "coats_redfern_equation",
    "freeman_carroll_method",
    "freeman_carroll_equation",
    "plot_diagnostic",
    "horowitz_metzger_method",
    "horowitz_metzger_equation",
    "kissinger_equation"
]
