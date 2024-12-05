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
    # Coats-Redfern methods
    "coats_redfern_equation",
    "coats_redfern_method",
    # Freeman-Carroll methods
    "freeman_carroll_equation",
    "freeman_carroll_method",
    "plot_diagnostic",
    # Horowitz-Metzger methods
    "horowitz_metzger_equation",
    "horowitz_metzger_method",
    # JMAK methods
    "fit_modified_jmak",
    "jmak_equation",
    "jmak_half_time",
    "jmak_method",
    "modified_jmak_equation",
    # Kissinger methods
    "kissinger_equation",
    "kissinger_method",
]
