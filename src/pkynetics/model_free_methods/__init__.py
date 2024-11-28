"""Model-free methods for kinetic analysis in Pkynetics."""

from .friedman_method import friedman_method
from .kas_method import kas_method

__all__ = ["friedman_method", "ofw_method", "kas_method"]
