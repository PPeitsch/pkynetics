"""Synthetic data generation module for Pkynetics."""

from .basic_kinetic_data import generate_basic_kinetic_data
from .model_specific_data import (
    generate_coats_redfern_data,
    generate_freeman_carroll_data,
    generate_kissinger_data,
    generate_ofw_data,
    generate_friedman_data
)
from .noise_generators import add_gaussian_noise, add_outliers

__all__ = [
    'generate_basic_kinetic_data',
    'generate_coats_redfern_data',
    'generate_freeman_carroll_data',
    'generate_kissinger_data',
    'generate_ofw_data',
    'generate_friedman_data',
    'add_gaussian_noise',
    'add_outliers'
]
