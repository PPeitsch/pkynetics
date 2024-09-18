"""General kinetic plotting functions for Pkynetics."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_arrhenius(temperatures: np.ndarray, rate_constants: np.ndarray, e_a: float, a: float):
    """
    Create an Arrhenius plot.
    
    Args:
        temperatures (np.array): Array of temperatures in K
        rate_constants (np.array): Array of rate constants
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor
    """
    plt.figure(figsize=(10, 6))
    plt.plot(1 / temperatures, np.log(rate_constants), 'bo', label='Data')
    x = np.linspace(min(1 / temperatures), max(1 / temperatures), 100)
    y = np.log(a) - e_a / (8.314 * 1 / x)
    plt.plot(x, y, 'r-', label='Fit')
    plt.xlabel('1/T (K^-1)')
    plt.ylabel('ln(k)')
    plt.title('Arrhenius Plot')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_conversion_vs_temperature(temperatures: List[np.ndarray], conversions: List[np.ndarray],
                                   heating_rates: List[float]):
    """
    Plot conversion vs temperature for multiple heating rates.
    
    Args:
        temperatures (list): List of temperature arrays
        conversions (list): List of conversion arrays
        heating_rates (list): List of heating rates
    """
    plt.figure(figsize=(10, 6))
    for T, alpha, beta in zip(temperatures, conversions, heating_rates):
        plt.plot(T, alpha, label=f'{beta} K/min')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Conversion (α)')
    plt.title('Conversion vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_derivative_thermogravimetry(temperatures: List[np.ndarray], conversions: List[np.ndarray],
                                     heating_rates: List[float]):
    """
    Plot derivative thermogravimetry (DTG) curves for multiple heating rates.
    
    Args:
        temperatures (list): List of temperature arrays
        conversions (list): List of conversion arrays
        heating_rates (list): List of heating rates
    """
    plt.figure(figsize=(10, 6))
    for T, alpha, beta in zip(temperatures, conversions, heating_rates):
        dtg = np.gradient(alpha, T)
        plt.plot(T, dtg, label=f'{beta} K/min')
    plt.xlabel('Temperature (K)')
    plt.ylabel('dα/dT (K^-1)')
    plt.title('Derivative Thermogravimetry (DTG) Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_activation_energy_vs_conversion(conversions: np.ndarray, activation_energies: np.ndarray, method: str):
    """
    Plot activation energy as a function of conversion.
    
    Args:
        conversions (np.array): Array of conversion values
        activation_energies (np.array): Array of activation energies in kJ/mol
        method (str): Name of the method used for analysis
    """
    plt.figure(figsize=(10, 6))
    plt.plot(conversions, activation_energies, 'bo-')
    plt.xlabel('Conversion (α)')
    plt.ylabel('Activation Energy (kJ/mol)')
    plt.title(f'Activation Energy vs Conversion ({method} method)')
    plt.grid(True)
    plt.show()
