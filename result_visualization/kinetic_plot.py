"""General kinetic plotting functions for Pkynetics."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from model_fitting_methods import kissinger_equation

# Constants
R = 8.314  # Gas constant in J/(mol·K)


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
    y = np.log(a) - e_a / (R * 1 / x)
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


def plot_kissinger(t_p: np.ndarray, beta: np.ndarray, e_a: float, a: float, r_squared: float):
    """
    Create a Kissinger plot.

    Args:
        t_p (np.ndarray): Peak temperatures for different heating rates in K
        beta (np.ndarray): Heating rates in K/min
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in min^-1
        r_squared (float): R-squared value of the fit
    """
    plt.figure(figsize=(10, 6))

    x_exp = 1000 / t_p
    y_exp = np.log(beta / t_p ** 2)

    plt.scatter(x_exp, y_exp, label='Experimental data')

    if not np.isnan(e_a) and not np.isnan(a):
        # Generate theoretical curve
        x_theory = np.linspace(min(x_exp), max(x_exp), 100)
        t_theory = 1000 / x_theory
        beta_theory = np.mean(beta)  # Use average heating rate for theoretical curve
        y_theory = kissinger_equation(t_theory, e_a, a, np.full_like(t_theory, beta_theory))

        plt.plot(x_theory, y_theory, 'r-', label='Theoretical curve')

        # Add text box with results
        textstr = f'E_a = {e_a / 1000:.2f} kJ/mol\nA = {a:.2e} min$^{{-1}}$\nR$^2$ = {r_squared:.4f}'
    else:
        textstr = "Insufficient data for Kissinger analysis"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    plt.xlabel('1000/T (K$^{-1}$)')
    plt.ylabel('ln(β/T$_p^2$) (K$^{-1}$·min$^{-1}$)')
    plt.title('Kissinger Plot')
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_jmak_results(time: np.ndarray, transformed_fraction: np.ndarray,
                      fitted_curve: np.ndarray, n: float, k: float,
                      r_squared: float):
    """
    Plot the results of JMAK (Johnson-Mehl-Avrami-Kolmogorov) analysis.

    Args:
        time (np.ndarray): Time data
        transformed_fraction (np.ndarray): Experimental transformed fraction data
        fitted_curve (np.ndarray): Fitted JMAK curve
        n (float): Fitted JMAK exponent
        k (float): Fitted rate constant
        r_squared (float): R-squared value of the fit
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Data and fitted curve
    ax1.scatter(time, transformed_fraction, label='Experimental data', alpha=0.5)
    ax1.plot(time, fitted_curve, 'r-', label='Fitted curve')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Transformed Fraction')
    ax1.set_title('JMAK Analysis of Phase Transformation')
    ax1.legend()
    ax1.grid(True)

    # Add text box with results
    textstr = f'n = {n:.3f}\nk = {k:.3e}\nR² = {r_squared:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # JMAK plot
    mask = (transformed_fraction > 0.01) & (transformed_fraction < 0.99)
    y = np.log(-np.log(1 - transformed_fraction[mask]))
    x = np.log(time[mask])
    ax2.scatter(x, y, label='JMAK plot', alpha=0.5)
    ax2.plot(x, n * x + np.log(k) * n, 'r-', label='Linear fit')
    ax2.set_xlabel('log(Time)')
    ax2.set_ylabel('log(-log(1-X))')
    ax2.set_title('JMAK Plot')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_modified_jmak_results(temperature: np.ndarray, transformed_fraction: np.ndarray,
                               fitted_curve: np.ndarray, k0: float, n: float, E: float,
                               r_squared: float):
    """
    Plot the results of modified JMAK analysis.

    Args:
        temperature (np.ndarray): Temperature data
        transformed_fraction (np.ndarray): Experimental transformed fraction data
        fitted_curve (np.ndarray): Fitted modified JMAK curve
        k0 (float): Pre-exponential factor
        n (float): Modified JMAK exponent
        E (float): Activation energy in J/mol
        r_squared (float): R-squared value of the fit
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot experimental data
    ax.scatter(temperature, transformed_fraction, label='Experimental data', alpha=0.5)

    # Plot fitted curve
    ax.plot(temperature, fitted_curve, 'r-', label='Modified JMAK fit')

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Transformed Fraction')
    ax.set_title('Modified JMAK Analysis of Non-Isothermal Transformation')
    ax.legend()
    ax.grid(True)

    # Add text box with results
    textstr = f'k0 = {k0:.2e}\nn = {n:.3f}\nE = {E / 1000:.2f} kJ/mol\nR² = {r_squared:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()
