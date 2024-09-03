"""Model-specific plotting functions for Pkynetics."""

import matplotlib.pyplot as plt
import numpy as np

def plot_coats_redfern(x: np.ndarray, y: np.ndarray, e_a: float, a: float, r_squared: float):
    """
    Create a Coats-Redfern plot.
    
    Args:
        x (np.array): 1000/T values
        y (np.array): ln(-ln(1-α)/T^2) values
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor
        r_squared (float): R-squared value of the fit
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data')
    fit = np.polyfit(x, y, 1)
    plt.plot(x, np.polyval(fit, x), 'r-', label='Fit')
    plt.xlabel('1000/T (K^-1)')
    plt.ylabel('ln(-ln(1-α)/T^2)')
    plt.title('Coats-Redfern Plot')
    plt.legend()
    plt.grid(True)
    
    # Add text box with results
    textstr = f'E_a = {e_a/1000:.2f} kJ/mol\nA = {a:.2e} min^-1\nR^2 = {r_squared:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.show()

def plot_freeman_carroll(x: np.ndarray, y: np.ndarray, e_a: float, n: float, r_squared: float):
    """
    Create a Freeman-Carroll plot.
    
    Args:
        x (np.array): Δ(1/T) / Δln(1-α) values
        y (np.array): Δln(dα/dt) / Δln(1-α) values
        e_a (float): Activation energy in J/mol
        n (float): Reaction order
        r_squared (float): R-squared value of the fit
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data')
    fit = np.polyfit(x, y, 1)
    plt.plot(x, np.polyval(fit, x), 'r-', label='Fit')
    plt.xlabel('Δ(1/T) / Δln(1-α)')
    plt.ylabel('Δln(dα/dt) / Δln(1-α)')
    plt.title('Freeman-Carroll Plot')
    plt.legend()
    plt.grid(True)
    
    # Add text box with results
    textstr = f'E_a = {e_a/1000:.2f} kJ/mol\nn = {n:.2f}\nR^2 = {r_squared:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.show()

def plot_kissinger(temperatures: np.ndarray, heating_rates: np.ndarray, e_a: float, a: float, r_squared: float):
    """
    Create a Kissinger plot.
    
    Args:
        temperatures (np.array): Peak temperatures in K
        heating_rates (np.array): Heating rates in K/min
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor
        r_squared (float): R-squared value of the fit
    """
    plt.figure(figsize=(10, 6))
    x = 1000 / temperatures
    y = np.log(heating_rates / temperatures**2)
    plt.scatter(x, y, label='Data')
    fit = np.polyfit(x, y, 1)
    plt.plot(x, np.polyval(fit, x), 'r-', label='Fit')
    plt.xlabel('1000/T_p (K^-1)')
    plt.ylabel('ln(β/T_p^2)')
    plt.title('Kissinger Plot')
    plt.legend()
    plt.grid(True)
    
    # Add text box with results
    textstr = f'E_a = {e_a/1000:.2f} kJ/mol\nA = {a:.2e} min^-1\nR^2 = {r_squared:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.show()

# Add more model-specific plotting functions as needed
