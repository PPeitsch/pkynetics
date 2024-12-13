"""Example usage of the Freeman-Carroll method for non-isothermal kinetics analysis."""

import sys
import os

import numpy as np
from pkynetics.synthetic_data import generate_freeman_carroll_data
from pkynetics.model_fitting_methods import freeman_carroll_method, plot_diagnostic
from pkynetics.result_visualization import plot_conversion_vs_temperature
from pkynetics.result_visualization.model_specific_plots import plot_freeman_carroll

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set true values for synthetic data generation
e_a_true = 150000  # J/mol
a_true = 1e15  # 1/s
heating_rate = 10  # K/min
t_range = (400, 800)  # K
n_true = 1.5  # Reaction order

# Generate synthetic data
temperature, alpha, time = generate_freeman_carroll_data(e_a_true, a_true, heating_rate,
                                                         t_range, n=n_true, noise_level=0.001)

# Ensure alpha is within [0, 1] and remove potential artifacts at the boundaries
alpha = np.clip(alpha, 0.01, 0.99)

# Plot conversion vs temperature
plot_conversion_vs_temperature([temperature], [alpha], [heating_rate])

# Perform Freeman-Carroll analysis
e_a, n, r_squared, x, y, x_fit, y_fit = freeman_carroll_method(temperature, alpha, time)

# Plot Freeman-Carroll results
plot_freeman_carroll(x, y, x_fit, y_fit, e_a, n, r_squared)

# Print results
print(f"True values: E_a = {e_a_true/1000:.2f} kJ/mol, n = {n_true:.2f}")
print(f"Fitted values: E_a = {e_a/1000:.2f} kJ/mol, n = {n:.2f}")
print(f"R^2 = {r_squared:.4f}")

# Calculate relative error
e_a_error = abs(e_a - e_a_true) / e_a_true * 100
n_error = abs(n - n_true) / n_true * 100

print(f"Relative error in E_a: {e_a_error:.2f}%")
print(f"Relative error in n: {n_error:.2f}%")

plot_diagnostic(time, alpha, temperature)