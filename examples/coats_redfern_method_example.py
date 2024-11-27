"""Example usage of the Coats-Redfern method for non-isothermal kinetics analysis."""

import sys
import os

import numpy as np
from src.pkynetics.synthetic_data import generate_coats_redfern_data
from src.pkynetics.model_fitting_methods import coats_redfern_method
from src.pkynetics.result_visualization import plot_conversion_vs_temperature
from src.pkynetics.result_visualization import plot_coats_redfern

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set true values for synthetic data generation
e_a_true = 150000  # J/mol
a_true = 1e15  # 1/s
heating_rate = 10  # K/min
t_range = (400, 800)  # K
n_true = 1  # Reaction order

# Generate synthetic data
temperature, alpha = generate_coats_redfern_data(e_a_true, a_true, heating_rate, t_range, n=n_true, noise_level=0.001)

# Ensure alpha is within [0, 1] and remove potential artifacts at the boundaries
alpha = np.clip(alpha, 0.01, 0.99)

# Plot conversion vs temperature
plot_conversion_vs_temperature([temperature], [alpha], [heating_rate])

# Perform Coats-Redfern analysis
e_a, a, r_squared, x, y, x_fit, y_fit = coats_redfern_method(temperature, alpha, heating_rate, n=n_true)

# Plot Coats-Redfern results
plot_coats_redfern(x, y, x_fit, y_fit, e_a, a, r_squared)

# Print results
print(f"True values: E_a = {e_a_true/1000:.2f} kJ/mol, A = {a_true:.2e} min^-1")
print(f"Fitted values: E_a = {e_a/1000:.2f} kJ/mol, A = {a:.2e} min^-1")
print(f"R^2 = {r_squared:.4f}")

# Calculate relative error
e_a_error = abs(e_a - e_a_true) / e_a_true * 100
a_error = abs(a - a_true) / a_true * 100

print(f"Relative error in E_a: {e_a_error:.2f}%")
print(f"Relative error in A: {a_error:.2f}%")
