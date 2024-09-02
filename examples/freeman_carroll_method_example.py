"""Example usage of the Freeman-Carroll method for non-isothermal kinetics analysis."""

import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods import freeman_carroll_method, freeman_carroll_plot

# Generate sample data
time = np.linspace(0, 100, 1000)  # Time in minutes
temperature = 300 + 5 * time  # Temperature in Kelvin, with a heating rate of 5 K/min
true_e_a = 100000  # J/mol
true_a = 1e10  # min^-1
true_n = 1.5  # Reaction order

# Calculate conversion
r = 8.314  # Gas constant in J/(mol·K)
k = true_a * np.exp(-true_e_a / (r * temperature))
alpha = 1 - np.exp(-(k * time) ** true_n)

# Add some noise to make it more realistic
noise_level = 0.005
alpha_noisy = np.clip(alpha + np.random.normal(0, noise_level, alpha.shape), 0.001, 0.999)

# Perform Freeman-Carroll analysis
e_a, n, r_squared = freeman_carroll_method(temperature, alpha_noisy, time)

print(f"True values: E_a = {true_e_a/1000:.2f} kJ/mol, n = {true_n:.2f}")
print(f"Fitted values: E_a = {e_a/1000:.2f} kJ/mol, n = {n:.2f}")
print(f"R^2 = {r_squared:.4f}")

# Generate plot data
x, y, _, _, _ = freeman_carroll_plot(temperature, alpha_noisy, time)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Experimental data', alpha=0.5, s=10)
plt.plot(x, np.polyval(np.polyfit(x, y, 1), x), 'r-', label='Fitted line')
plt.xlabel('Δ(1/T) / Δln(1-α)')
plt.ylabel('Δln(dα/dt) / Δln(1-α)')
plt.title('Freeman-Carroll Plot')
plt.legend()
plt.grid(True)

# Add text box with results
textstr = f'E_a = {e_a/1000:.2f} kJ/mol\nn = {n:.2f}\nR$^2$ = {r_squared:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Calculate relative error
e_a_error = abs(e_a - true_e_a) / true_e_a * 100
n_error = abs(n - true_n) / true_n * 100

print(f"Relative error in E_a: {e_a_error:.2f}%")
print(f"Relative error in n: {n_error:.2f}%")
