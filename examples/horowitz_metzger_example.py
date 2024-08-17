"""Example usage of the Horowitz-Metzger method for non-isothermal kinetics analysis."""

import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods import horowitz_metzger_method, horowitz_metzger_plot

# Generate more realistic sample data
temperature = np.linspace(300, 900, 1000)  # Temperature in Kelvin
true_e_a = 80000  # J/mol
true_a = 1e7  # min^-1
true_n = 1  # Reaction order
heating_rate = 10  # K/min

# Calculate conversion
r = 8.314  # Gas constant in J/(mol·K)
true_t_s = 740  # Assumed temperature of maximum decomposition rate
k = true_a * np.exp(-true_e_a / (r * temperature))
time = (temperature - temperature[0]) / heating_rate
alpha = 1 - np.exp(-(k * time) ** true_n)

# Add some noise to make it more realistic
np.random.seed(42)  # for reproducibility
noise_level = 0.02
alpha_noisy = np.clip(alpha + np.random.normal(0, noise_level, alpha.shape), 0.001, 0.999)

# Perform Horowitz-Metzger analysis
e_a, a, t_s, r_squared = horowitz_metzger_method(temperature, alpha_noisy, n=true_n)

print(f"True values: E_a = {true_e_a/1000:.2f} kJ/mol, A = {true_a:.2e} min^-1, T_s = {true_t_s:.2f} K")
print(f"Fitted values: E_a = {e_a/1000:.2f} kJ/mol, A = {a:.2e} min^-1")
print(f"Fitted temperature of maximum decomposition rate: {t_s:.2f} K")
print(f"R^2 = {r_squared:.4f}")

# Generate plot data
theta, y, _, _, _, _ = horowitz_metzger_plot(temperature, alpha_noisy, n=true_n)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(theta, y, label='Experimental data', alpha=0.5, s=10)
plt.plot(theta, np.polyval(np.polyfit(theta, y, 1), theta), 'r-', label='Fitted line')
plt.xlabel('θ (K)')
plt.ylabel('ln(-ln(1-α))')
plt.title('Horowitz-Metzger Plot')
plt.legend()
plt.grid(True)

# Add text box with results
textstr = f'E_a = {e_a/1000:.2f} kJ/mol\nA = {a:.2e} min$^{{-1}}$\nT_s = {t_s:.2f} K\nR$^2$ = {r_squared:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Calculate relative error
e_a_error = abs(e_a - true_e_a) / true_e_a * 100
a_error = abs(a - true_a) / true_a * 100
t_s_error = abs(t_s - true_t_s) / true_t_s * 100

print(f"Relative error in E_a: {e_a_error:.2f}%")
print(f"Relative error in A: {a_error:.2f}%")
print(f"Relative error in T_s: {t_s_error:.2f}%")
