"""Example usage of the Coats-Redfern method for non-isothermal kinetics analysis."""

import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods import coats_redfern_method, coats_redfern_plot

# Generate sample data
temperature = np.linspace(300, 800, 500)  # Temperature in Kelvin
true_e_a = 100000  # J/mol
true_a = 1e10  # min^-1
true_n = 1  # Reaction order
heating_rate = 10  # K/min

# Calculate conversion
r = 8.314  # Gas constant in J/(mol·K)
k = true_a * np.exp(-true_e_a / (r * temperature))
alpha = 1 - np.exp(-k * temperature / heating_rate)

# Add some noise to make it more realistic
noise_level = 0.01
alpha_noisy = np.clip(alpha + np.random.normal(0, noise_level, alpha.shape), 0, 0.99)

# Perform Coats-Redfern analysis
e_a, a, r_squared = coats_redfern_method(temperature, alpha_noisy, heating_rate, n=true_n)

print(f"True values: E_a = {true_e_a/1000:.2f} kJ/mol, A = {true_a:.2e} min^-1")
print(f"Fitted values: E_a = {e_a/1000:.2f} kJ/mol, A = {a:.2e} min^-1")
print(f"R^2 = {r_squared:.4f}")

# Generate plot data
x, y, _, _, _ = coats_redfern_plot(temperature, alpha_noisy, heating_rate, n=true_n)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Experimental data')
plt.plot(x, np.polyval(np.polyfit(x, y, 1), x), 'r-', label='Fitted line')
plt.xlabel('1000/T (K$^{-1}$)')
plt.ylabel('ln(-ln(1-α)/T$^2$)' if true_n == 1 else 'ln((1-(1-α)$^{1-n}$)/((1-n)T$^2$))')
plt.title('Coats-Redfern Plot')
plt.legend()
plt.grid(True)

# Add text box with results
textstr = f'E_a = {e_a/1000:.2f} kJ/mol\nA = {a:.2e} min$^{{-1}}$\nR$^2$ = {r_squared:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Calculate relative error
e_a_error = abs(e_a - true_e_a) / true_e_a * 100
a_error = abs(a - true_a) / true_a * 100

print(f"Relative error in E_a: {e_a_error:.2f}%")
print(f"Relative error in A: {a_error:.2f}%")
