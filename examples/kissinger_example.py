"""Example usage of the Kissinger method for non-isothermal kinetics analysis."""

import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods.kissinger import kissinger_method, kissinger_equation, calculate_t_p

# Set true values and generate sample data
true_e_a = 150000  # J/mol
true_a = 1e15  # min^-1
beta = np.array([2, 5, 10, 20, 50])  # K/min

t_p = calculate_t_p(true_e_a, true_a, beta)

# Add some noise to make it more realistic
noise_level = 0.01
t_p += np.random.normal(0, noise_level * t_p, t_p.shape)

# Perform Kissinger analysis
e_a, a, r_squared = kissinger_method(t_p, beta)

print(f"True values: E_a = {true_e_a/1000:.2f} kJ/mol, A = {true_a:.2e} min^-1")
print(f"Fitted values: E_a = {e_a/1000:.2f} kJ/mol, A = {a:.2e} min^-1")
print(f"R^2 = {r_squared:.4f}")

# Prepare data for plotting
x = 1000 / t_p  # Convert to 1000/T for better scale
y = np.log(beta / t_p**2)

# Calculate the fitted line
x_fit = np.linspace(min(x), max(x), 100)
y_fit = -e_a / (8.314 * 1000) * x_fit + np.log(a * 8.314 / e_a)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Experimental data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted line')
plt.xlabel('1000/T (K$^{-1}$)')
plt.ylabel('ln(β/T$_p^2$) (K$^{-1}$·min$^{-1}$)')
plt.title('Kissinger Plot')
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
