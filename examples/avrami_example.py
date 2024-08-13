"""Example usage of the Avrami method for isothermal crystallization kinetics."""

import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods import avrami_method, avrami_equation

# Generate sample data
time = np.linspace(0, 100, 100)
true_n, true_k = 2.5, 0.01
relative_crystallinity = avrami_equation(time, true_k, true_n)

# Add some noise to the data
np.random.seed(42)  # for reproducibility
noise = np.random.normal(0, 0.01, len(time))
noisy_crystallinity = np.clip(relative_crystallinity + noise, 0, 1)

# Perform Avrami analysis
n, k, r_squared = avrami_method(time, noisy_crystallinity)

# Print results
print(f"True values: n = {true_n}, k = {true_k}")
print(f"Fitted values: n = {n:.3f}, k = {k:.3e}")
print(f"R^2 = {r_squared:.3f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(time, noisy_crystallinity, label='Noisy data', alpha=0.5)
plt.plot(time, relative_crystallinity, label='True curve', linestyle='--')
plt.plot(time, avrami_equation(time, k, n), label='Fitted curve', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Relative Crystallinity')
plt.title('Avrami Analysis of Isothermal Crystallization')
plt.legend()
plt.grid(True)
plt.show()
