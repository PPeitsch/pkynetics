"""Example usage of the Ozawa-Flynn-Wall method for model-free kinetic analysis."""

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.model_free_methods import ofw_method
from pkynetics.synthetic_data import generate_basic_kinetic_data

# Set parameters
e_a_true = 150000  # J/mol
a_true = 1e12  # 1/s
heating_rates = [5, 10, 20, 40]  # K/min
t_range = (350, 700)  # K

# Generate data
# First-order reaction, integrated over temperature at each heating rate
temperature_data, conversion_data = generate_basic_kinetic_data(
    e_a_true, a_true, np.array(heating_rates, dtype=float), t_range, num_points=2000
)

# Add some noise
for i in range(len(conversion_data)):
    noise = np.random.normal(0, 0.005, size=conversion_data[i].shape)
    conversion_data[i] = np.clip(conversion_data[i] + noise, 0, 1)

# Perform OFW analysis
activation_energy, pre_exp_factor, conv_levels, r_squared = ofw_method(
    temperature_data, conversion_data, heating_rates
)

# Plotting
plt.figure(figsize=(12, 10))

# Plot 1: OFW plot
plt.subplot(2, 1, 1)
for i, beta in enumerate(heating_rates):
    t = temperature_data[i]
    alpha = conversion_data[i]
    plt.plot(1000 / t, np.log(beta) * np.ones_like(t), label=f"{beta} K/min")

plt.xlabel("1000/T (K^-1)")
plt.ylabel("log(β)")
plt.title("Ozawa-Flynn-Wall Plot")
plt.legend()
plt.grid(True)

# Plot 2: Activation energy vs Conversion
plt.subplot(2, 1, 2)
plt.plot(conv_levels, activation_energy / 1000, "bo-")
plt.axhline(y=e_a_true / 1000, color="r", linestyle="--", label="True E_a")
plt.xlabel("Conversion (α)")
plt.ylabel("Activation Energy (kJ/mol)")
plt.title("Activation Energy vs Conversion")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print results
print(f"True E_a: {e_a_true / 1000:.2f} kJ/mol")
print(f"Mean estimated E_a: {np.nanmean(activation_energy) / 1000:.2f} kJ/mol")
print(f"True ln(A): {np.log(a_true):.2f}")
# pre_exp_factor is A/g(alpha) in 1/min: recover A in 1/s assuming first order
a_estimated = pre_exp_factor * -np.log(1 - conv_levels) / 60
print(f"Mean estimated ln(A): {np.nanmean(np.log(a_estimated)):.2f}")
print(f"Mean R-squared: {np.nanmean(r_squared):.4f}")
