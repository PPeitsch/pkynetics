"""Example usage of the Kissinger-Akahira-Sunose (KAS) method for model-free kinetic analysis."""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.model_free_methods import kas_method


def generate_sample_data(
    e_a: float, a: float, heating_rates: List[float], t_range: Tuple[float, float]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate sample data for kinetic analysis."""
    r = 8.314  # Gas constant in J/(mol·K)
    temperature_data = []
    conversion_data = []

    for beta in heating_rates:
        t = np.linspace(*t_range, 1000)
        time = (t - t[0]) / beta
        k = a * np.exp(-e_a / (r * t))
        alpha = 1 - np.exp(-k * time)
        temperature_data.append(t)
        conversion_data.append(alpha)

    return temperature_data, conversion_data


def kas_plot_data(
    temperature_data: List[np.ndarray],
    conversion_data: List[np.ndarray],
    heating_rates: List[float],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate data for KAS plot.

    Args:
        temperature_data (List[np.ndarray]): List of temperature data arrays for each heating rate.
        conversion_data (List[np.ndarray]): List of conversion data arrays for each heating rate.
        heating_rates (List[float]): List of heating rates corresponding to the data arrays.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            - List of x values (1000/T) for each dataset
            - List of y values (ln(β/T^2)) for each dataset
    """
    x_data = []
    y_data = []

    for temp, conv, beta in zip(temperature_data, conversion_data, heating_rates):
        x = 1000 / temp  # 1000/T for better scale
        y = np.log(beta / temp**2)

        x_data.append(x)
        y_data.append(y)

    return x_data, y_data


# Set parameters for sample data generation
e_a_true = 150000  # J/mol
a_true = 1e15  # 1/s
heating_rates = [5, 10, 20, 40]  # K/min
t_range = (400, 800)  # K

# Generate sample data
temperature_data, conversion_data = generate_sample_data(
    e_a_true, a_true, heating_rates, t_range
)

# Add some noise to make it more realistic
np.random.seed(42)  # for reproducibility
for i in range(len(conversion_data)):
    noise = np.random.normal(0, 0.005, size=conversion_data[i].shape)
    conversion_data[i] = np.clip(conversion_data[i] + noise, 0, 1)

# Perform KAS analysis
activation_energy, pre_exp_factor, conv_levels, r_squared = kas_method(
    temperature_data, conversion_data, heating_rates
)

# Generate plot data
x_data, y_data = kas_plot_data(temperature_data, conversion_data, heating_rates)

# Plotting
plt.figure(figsize=(12, 10))

# Plot 1: KAS plot
plt.subplot(2, 1, 1)
for i, beta in enumerate(heating_rates):
    plt.plot(x_data[i], y_data[i], "o", label=f"{beta} K/min")

plt.xlabel("1000/T (K^-1)")
plt.ylabel("ln(β/T^2)")
plt.title("Kissinger-Akahira-Sunose (KAS) Plot")
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
print(f"Mean estimated E_a: {np.mean(activation_energy) / 1000:.2f} kJ/mol")
print(f"True ln(A): {np.log(a_true):.2f}")
print(f"Mean estimated ln(A): {np.mean(np.log(pre_exp_factor)):.2f}")
print(f"Mean R-squared: {np.mean(r_squared):.4f}")
