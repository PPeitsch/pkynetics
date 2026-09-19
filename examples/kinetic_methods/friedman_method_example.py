"""Example usage of the Friedman method for model-free kinetic analysis."""

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.model_free_methods import friedman_method
from pkynetics.synthetic_data import generate_basic_kinetic_data


def friedman_plot_data(temperature, conversion, heating_rate):
    """
    Generate data for Friedman plot.

    Args:
        temperature (List[np.ndarray]): List of temperature data arrays for each heating rate.
        conversion (List[np.ndarray]): List of conversion data arrays for each heating rate.
        heating_rate (List[float]): List of heating rates.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            - List of x values (1/RT) for each dataset
            - List of y values (ln(dα/dt)) for each dataset
    """
    x_data = []
    y_data = []

    for temp, conv, beta in zip(temperature, conversion, heating_rate):
        # Calculate reaction rate
        da_dt = np.gradient(conv, temp) * beta

        # Remove any zero or negative values before taking log
        mask = da_dt > 0
        x = 1 / (8.314 * temp[mask])  # 1/RT
        y = np.log(da_dt[mask])

        x_data.append(x)
        y_data.append(y)

    return x_data, y_data


# Set parameters
e_a_true = 150000  # J/mol
a_true = 1e12  # 1/s
heating_rates = [5, 10, 20]  # K/min
t_range = (350, 700)  # K

# Generate data
# First-order reaction, integrated over temperature at each heating rate
temperature_data, conversion_data = generate_basic_kinetic_data(
    e_a_true, a_true, np.array(heating_rates, dtype=float), t_range, num_points=2000
)

# No noise here: Friedman differentiates the conversion, which amplifies
# noise. Smooth measured data before applying it.

# Perform Friedman analysis
activation_energy, pre_exp_factor, conv_levels, r_squared = friedman_method(
    temperature_data, conversion_data, heating_rates
)

# Generate plot data
x_data, y_data = friedman_plot_data(temperature_data, conversion_data, heating_rates)

# Plotting
plt.figure(figsize=(12, 10))

# Plot 1: Friedman plot
plt.subplot(2, 1, 1)
for i, beta in enumerate(heating_rates):
    plt.scatter(x_data[i], y_data[i], label=f"{beta} K/min", alpha=0.5)

plt.xlabel("1/RT (mol/J)")
plt.ylabel("ln(dα/dt) (1/s)")
plt.title("Friedman Plot")
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
# pre_exp_factor is A f(alpha) in 1/min: recover A in 1/s assuming first order
a_estimated = pre_exp_factor / (1 - conv_levels) / 60
print(f"Mean estimated ln(A): {np.nanmean(np.log(a_estimated)):.2f}")
print(f"Mean R-squared: {np.nanmean(r_squared):.4f}")
