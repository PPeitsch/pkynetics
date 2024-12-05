import matplotlib.pyplot as plt
from src.pkynetics.synthetic_data import (
    generate_basic_kinetic_data,
    add_gaussian_noise,
    add_outliers
)

# Example usage of generate_basic_kinetic_data
e_a = 100000  # J/mol
a = 1e10  # 1/s
heating_rates = [5, 10, 20]  # K/min
t_range = (300, 800)  # K

temp_data, conv_data = generate_basic_kinetic_data(
    e_a, a, heating_rates, t_range, reaction_model='nth_order', noise_level=0.01
)

# Plotting the results
plt.figure(figsize=(10, 6))
for i, beta in enumerate(heating_rates):
    plt.plot(temp_data[i], conv_data[i], label=f'β = {beta} K/min')

plt.xlabel('Temperature (K)')
plt.ylabel('Conversion')
plt.title('Synthetic Kinetic Data for Different Heating Rates')
plt.legend()
plt.grid(True)
plt.show()

# Example of adding Gaussian noise and outliers
original_data = conv_data[0]
noisy_data = add_gaussian_noise(original_data, std_dev=0.02)
outlier_data = add_outliers(noisy_data, outlier_fraction=0.05, outlier_std_dev=0.1)

plt.figure(figsize=(10, 6))
plt.plot(temp_data[0], original_data, label='Original')
plt.plot(temp_data[0], noisy_data, label='With Gaussian Noise')
plt.plot(temp_data[0], outlier_data, label='With Outliers')
plt.xlabel('Temperature (K)')
plt.ylabel('Conversion')
plt.title('Comparison of Original, Noisy, and Outlier Data')
plt.legend()
plt.grid(True)
plt.show()
