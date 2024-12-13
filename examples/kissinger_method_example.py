import numpy as np
import matplotlib.pyplot as plt
from src.pkynetics.model_fitting_methods.kissinger import kissinger_method, calculate_t_p, kissinger_equation
from src.pkynetics.synthetic_data import generate_basic_kinetic_data

# Set true values for synthetic data generation
true_e_a = 40000  # J/mol
true_a = 3.9e6  # s^-1
heating_rates = np.array([20, 40, 60, 80])  # °C/min
t_range = (200, 600)  # °C
R = 8.314  # Gas constant in J/(mol·K)

# Generate synthetic data
temp_data, conv_data = generate_basic_kinetic_data(
    e_a=true_e_a,
    a=true_a,
    heating_rates=heating_rates,
    t_range=t_range,
    reaction_model='first_order',
    noise_level=0.01
)

# Plot conversion curves for each heating rate
plt.figure(figsize=(10, 6))
for i, beta in enumerate(heating_rates):
    plt.plot(temp_data[i], conv_data[i], label=f'{beta} °C/min')

plt.xlabel('Temperature (°C)')
plt.ylabel('Conversion')
plt.title('Synthetic Conversion Curves for Different Heating Rates')
plt.legend()
plt.grid(True)
plt.show()

# Calculate peak temperatures using the calculate_t_p function
t_p = calculate_t_p(true_e_a, true_a, heating_rates)

# Perform Kissinger analysis
e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(t_p, heating_rates)

print(f"True values: E_a = {true_e_a/1000:.2f} kJ/mol, A = {true_a:.2e} s^-1")
print(f"Fitted values: E_a = {e_a/1000:.2f} ± {se_e_a/1000:.2f} kJ/mol, A = {a:.2e} s^-1")
print(f"95% Confidence Interval for E_a: [{(e_a-1.96*se_e_a)/1000:.2f}, {(e_a+1.96*se_e_a)/1000:.2f}] kJ/mol")
print(f"95% Confidence Interval for ln(A): [{np.log(a)-1.96*se_ln_a:.2f}, {np.log(a)+1.96*se_ln_a:.2f}]")
print(f"R^2 = {r_squared:.4f}")

# Prepare data for plotting
x_exp = 1000 / t_p
y_exp = kissinger_equation(t_p=t_p, beta=heating_rates)

# Generate theoretical curves
heating_rates_theory = np.logspace(np.log10(min(heating_rates)), np.log10(max(heating_rates)), 100)
t_p_true = calculate_t_p(true_e_a, true_a, heating_rates_theory)
t_p_fit = calculate_t_p(e_a, a, heating_rates_theory)

x_theory_true = 1000 / t_p_true
y_theory_true = np.log(heating_rates_theory / t_p_true**2)

x_theory_fit = 1000 / t_p_fit
y_theory_fit = np.log(heating_rates_theory / t_p_fit**2)

# Plot the Kissinger plot
plt.figure(figsize=(10, 6))
plt.scatter(x_exp, y_exp, label='Synthetic data')
plt.plot(x_theory_true, y_theory_true, 'g--', label='True theoretical curve')
plt.plot(x_theory_fit, y_theory_fit, 'r-', label='Fitted theoretical curve')
plt.xlabel('1000/T (K$^{-1}$)')
plt.ylabel('ln(β/T$_p^2$) (K$^{-1}$·min$^{-1}$)')
plt.title('Kissinger Plot')
plt.legend()
plt.grid(True)

# Add text box with results
textstr = f'E_a = {e_a/1000:.2f} ± {se_e_a/1000:.2f} kJ/mol\nA = {a:.2e} s$^{{-1}}$\nR$^2$ = {r_squared:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Calculate relative error
e_a_error = abs(e_a - true_e_a) / true_e_a * 100
a_error = abs(a - true_a) / true_a * 100
ln_a_error = abs(np.log(a) - np.log(true_a)) / np.log(true_a) * 100

print(f"Relative error in E_a: {e_a_error:.2f}%")
print(f"Relative error in A: {a_error:.2f}%")
print(f"Relative error in ln(A): {ln_a_error:.2f}%")