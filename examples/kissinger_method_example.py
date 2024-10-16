import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods.kissinger import kissinger_method, calculate_t_p

# Set true values and generate sample data
true_e_a = 150000  # J/mol
true_a = 1e15  # min^-1
beta = np.array([2, 5, 10, 20, 50])  # K/min
r = 8.314  # Gas constant in J/(mol·K)

t_p = calculate_t_p(true_e_a, true_a, beta)

# Add some noise to make it more realistic
np.random.seed(42)  # for reproducibility
noise_level = 0.005
t_p_noisy = t_p + np.random.normal(0, noise_level * t_p, t_p.shape)

# Perform Kissinger analysis
e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(t_p_noisy, beta)

print(f"True values: E_a = {true_e_a/1000:.2f} kJ/mol, A = {true_a:.2e} min^-1")
print(f"Fitted values: E_a = {e_a/1000:.2f} ± {se_e_a/1000:.2f} kJ/mol, A = {a:.2e} min^-1")
print(f"95% Confidence Interval for E_a: [{(e_a-1.96*se_e_a)/1000:.2f}, {(e_a+1.96*se_e_a)/1000:.2f}] kJ/mol")
print(f"95% Confidence Interval for ln(A): [{np.log(a)-1.96*se_ln_a:.2f}, {np.log(a)+1.96*se_ln_a:.2f}]")
print(f"R^2 = {r_squared:.4f}")

# Prepare data for plotting
x_exp = 1000 / t_p_noisy
y_exp = np.log(beta / t_p_noisy**2)

# Generate theoretical curves
beta_theory = np.logspace(np.log10(min(beta)), np.log10(max(beta)), 100)
t_p_true = calculate_t_p(true_e_a, true_a, beta_theory)
t_p_fit = calculate_t_p(e_a, a, beta_theory)

x_theory_true = 1000 / t_p_true
y_theory_true = np.log(beta_theory / t_p_true**2)

x_theory_fit = 1000 / t_p_fit
y_theory_fit = np.log(beta_theory / t_p_fit**2)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x_exp, y_exp, label='Experimental data')
plt.plot(x_theory_true, y_theory_true, 'g--', label='True theoretical curve')
plt.plot(x_theory_fit, y_theory_fit, 'r-', label='Fitted theoretical curve')
plt.xlabel('1000/T (K$^{-1}$)')
plt.ylabel('ln(β/T$_p^2$) (K$^{-1}$·min$^{-1}$)')
plt.title('Kissinger Plot')
plt.legend()
plt.grid(True)

# Add text box with results
textstr = f'E_a = {e_a/1000:.2f} ± {se_e_a/1000:.2f} kJ/mol\nA = {a:.2e} min$^{{-1}}$\nR$^2$ = {r_squared:.4f}'
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
