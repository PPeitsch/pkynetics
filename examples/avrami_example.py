"""Example usage of the improved Avrami method for isothermal crystallization kinetics."""

import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods import avrami_method, avrami_equation, calculate_half_time


def generate_sample_data(time, true_n, true_k, noise_level=0.01):
    """Generate sample data with optional noise."""
    relative_crystallinity = avrami_equation(time, true_k, true_n)
    if noise_level > 0:
        np.random.seed(42)  # for reproducibility
        noise = np.random.normal(0, noise_level, len(time))
        relative_crystallinity = np.clip(relative_crystallinity + noise, 0, 1)
    return relative_crystallinity


def plot_results(time, relative_crystallinity, fitted_curve, n, k, r_squared):
    """Plot the results of Avrami analysis."""
    plt.figure(figsize=(12, 8))

    # Data and fitted curve
    plt.subplot(2, 1, 1)
    plt.scatter(time, relative_crystallinity, label='Experimental data', alpha=0.5)
    plt.plot(time, fitted_curve, 'r-', label='Fitted curve')
    plt.xlabel('Time')
    plt.ylabel('Relative Crystallinity')
    plt.title('Avrami Analysis of Isothermal Crystallization')
    plt.legend()
    plt.grid(True)

    # Avrami plot
    plt.subplot(2, 1, 2)
    mask = (relative_crystallinity > 0.01) & (relative_crystallinity < 0.99)  # Focus on relevant data
    y = np.log(-np.log(1 - relative_crystallinity[mask]))
    x = np.log(time[mask])
    plt.scatter(x, y, label='Avrami plot', alpha=0.5)
    plt.plot(x, n * x + np.log(k) * n, 'r-', label='Linear fit')
    plt.xlabel('log(Time)')
    plt.ylabel('log(-log(1-X))')
    plt.title('Avrami Plot')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # Generate sample data
    time = np.linspace(0, 200, 400)  # Extend time range to show complete S-curve
    true_n, true_k = 2.5, 0.01
    relative_crystallinity = generate_sample_data(time, true_n, true_k, noise_level=0.01)

    # Perform Avrami analysis
    n, k, r_squared = avrami_method(time, relative_crystallinity)

    # Generate fitted curve
    fitted_curve = avrami_equation(time, k, n)

    # Calculate half-time
    t_half = calculate_half_time(k, n)

    # Plot results
    plot_results(time, relative_crystallinity, fitted_curve, n, k, r_squared)

    # Print results
    print(f"True values: n = {true_n}, k = {true_k}")
    print(f"Fitted values: n = {n:.3f}, k = {k:.3e}")
    print(f"R^2 = {r_squared:.3f}")
    print(f"Half-time of crystallization: {t_half:.2f}")

    # Calculate relative errors
    n_error = abs(n - true_n) / true_n * 100
    k_error = abs(k - true_k) / true_k * 100
    print(f"Relative error in n: {n_error:.2f}%")
    print(f"Relative error in k: {k_error:.2f}%")


if __name__ == "__main__":
    main()
