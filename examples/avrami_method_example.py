import numpy as np
from model_fitting_methods import jmak_method, jmak_equation, calculate_half_time
from synthetic_data.model_specific_data import generate_avrami_data
from result_visualization.kinetic_plot import plot_avrami_results


def main():
    # Generate sample data
    time = np.linspace(0, 200, 400)
    true_n, true_k = 2.5, 0.01
    relative_crystallinity = generate_avrami_data(time, true_n, true_k, noise_level=0.01)

    try:
        # Perform Avrami analysis
        n, k, r_squared = avrami_method(time, relative_crystallinity)

        # Generate fitted curve
        fitted_curve = avrami_equation(time, k, n)

        # Calculate half-time
        t_half = calculate_half_time(k, n)

        # Plot results
        plot_avrami_results(time, relative_crystallinity, fitted_curve, n, k, r_squared, t_half)

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

    except ValueError as e:
        print(f"Error in Avrami analysis: {str(e)}")


if __name__ == "__main__":
    main()
