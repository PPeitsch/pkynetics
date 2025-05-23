import numpy as np

from pkynetics.model_fitting_methods import jmak_equation, jmak_method
from pkynetics.result_visualization import plot_jmak_results
from pkynetics.synthetic_data import generate_jmak_data


def main():
    # Generate sample data
    time = np.linspace(0, 200, 400)
    true_n, true_k = 2.5, 0.01
    transformed_fraction = generate_jmak_data(time, true_n, true_k, noise_level=0.01)

    try:
        # Perform JMAK analysis
        n, k, r_squared = jmak_method(time, transformed_fraction)

        # Generate fitted curve
        fitted_curve = jmak_equation(time, k, n)

        # Plot results
        plot_jmak_results(time, transformed_fraction, fitted_curve, n, k, r_squared)

        # Print results
        print(f"True values: n = {true_n}, k = {true_k}")
        print(f"Fitted values: n = {n:.3f}, k = {k:.3e}")
        print(f"R^2 = {r_squared:.3f}")

        # Calculate relative errors
        n_error = abs(n - true_n) / true_n * 100
        k_error = abs(k - true_k) / true_k * 100
        print(f"Relative error in n: {n_error:.2f}%")
        print(f"Relative error in k: {k_error:.2f}%")

    except ValueError as e:
        print(f"Error in JMAK analysis: {str(e)}")


if __name__ == "__main__":
    main()
