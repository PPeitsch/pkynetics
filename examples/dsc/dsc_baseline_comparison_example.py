"""
DSC Baseline Correction Comparison Example
==========================================

This example demonstrates the application of various baseline correction methods
available in the `BaselineCorrector`.

The workflow includes:
1.  Generating synthetic data with a known non-linear baseline and a peak.
2.  Applying several different baseline correction methods:
    - Linear
    - Polynomial (degree 3)
    - Asymmetric Least Squares
    - Rubberband (Convex Hull)
3.  Visualizing the original signal, the true baseline, and the results of each
    correction method for a direct visual comparison.
"""

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.technique_analysis.dsc import BaselineCorrector


def generate_baseline_data(temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generates data with a complex baseline and a single peak."""
    # A curved, non-linear baseline
    true_baseline = 0.001 * (temperature - 300) + 2e-5 * (temperature - 300) ** 2

    # A single endothermic peak
    peak = 1.2 * np.exp(-(((temperature - 450) / 25.0) ** 2))

    noise = np.random.normal(0, 0.02, len(temperature))

    heat_flow = true_baseline + peak + noise
    return heat_flow, true_baseline


def main():
    """Main function to run the baseline comparison example."""
    print("--- DSC Baseline Correction Comparison Example ---")

    temperature = np.linspace(300, 600, 500)
    heat_flow, true_baseline = generate_baseline_data(temperature)

    corrector = BaselineCorrector()

    # --- Apply different baseline methods ---
    methods_to_compare = ["linear", "polynomial", "asymmetric", "rubberband"]
    results = {}

    for method in methods_to_compare:
        print(f"Applying '{method}' baseline correction...")
        # For polynomial, we can pass extra arguments
        kwargs = {"degree": 3} if method == "polynomial" else {}
        try:
            baseline_result = corrector.correct(
                temperature, heat_flow, method=method, **kwargs
            )
            results[method] = baseline_result
        except Exception as e:
            print(f"Could not apply method '{method}': {e}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, method in enumerate(methods_to_compare):
        if method in results:
            ax = axes[i]
            result = results[method]

            ax.plot(
                temperature, heat_flow, label="Original Signal", color="gray", alpha=0.7
            )
            ax.plot(temperature, true_baseline, "k--", label="True Baseline")
            ax.plot(
                temperature, result.baseline, "r-", linewidth=2, label=f"'{method}' Fit"
            )

            ax.set_title(f"Method: {method.capitalize()}")
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Heat Flow (mW)")
            ax.legend()
            ax.grid(True)

    fig.suptitle("Comparison of DSC Baseline Correction Methods", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
