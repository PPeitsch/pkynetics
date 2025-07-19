"""
DSC Baseline Correction Comparison Example
==========================================

This example demonstrates and compares various baseline correction methods
available in the `BaselineCorrector` across different common scenarios.

The workflow includes:
1.  Generating three types of synthetic DSC data:
    - A simple endothermic peak on a non-linear heating curve.
    - A glass transition (step-change) on a heating curve.
    - An exothermic peak on a cooling curve.
2.  Applying several correction methods to each dataset:
    - Linear, Polynomial, Asymmetric Least Squares, and Rubberband.
3.  Visualizing the results for each scenario in separate plots for a clear
    and direct comparison.
"""

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.technique_analysis.dsc import BaselineCorrector


def generate_heating_peak_data(temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generates a heating curve with a non-linear baseline and an endothermic peak."""
    true_baseline = 0.001 * (temp - 300) + 2e-5 * (temp - 300) ** 2
    peak = 1.2 * np.exp(-(((temp - 450) / 25.0) ** 2))
    noise = np.random.normal(0, 0.02, len(temp))
    return true_baseline + peak + noise, true_baseline


def generate_glass_transition_data(temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generates a heating curve with a glass transition step."""
    tg_midpoint = 400.0
    step = 0.8 / (1 + np.exp(-(temp - tg_midpoint) / 8.0))
    # Add a small relaxation peak
    relaxation_peak = 0.2 * np.exp(-(((temp - tg_midpoint - 15) / 5.0) ** 2))
    true_baseline = 0.1 + 0.0005 * temp
    noise = np.random.normal(0, 0.015, len(temp))
    return true_baseline + step + relaxation_peak + noise, true_baseline + step


def generate_cooling_peak_data(temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generates a cooling curve with an exothermic peak."""
    true_baseline = 0.5 - 0.0008 * (temp - 300)
    # Exothermic crystallization peak
    peak = -1.5 * np.exp(-(((temp - 420) / 20.0) ** 2))
    noise = np.random.normal(0, 0.02, len(temp))
    return true_baseline + peak + noise, true_baseline


def plot_comparison(
    title: str,
    temperature: np.ndarray,
    heat_flow: np.ndarray,
    true_baseline: np.ndarray,
):
    """Plots a 2x2 grid comparing baseline correction methods for a given dataset."""
    corrector = BaselineCorrector()
    methods_to_compare = ["linear", "polynomial", "asymmetric", "rubberband"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, method in enumerate(methods_to_compare):
        ax = axes[i]
        kwargs = {"degree": 3} if method == "polynomial" else {}
        try:
            result = corrector.correct(temperature, heat_flow, method=method, **kwargs)
            ax.plot(temperature, heat_flow, label="Signal", color="gray", alpha=0.7)
            ax.plot(temperature, true_baseline, "k--", label="True Baseline")
            ax.plot(temperature, result.baseline, "r-", linewidth=2, label=f"Fit")
            ax.set_title(f"Method: {method.capitalize()}")
        except Exception as e:
            ax.set_title(f"Method: {method.capitalize()} (Failed)")
            ax.text(
                0.5,
                0.5,
                f"Error: {e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    # Set common labels
    fig.suptitle(title, fontsize=18)
    for i in range(2):
        axes[i * 2].set_ylabel("Heat Flow (a.u.)")
    for i in range(2):
        axes[2 + i].set_xlabel("Temperature (K)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    """Main function to run the baseline comparison examples."""
    print("--- DSC Baseline Correction Comparison Example ---")
    np.random.seed(42)

    # --- Scenario 1: Heating Curve with Endothermic Peak ---
    temp_heating = np.linspace(300, 600, 501)
    hf_heating, bl_heating = generate_heating_peak_data(temp_heating)
    plot_comparison(
        "Scenario 1: Heating with Endothermic Peak",
        temp_heating,
        hf_heating,
        bl_heating,
    )

    # --- Scenario 2: Heating Curve with Glass Transition ---
    hf_tg, bl_tg = generate_glass_transition_data(temp_heating)
    plot_comparison(
        "Scenario 2: Heating with Glass Transition", temp_heating, hf_tg, bl_tg
    )

    # --- Scenario 3: Cooling Curve with Exothermic Peak ---
    temp_cooling = np.linspace(600, 300, 501)  # Temperature decreasing
    hf_cooling, bl_cooling = generate_cooling_peak_data(temp_cooling)
    plot_comparison(
        "Scenario 3: Cooling with Exothermic Peak", temp_cooling, hf_cooling, bl_cooling
    )


if __name__ == "__main__":
    main()
