"""
DSC Baseline Correction Comparison Example
==========================================

This example demonstrates and compares various baseline correction methods
available in the `BaselineCorrector` across different common scenarios.

The workflow includes:
1.  Generating three types of synthetic DSC data:
    - A simple endothermic peak on a non-linear heating curve.
    - A sharp solid-solid transition on a heating curve.
    - An exothermic peak on a cooling curve.
2.  Applying several correction methods to each dataset, demonstrating how to
    properly use regions for an accurate fit.
3.  Visualizing the results for each scenario in separate plots for a clear
    and direct comparison.
"""

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.technique_analysis.dsc import BaselineCorrector


def generate_heating_peak_data(temp: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Generates a heating curve with a non-linear baseline and an endothermic peak."""
    true_baseline = 0.001 * (temp - 300) + 2e-5 * (temp - 300) ** 2
    peak = 1.2 * np.exp(-(((temp - 450) / 25.0) ** 2))
    noise = np.random.normal(0, 0.02, len(temp))
    regions = [(300, 400), (500, 600)]
    return true_baseline + peak + noise, true_baseline, regions


def generate_solid_solid_transition_data(
    temp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Generates a heating curve with a sharp solid-solid transition."""
    true_baseline = 0.2 + 0.001 * (temp - 300)
    # A sharp, reversible solid-solid transition peak
    peak = 0.8 * np.exp(-(((temp - 380) / 8.0) ** 2))
    noise = np.random.normal(0, 0.015, len(temp))
    regions = [(300, 350), (410, 600)]
    return true_baseline + peak, true_baseline, regions


def generate_cooling_peak_data(
    temp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Generates a cooling curve with an exothermic peak."""
    true_baseline = 0.5 - 0.0008 * (temp[0] - temp)
    peak = -1.5 * np.exp(-(((temp - 420) / 20.0) ** 2))
    noise = np.random.normal(0, 0.02, len(temp))
    regions = [(300, 380), (500, 600)]
    return true_baseline + peak + noise, true_baseline, regions


def plot_comparison(
    title: str,
    temperature: np.ndarray,
    heat_flow: np.ndarray,
    true_baseline: np.ndarray,
    regions: list,
):
    """Plots a 2x2 grid comparing baseline correction methods for a given dataset."""
    corrector = BaselineCorrector()
    methods_to_compare = ["linear", "polynomial", "asymmetric", "rubberband"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    is_exothermic = np.min(heat_flow - true_baseline) < -0.1

    for i, method in enumerate(methods_to_compare):
        ax = axes[i]
        kwargs = {"degree": 3} if method == "polynomial" else {}

        if method in ["linear", "polynomial"]:
            kwargs["regions"] = regions

        try:
            if method in ["asymmetric", "rubberband"] and is_exothermic:
                inverted_hf = -heat_flow
                result = corrector.correct(
                    temperature, inverted_hf, method=method, **kwargs
                )
                result.baseline = -result.baseline
            else:
                result = corrector.correct(
                    temperature, heat_flow, method=method, **kwargs
                )

            ax.plot(temperature, heat_flow, label="Signal", color="gray", alpha=0.7)
            ax.plot(temperature, true_baseline, "k--", label="True Baseline")
            ax.plot(temperature, result.baseline, "r-", linewidth=2, label="Fit")
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
    hf_heating, bl_heating, regions_heating = generate_heating_peak_data(temp_heating)
    plot_comparison(
        "Scenario 1: Heating with Endothermic Peak",
        temp_heating,
        hf_heating,
        bl_heating,
        regions_heating,
    )

    # --- Scenario 2: Heating Curve with Solid-Solid Transition ---
    hf_trans, bl_trans, regions_trans = generate_solid_solid_transition_data(
        temp_heating
    )
    plot_comparison(
        "Scenario 2: Heating with Solid-Solid Transition",
        temp_heating,
        hf_trans,
        bl_trans,
        regions_trans,
    )

    # --- Scenario 3: Cooling Curve with Exothermic Peak ---
    temp_cooling = np.linspace(600, 300, 501)
    hf_cooling, bl_cooling, regions_cooling = generate_cooling_peak_data(temp_cooling)
    plot_comparison(
        "Scenario 3: Cooling with Exothermic Peak",
        temp_cooling,
        hf_cooling,
        bl_cooling,
        regions_cooling,
    )


if __name__ == "__main__":
    main()
