"""
DSC Polymer Analysis Example
============================

This example demonstrates a complete workflow for analyzing a synthetic DSC curve
of a semi-crystalline polymer. The analysis includes:
1.  Generating synthetic data representing a glass transition, cold crystallization, and melting.
2.  Applying an automatic baseline correction.
3.  Detecting and characterizing all major thermal events.
4.  Visualizing the results with a comprehensive plot.
"""

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.technique_analysis.dsc import (
    DSCAnalyzer,
    DSCExperiment,
    ThermalEventDetector,
)


def generate_polymer_signal(
    temperature: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generates a synthetic DSC signal for a polymer."""
    # Define event temperatures
    events = {"tg": 353.15, "cryst": 423.15, "melt": 493.15}  # K

    # 1. Glass Transition (step-like change)
    tg_signal = 0.5 / (1 + np.exp(-(temperature - events["tg"]) / 5.0))

    # 2. Cold Crystallization (exothermic peak)
    cryst_signal = -1.0 * np.exp(-(((temperature - events["cryst"]) / 10.0) ** 2))

    # 3. Melting (endothermic peak)
    melt_signal = 1.5 * np.exp(-(((temperature - events["melt"]) / 12.0) ** 2))

    # Combine signals and add a sloped baseline
    baseline = 0.0005 * (temperature - temperature[0])
    noise = np.random.normal(0, 0.01, len(temperature))
    heat_flow = baseline + tg_signal + cryst_signal + melt_signal + noise

    return heat_flow, events


def main():
    """Main function to run the polymer analysis example."""
    print("--- DSC Polymer Analysis Example ---")

    # --- 1. Setup Experiment ---
    temperature = np.linspace(300, 550, 1000)  # Temperature from ~27°C to 277°C
    time = np.linspace(0, (550 - 300) / 10 * 60, 1000)  # Assuming 10 K/min
    heat_flow, expected_events = generate_polymer_signal(temperature)

    experiment = DSCExperiment(
        temperature=temperature,
        heat_flow=heat_flow,
        time=time,
        mass=10.0,
        sample_name="Synthetic Polymer",
    )

    # --- 2. Initialize and Run Analyzer ---
    # Use a specific event detector to ensure consistency
    event_detector = ThermalEventDetector(peak_prominence=0.1)
    analyzer = DSCAnalyzer(experiment=experiment, event_detector=event_detector)

    # Perform the full analysis
    results = analyzer.analyze()

    # --- 3. Print Results ---
    print("\n--- Analysis Results ---")
    if "glass_transitions" in results["events"]:
        tg = results["events"]["glass_transitions"][0]
        print(f"Detected Glass Transition (Tg): {tg.midpoint_temperature:.2f} K")
    else:
        print("Glass Transition not detected.")

    if "crystallization" in results["events"]:
        cryst = results["events"]["crystallization"][0]
        print(
            f"Detected Crystallization: Peak at {cryst.peak_temperature:.2f} K, Enthalpy: {cryst.enthalpy:.2f} J/g"
        )
    else:
        print("Crystallization not detected.")

    if "melting" in results["events"]:
        melt = results["events"]["melting"][0]
        print(
            f"Detected Melting: Peak at {melt.peak_temperature:.2f} K, Enthalpy: {melt.enthalpy:.2f} J/g"
        )
    else:
        print("Melting not detected.")

    print(f"Baseline method used: {results['baseline']['type']}")

    # --- 4. Visualization ---
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        analyzer.experiment.temperature,
        analyzer.experiment.heat_flow,
        label="Original Signal",
        color="gray",
        alpha=0.5,
    )
    ax.plot(
        analyzer.experiment.temperature,
        analyzer.corrected_heat_flow,
        label="Baseline Corrected Signal",
        color="C0",
        linewidth=2,
    )
    ax.plot(
        analyzer.experiment.temperature,
        analyzer.baseline,
        label=f"Detected Baseline ({results['baseline']['type']})",
        color="black",
        linestyle="--",
    )

    # Annotate events
    if "glass_transitions" in results["events"]:
        tg = results["events"]["glass_transitions"][0]
        ax.axvspan(
            tg.onset_temperature,
            tg.endpoint_temperature,
            color="green",
            alpha=0.2,
            label=f"Tg @ {tg.midpoint_temperature:.2f}K",
        )

    if "melting" in results["events"]:
        melt = results["events"]["melting"][0]
        ax.axvspan(
            melt.onset_temperature,
            melt.endpoint_temperature,
            color="red",
            alpha=0.2,
            label=f"Melting @ {melt.peak_temperature:.2f}K",
        )

    if "crystallization" in results["events"]:
        cryst = results["events"]["crystallization"][0]
        ax.axvspan(
            cryst.onset_temperature,
            cryst.endpoint_temperature,
            color="blue",
            alpha=0.2,
            label=f"Crystallization @ {cryst.peak_temperature:.2f}K",
        )

    ax.set_title("DSC Analysis of a Synthetic Polymer")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Heat Flow (mW)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
