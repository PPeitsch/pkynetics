import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from pkynetics.data_import import dsc_importer
from pkynetics.technique_analysis.dsc import (
    DSCAnalyzer,
    DSCExperiment,
    BaselineCorrector,
    PeakAnalyzer,
    ThermalEventDetector,
    CpCalculator,
    CpMethod,
    OperationMode,
    StabilityMethod,
)

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "pkynetics")
)
DSC_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dsc")
CP_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "heat_capacity")


def analyze_heat_capacity():
    """Example of heat capacity analysis."""
    try:
        # Import heat capacity data
        sample_data = dsc_importer(
            os.path.join(CP_DATA_DIR, "sample.txt"), manufacturer="Setaram"
        )
        blank_data = dsc_importer(
            os.path.join(CP_DATA_DIR, "zero.txt"), manufacturer="Setaram"
        )
        ref_data = dsc_importer(
            os.path.join(CP_DATA_DIR, "sapphire.txt"), manufacturer="Setaram"
        )

        # Create experiment objects
        experiments = {
            "sample": DSCExperiment(
                temperature=np.array(sample_data["temperature"], dtype=np.float64),
                heat_flow=np.array(sample_data["heat_flow"], dtype=np.float64),
                time=np.array(sample_data["time"], dtype=np.float64),
                mass=10.0,
                name="Sample",
            ),
            "blank": DSCExperiment(
                temperature=np.array(blank_data["temperature"], dtype=np.float64),
                heat_flow=np.array(blank_data["heat_flow"], dtype=np.float64),
                time=np.array(blank_data["time"], dtype=np.float64),
                mass=0.0,
                name="Blank",
            ),
            "reference": DSCExperiment(
                temperature=np.array(ref_data["temperature"], dtype=np.float64),
                heat_flow=np.array(ref_data["heat_flow"], dtype=np.float64),
                time=np.array(ref_data["time"], dtype=np.float64),
                mass=15.0,
                name="Sapphire",
            ),
        }

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Original signals
        for name, exp in experiments.items():
            ax1.plot(exp.temperature, exp.heat_flow, label=name)
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Heat Flow (mW)")
        ax1.set_title("Original Heat Capacity Measurement Signals")
        ax1.legend()
        ax1.grid(True)

        # Initialize storage for stable regions
        all_regions = {}
        colors = {"sample": "blue", "blank": "orange", "reference": "green"}

        def detect_ramps(temperature, time, min_points=100):
            """Detect temperature ramps using segmentation."""
            regions = []
            n = len(temperature)
            window = min(50, n // 20)  # Ventana móvil para promediar

            # Calcular tasa de cambio promedio en ventana móvil
            rates = []
            for i in range(0, n - window):
                rate = (temperature[i + window] - temperature[i]) / (
                    time[i + window] - time[i]
                )
                rates.append(rate)
            rates = np.array(rates)

            # Encontrar los cambios significativos en la tasa
            mean_rate = np.mean(np.abs(rates))

            i = 0
            while i < len(rates):
                if abs(rates[i]) > 0.1 * mean_rate:  # Es una rampa
                    start_idx = i
                    # Buscar el final de la rampa
                    while i < len(rates) and abs(rates[i]) > 0.1 * mean_rate:
                        i += 1
                    end_idx = min(i + window, n)

                    if end_idx - start_idx >= min_points:
                        regions.append((start_idx, end_idx))
                i += 1

            return regions

        # Process each experiment
        for name, exp in experiments.items():
            # Detect ramps
            regions = detect_ramps(exp.temperature, exp.time)
            all_regions[name] = regions

            # Plot ramp regions
            for reg_idx, (start_idx, end_idx) in enumerate(regions):
                label = name if reg_idx == 0 else None

                # Plot the ramp region
                ax2.plot(
                    exp.temperature[start_idx:end_idx],
                    exp.heat_flow[start_idx:end_idx],
                    color=colors[name],
                    label=label,
                )

        # Configure second plot
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Heat Flow (mW)")
        ax2.set_title("Ramp Regions of Heat Capacity Signals")
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(ax1.get_ylim())

        plt.tight_layout()
        plt.show()

        # Print information about detected regions
        print("\nDetected ramp regions:")
        for name, regions in all_regions.items():
            print(f"\n{name.capitalize()}:")
            for i, (start_idx, end_idx) in enumerate(regions):
                exp = experiments[name]
                rate = (exp.temperature[end_idx - 1] - exp.temperature[start_idx]) / (
                    exp.time[end_idx - 1] - exp.time[start_idx]
                )
                print(f"  Region {i + 1}:")
                print(
                    f"    Temperature range: {exp.temperature[start_idx]:.1f}°C - {exp.temperature[end_idx - 1]:.1f}°C"
                )
                print(f"    Average heating rate: {rate * 60:.1f}°C/min")
                print(f"    Points: {end_idx - start_idx}")

    except Exception as e:
        print(f"Error in heat capacity analysis: {str(e)}")
        raise


def perform_dsc_analysis():
    """Example of basic DSC analysis."""
    try:
        # Import DSC data
        data = dsc_importer(
            os.path.join(DSC_DATA_DIR, "sample_dsc_data.txt"), manufacturer="Setaram"
        )

        # Create experiment
        experiment = DSCExperiment(
            temperature=np.array(data["temperature"], dtype=np.float64),
            heat_flow=np.array(data["heat_flow"], dtype=np.float64),
            time=np.array(data["time"], dtype=np.float64),
            mass=10.0,
            name="Sample DSC",
        )

        # Initialize analyzers
        baseline_corrector = BaselineCorrector()
        analyzer = DSCAnalyzer(experiment)

        # Perform baseline correction
        baseline_result = baseline_corrector.correct(
            experiment.temperature, experiment.heat_flow, method="polynomial", degree=3
        )

        # Analyze peaks
        peaks = analyzer.peak_analyzer.find_peaks(
            experiment.temperature, baseline_result.corrected_data
        )

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot raw and corrected data
        ax1.plot(experiment.temperature, experiment.heat_flow, label="Raw Data")
        ax1.plot(
            experiment.temperature, baseline_result.baseline, "--", label="Baseline"
        )
        ax1.plot(
            experiment.temperature, baseline_result.corrected_data, label="Corrected"
        )
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Heat Flow (mW)")
        ax1.set_title("DSC Data and Baseline Correction")
        ax1.legend()
        ax1.grid(True)

        # Plot peaks
        ax2.plot(experiment.temperature, baseline_result.corrected_data)
        for peak in peaks:
            ax2.axvline(peak.peak_temperature, color="r", linestyle="--")
            ax2.annotate(
                f"Peak T = {peak.peak_temperature:.1f}°C\n"
                f"ΔH = {peak.enthalpy:.1f} J/g",
                xy=(peak.peak_temperature, peak.peak_height),
                xytext=(10, 10),
                textcoords="offset points",
            )
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Heat Flow (mW)")
        ax2.set_title("Peak Analysis")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in DSC analysis: {str(e)}")
        raise


if __name__ == "__main__":
    print("1. Analyzing heat capacity...")
    analyze_heat_capacity()

    print("\n2. Performing basic DSC analysis...")
    perform_dsc_analysis()
