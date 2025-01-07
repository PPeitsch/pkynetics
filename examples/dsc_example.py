import os
from typing import Dict, List, Optional, Tuple

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

        # Initialize Cp calculator
        calculator = CpCalculator()

        # Get reference sapphire data
        ref_sapphire_temp = calculator._reference_data["sapphire"]["temperature"]
        ref_sapphire_cp = calculator._reference_data["sapphire"]["cp"]

        # Create interpolation function for sapphire Cp
        sapphire_cp_interp = interp1d(
            ref_sapphire_temp, ref_sapphire_cp, kind="linear", fill_value="extrapolate"
        )

        # Interpolate sapphire Cp to match experimental temperature points
        interpolated_sapphire_cp = sapphire_cp_interp(experiments["sample"].temperature)

        # Calculate Cp using different methods
        results = {
            "standard": calculator.calculate_cp(
                sample_data={
                    "temperature": experiments["sample"].temperature,
                    "sample_heat_flow": experiments["sample"].heat_flow,
                    "time": experiments["sample"].time,
                    "sample_mass": experiments["sample"].mass,
                    "reference_heat_flow": experiments["reference"].heat_flow,
                    "baseline_heat_flow": experiments["blank"].heat_flow,
                    "reference_mass": experiments["reference"].mass,
                    "reference_cp": interpolated_sapphire_cp,
                },
                method="standard",
            ),
            "continuous": calculator.calculate_cp(
                sample_data={
                    "temperature": experiments["sample"].temperature,
                    "heat_flow": experiments["sample"].heat_flow,
                    "time": experiments["sample"].time,
                    "sample_mass": experiments["sample"].mass,
                    "heating_rate": experiments["sample"].heating_rate,
                },
                method="continuous",
            ),
        }

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot heat flow signals
        for exp in experiments.values():
            ax1.plot(exp.temperature, exp.heat_flow, label=exp.name)
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Heat Flow (mW)")
        ax1.set_title("Heat Capacity Measurement Signals")
        ax1.legend()
        ax1.grid(True)

        # Plot calculated Cp
        for method, result in results.items():
            ax2.plot(
                result.temperature,
                result.specific_heat,
                label=f"{method.capitalize()} Method",
            )
            # Plot uncertainty bands
            ax2.fill_between(
                result.temperature,
                result.specific_heat - result.uncertainty,
                result.specific_heat + result.uncertainty,
                alpha=0.2,
            )
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Specific Heat (J/g·K)")
        ax2.set_title("Specific Heat Capacity Results")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in heat capacity analysis: {str(e)}")
        raise


def perform_dsc_analysis():
    """Example of basic DSC analysis."""
    try:
        # Import DSC data
        data = dsc_importer(
            os.path.join(DSC_DATA_DIR, "sample_dsc_setaram.txt"), manufacturer="Setaram"
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
