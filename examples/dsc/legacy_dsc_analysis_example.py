"""
Legacy DSC analysis example.

NOTE: This script is a simplified version of the analysis workflow and serves
as a legacy example. For more detailed and specialized use cases, please
refer to the other examples in the `examples/dsc/` directory.

This script demonstrates:
- Loading a DSC dataset.
- Applying baseline correction.
- Detecting peaks.
- Identifying thermal events.
- Plotting the results.
"""

import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.data_import import dsc_importer
from pkynetics.technique_analysis.dsc.baseline import BaselineCorrector
from pkynetics.technique_analysis.dsc.peak_analysis import PeakAnalyzer
from pkynetics.technique_analysis.dsc.thermal_events import ThermalEventDetector
from pkynetics.technique_analysis.dsc.types import DSCExperiment

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def analyze_experiment(experiment: DSCExperiment) -> Dict:
    """
    Analyzes a full DSC experiment or a segment.
    """
    results = {}
    logger.info(
        f"Starting analysis of '{experiment.sample_name}' with {len(experiment.temperature)} points."
    )

    try:
        # 1. Baseline Correction
        logger.info("Performing baseline correction...")
        baseline_corrector = BaselineCorrector()
        baseline_result = baseline_corrector.correct(
            experiment.temperature, experiment.heat_flow, method="asymmetric"
        )
        results["baseline"] = baseline_result
        logger.info(f"Baseline correction complete. Method: {baseline_result.method}")

        # 2. Peak Analysis
        logger.info("Finding peaks...")
        peak_analyzer = PeakAnalyzer()
        # Use the corrected data for peak finding
        peaks = peak_analyzer.find_peaks(
            experiment.temperature,
            baseline_result.corrected_data,
            baseline=baseline_result.baseline,
        )
        results["peaks"] = peaks
        logger.info(f"Found {len(peaks)} peaks.")

        # 3. Thermal Event Detection
        logger.info("Detecting thermal events...")
        event_detector = ThermalEventDetector()
        events = event_detector.detect_events(
            experiment.temperature, baseline_result.corrected_data, peaks
        )
        results["events"] = events
        logger.info("Thermal event detection complete.")

    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}", exc_info=True)
        results["error"] = str(e)

    return results


def plot_analysis_results(experiment: DSCExperiment, results: Dict) -> None:
    """
    Plots the results of the DSC analysis.
    """
    if "error" in results:
        logger.error(f"Cannot plot results due to analysis error: {results['error']}")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.suptitle(f"DSC Analysis: {experiment.sample_name}", fontsize=16)

    # --- Top Plot: Data, Baseline, and Corrected Signal ---
    ax1.plot(
        experiment.temperature,
        experiment.heat_flow,
        label="Original Signal",
        color="blue",
        alpha=0.6,
    )
    if "baseline" in results:
        ax1.plot(
            experiment.temperature,
            results["baseline"].baseline,
            "r--",
            label=f"'{results['baseline'].method}' Baseline",
        )
        ax1.plot(
            experiment.temperature,
            results["baseline"].corrected_data,
            label="Baseline-Corrected Signal",
            color="green",
            linewidth=2,
        )
    ax1.set_ylabel("Heat Flow (mW)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_title("Baseline Correction and Signal")

    # --- Bottom Plot: Annotations for Events ---
    ax2.plot(
        experiment.temperature,
        results.get("baseline", {}).get("corrected_data", experiment.heat_flow),
        color="green",
        alpha=0.7,
    )

    if "events" in results:
        if "melting" in results["events"]:
            for event in results["events"]["melting"]:
                ax2.axvline(
                    event.peak_temperature, color="red", linestyle="--", label="Melting"
                )
        if "crystallization" in results["events"]:
            for event in results["events"]["crystallization"]:
                ax2.axvline(
                    event.peak_temperature,
                    color="purple",
                    linestyle="--",
                    label="Crystallization",
                )
        if "glass_transitions" in results["events"]:
            for event in results["events"]["glass_transitions"]:
                ax2.axvspan(
                    event.onset_temperature,
                    event.endpoint_temperature,
                    color="orange",
                    alpha=0.3,
                    label="Glass Transition (Tg)",
                )

    # Avoid duplicate labels in legend
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())

    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Corrected Heat Flow (mW)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_title("Detected Thermal Events")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    """
    Main execution function.
    """
    # Define the path to the sample data file
    # This assumes the script is run from the project's root directory.
    file_path = os.path.join(
        "src", "pkynetics", "data", "dsc", "sample_dsc_setaram.txt"
    )
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at: {file_path}")
        logger.error(
            "Please ensure you are running this script from the project root directory."
        )
        return

    try:
        # Load data using the dsc_importer
        data = dsc_importer(file_path=file_path, manufacturer="Setaram")
        experiment = DSCExperiment(
            temperature=data["temperature"],
            heat_flow=data["heat_flow"],
            time=data["time"],
            mass=10.0,
            sample_name="Setaram Sample",
        )

        # Analyze the full experiment
        analysis_results = analyze_experiment(experiment)

        # Plot the results
        plot_analysis_results(experiment, analysis_results)

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
