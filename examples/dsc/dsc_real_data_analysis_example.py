"""
Legacy DSC analysis example.

NOTE: This script demonstrates a practical analysis workflow on a real dataset
containing both a heating and a cooling segment. It shows how to first
segment the data and then analyze each segment individually.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.data_import import dsc_importer
from pkynetics.technique_analysis.dsc import (
    BaselineCorrector,
    DSCExperiment,
    PeakAnalyzer,
    ThermalEventDetector,
)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_data_file_path() -> Path:
    """Constructs a robust path to the data file."""
    try:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir
        while (
            not (project_root / "src").exists() and project_root.parent != project_root
        ):
            project_root = project_root.parent

        if not (project_root / "src").exists():
            raise FileNotFoundError("Could not determine project root.")

        file_path = (
            project_root
            / "src"
            / "pkynetics"
            / "data"
            / "dsc"
            / "sample_dsc_setaram.txt"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found at expected path: {file_path}"
            )

        return file_path
    except Exception as e:
        logger.error(f"Error determining data file path: {e}")
        raise


def segment_dsc_data(experiment: DSCExperiment) -> List[DSCExperiment]:
    """Segments a DSC experiment into heating and cooling runs."""
    segments = []
    temp_diff = np.diff(experiment.temperature)

    # Find points where the direction of temperature change flips
    split_indices = np.where(np.sign(temp_diff[:-1]) != np.sign(temp_diff[1:]))[0] + 1

    start_idx = 0
    for end_idx in np.append(split_indices, len(experiment.temperature)):
        if end_idx - start_idx > 100:  # Minimum segment length
            segment_temp = experiment.temperature[start_idx:end_idx]

            # Determine if it's a heating or cooling segment
            direction = "Heating" if segment_temp[-1] > segment_temp[0] else "Cooling"

            # For analysis, temperature must be monotonic. Sort cooling segments.
            if direction == "Cooling":
                sort_indices = np.argsort(segment_temp)
            else:
                sort_indices = slice(None)  # No sorting needed for heating

            segments.append(
                DSCExperiment(
                    temperature=experiment.temperature[start_idx:end_idx][sort_indices],
                    heat_flow=experiment.heat_flow[start_idx:end_idx][sort_indices],
                    time=experiment.time[start_idx:end_idx][sort_indices],
                    mass=experiment.mass,
                    sample_name=f"{experiment.sample_name} - {direction} Segment",
                )
            )
        start_idx = end_idx

    return segments


def analyze_segment(segment: DSCExperiment) -> Dict[str, Any]:
    """Analyzes a single DSC segment."""
    logger.info(f"Analyzing: {segment.sample_name}")

    # Define baseline regions from the start and end of the data
    num_points = len(segment.temperature)
    start_region_end_temp = segment.temperature[int(num_points * 0.15)]
    end_region_start_temp = segment.temperature[int(num_points * 0.85)]
    baseline_regions = [
        (segment.temperature[0], start_region_end_temp),
        (end_region_start_temp, segment.temperature[-1]),
    ]

    # Perform analysis
    baseline_corrector = BaselineCorrector()
    baseline_result = baseline_corrector.correct(
        segment.temperature,
        segment.heat_flow,
        method="linear",
        regions=baseline_regions,
    )

    peak_analyzer = PeakAnalyzer()
    peaks = peak_analyzer.find_peaks(
        segment.temperature, baseline_result.corrected_data
    )

    event_detector = ThermalEventDetector()
    events = event_detector.detect_events(
        segment.temperature, baseline_result.corrected_data, peaks
    )

    return {"baseline_result": baseline_result, "peaks": peaks, "events": events}


def main():
    """Main execution function."""
    try:
        file_path = get_data_file_path()
        logger.info(f"Loading data from: {file_path}")

        data = dsc_importer(file_path=str(file_path), manufacturer="Setaram")
        full_experiment = DSCExperiment(
            temperature=data["temperature"],
            heat_flow=data["heat_flow"],
            time=data["time"],
            mass=10.0,
            sample_name="Setaram Real Sample",
        )

        # 1. Segment the data into heating and cooling runs
        segments = segment_dsc_data(full_experiment)
        logger.info(f"Detected {len(segments)} segments in the data.")

        # 2. Analyze each segment and store results
        all_results = []
        for segment in segments:
            all_results.append(analyze_segment(segment))

        # 3. Plot the results for all segments
        fig, axes = plt.subplots(
            len(segments),
            2,
            figsize=(16, 8 * len(segments)),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        if len(segments) == 1:
            axes = np.array([axes])  # Ensure axes is always 2D

        fig.suptitle("DSC Analysis by Segments", fontsize=16)

        for i, (segment, results) in enumerate(zip(segments, all_results)):
            ax1, ax2 = axes[i]

            # Plot 1: Baseline correction
            ax1.set_title(f"{segment.sample_name}: Baseline Correction")
            ax1.plot(
                segment.temperature,
                segment.heat_flow,
                label="Original",
                color="gray",
                alpha=0.7,
            )
            ax1.plot(
                segment.temperature,
                results["baseline_result"].baseline,
                "k--",
                label="Baseline",
            )
            ax1.plot(
                segment.temperature,
                results["baseline_result"].corrected_data,
                label="Corrected",
                color="C0",
            )
            ax1.set_ylabel("Heat Flow (mW)")
            ax1.legend()

            # Plot 2: Detected events on corrected signal
            ax2.set_title("Detected Events")
            ax2.plot(
                segment.temperature,
                results["baseline_result"].corrected_data,
                color="C0",
            )

            events = results.get("events", {})
            if events.get("melting"):
                for event in events["melting"]:
                    ax2.axvline(
                        event.peak_temperature,
                        color="red",
                        ls="--",
                        label=f"Melting: {event.peak_temperature:.1f}K",
                    )
            if events.get("crystallization"):
                for event in events["crystallization"]:
                    ax2.axvline(
                        event.peak_temperature,
                        color="purple",
                        ls="-.",
                        label=f"Crystallization: {event.peak_temperature:.1f}K",
                    )

            ax2.legend()

        axes[-1, 0].set_xlabel("Temperature (K)")
        axes[-1, 1].set_xlabel("Temperature (K)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
