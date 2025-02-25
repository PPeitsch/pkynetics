"""Example of importing and analyzing dilatometry data with enhanced visualization."""

import os
import logging
import argparse
from typing import Tuple
import numpy as np

from pkynetics.data_import import dilatometry_importer
from pkynetics.data_preprocessing import smooth_data
from pkynetics.technique_analysis import analyze_dilatometry_curve
from pkynetics.technique_analysis.utilities import (
    analyze_range,
    validate_temperature_range,
    get_analysis_summary,
    get_transformation_metrics,
    detect_segment_direction,
)
from pkynetics.result_visualization import plot_dilatometry_analysis

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "pkynetics", "data")


def get_analysis_range(
    temperature: np.ndarray, strain: np.ndarray, auto_detect: bool = False
) -> Tuple[float, float]:
    """
    Get temperature range for analysis from user input or auto-detect.

    Args:
        temperature: Full temperature array to show available range
        strain: Strain data array
        auto_detect: If True, automatically determine analysis range

    Returns:
        Tuple of start and end temperatures for analysis
    """
    temp_min, temp_max = min(temperature), max(temperature)
    is_cooling = detect_segment_direction(temperature, strain)
    segment_type = "Cooling" if is_cooling else "Heating"

    print(f"\nAvailable temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
    print(f"Segment type: {segment_type}")

    if auto_detect:
        # Auto-detect a reasonable range (~30% from each end)
        margin = (temp_max - temp_min) * 0.3
        if is_cooling:
            start_temp = temp_max - margin
            end_temp = temp_min + margin
        else:
            start_temp = temp_min + margin
            end_temp = temp_max - margin

        print(f"Auto-detected range: {start_temp:.1f}°C to {end_temp:.1f}°C")
        return start_temp, end_temp

    while True:
        try:
            start_temp = float(input("Enter start temperature for analysis (°C): "))
            end_temp = float(input("Enter end temperature for analysis (°C): "))

            if validate_temperature_range(temperature, start_temp, end_temp):
                return start_temp, end_temp
            else:
                print("Invalid temperature range. Please ensure:")
                if is_cooling:
                    print(
                        "- Start temperature is greater than end temperature (cooling segment)"
                    )
                else:
                    print(
                        "- Start temperature is less than end temperature (heating segment)"
                    )
                print("- Temperatures are within the available range")

        except ValueError:
            print("Please enter valid numbers.")


def dilatometry_analysis_example(filenames=None, auto_detect_range=False):
    """
    Example of importing and analyzing dilatometry data.

    Args:
        filenames: List of filenames to analyze. If None, uses default files.
        auto_detect_range: If True, automatically determine analysis range
    """
    # Default files if none specified
    if filenames is None:
        filenames = ["sample_dilatometry_data.asc", "ejemplo_enfriamiento.asc"]

    # If a single string is passed, convert to list
    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:
        dilatometry_file_path = os.path.join(PKG_DATA_DIR, filename)

        if not os.path.exists(dilatometry_file_path):
            logger.error(f"File not found: {dilatometry_file_path}")
            print(f"Available files in {PKG_DATA_DIR}:")
            for file in os.listdir(PKG_DATA_DIR):
                if file.endswith(".asc"):
                    print(f"  - {file}")
            continue  # Skip to next file if current file not found

        try:
            print(f"\n{'=' * 80}")
            print(f"Analyzing file: {filename}")
            print(f"{'=' * 80}")

            # Import data
            data = dilatometry_importer(dilatometry_file_path)
            logger.info("Dilatometry data imported successfully.")

            temperature = data["temperature"]
            strain = data["relative_change"]

            # Get analysis range (auto or from user)
            start_temp, end_temp = get_analysis_range(
                temperature, strain, auto_detect=auto_detect_range
            )
            temperature_range, strain_range = analyze_range(
                temperature, strain, start_temp, end_temp
            )

            # Process and analyze data
            smooth_strain = smooth_data(strain_range)

            for method in ["lever", "tangent"]:
                logger.info(f"\nAnalyzing using {method} method:")

                # Perform analysis
                results = analyze_dilatometry_curve(
                    temperature_range, smooth_strain, method=method
                )

                # Calculate additional metrics
                metrics = get_transformation_metrics(results)

                # Print results
                print(f"\n{method.capitalize()} Method Results:")
                print(get_analysis_summary(results))
                print("\nAdditional Metrics:")
                print(f"Temperature range: {metrics['temperature_range']:.2f}°C")
                print(
                    f"Normalized mid position: {metrics['normalized_mid_position']:.3f}"
                )
                if "max_transformation_rate" in metrics:
                    print(
                        f"Maximum transformation rate: {metrics['max_transformation_rate']:.3e} /°C"
                    )

                # Plot results
                fig = plot_dilatometry_analysis(
                    temperature_range, strain_range, smooth_strain, results, method
                )
                fig.show()

        except Exception as e:
            logger.error(f"Error in dilatometry analysis: {str(e)}")
            logger.exception("Detailed traceback:")
            print(f"Error analyzing file {filename}: {str(e)}")
            print("Continuing with next file...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dilatometry data.")
    parser.add_argument(
        "--file",
        type=str,
        nargs="+",
        help="Specific data file(s) to analyze (e.g., ejemplo_enfriamiento.asc)",
    )
    parser.add_argument(
        "--auto-range",
        action="store_true",
        help="Automatically determine analysis range without user input",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all example files",
    )

    args = parser.parse_args()

    if args.all:
        # Process all example files
        dilatometry_analysis_example(auto_detect_range=args.auto_range)
    elif args.file:
        # Process specified files
        dilatometry_analysis_example(args.file, auto_detect_range=args.auto_range)
    else:
        # Default behavior: process both example files
        dilatometry_analysis_example(auto_detect_range=args.auto_range)
