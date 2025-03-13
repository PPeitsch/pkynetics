"""Example of importing and analyzing dilatometry data with enhanced visualization."""

import os
import logging
import argparse
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

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
            # For cooling: start at high temperature, end at low temperature
            start_temp = temp_max - margin
            end_temp = temp_min + margin
        else:
            # For heating: start at low temperature, end at high temperature
            start_temp = temp_min + margin
            end_temp = temp_max - margin

        print(f"Auto-detected range: {start_temp:.1f}°C to {end_temp:.1f}°C")
        return start_temp, end_temp

    while True:
        try:
            start_temp = float(input("Enter start temperature for analysis (°C): "))
            end_temp = float(input("Enter end temperature for analysis (°C): "))

            # Check if valid based on the segment direction
            if is_cooling:
                # For cooling, start temp should be higher than end temp
                if start_temp <= end_temp:
                    print(
                        "For cooling segments, start temperature must be higher than end temperature"
                    )
                    continue
            else:
                # For heating, start temp should be lower than end temp
                if start_temp >= end_temp:
                    print(
                        "For heating segments, start temperature must be lower than end temperature"
                    )
                    continue

            # Check if within range
            if (
                start_temp < temp_min
                or start_temp > temp_max
                or end_temp < temp_min
                or end_temp > temp_max
            ):
                print("Temperatures must be within the available range")
                continue

            return start_temp, end_temp

        except ValueError:
            print("Please enter valid numbers.")


def plot_with_direction(
    temperature: np.ndarray,
    strain: np.ndarray,
    smooth_strain: np.ndarray,
    results: dict,
    method: str,
    save_path: str = None,
) -> plt.Figure:
    """
    Create a custom plot that indicates the direction (heating/cooling).

    Args:
        temperature: Temperature data array
        strain: Raw strain data array
        smooth_strain: Smoothed strain data array
        results: Dictionary containing analysis results
        method: Analysis method name
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib.figure.Figure: Complete figure with all plots
    """
    # Use the standard plotting function
    fig = plot_dilatometry_analysis(temperature, strain, smooth_strain, results, method)

    # Add direction indicator to title
    is_cooling = results.get("is_cooling", False)
    direction = "Cooling" if is_cooling else "Heating"

    # Add direction to each subplot title
    for i, ax in enumerate(fig.axes):
        title = ax.get_title()
        if i == 0:  # First subplot
            ax.set_title(f"{title} - {direction} Segment")
        else:
            ax.set_title(f"{title} - {direction} Segment")

    # Add direction arrow to the first subplot - REPOSITIONED TO AVOID CURVE
    ax1 = fig.axes[0]
    x_range = ax1.get_xlim()
    y_range = ax1.get_ylim()

    # Position arrow in top-left corner for better visibility
    y_pos = y_range[0] + (y_range[1] - y_range[0]) * 0.1  # Near bottom
    arrow_length = (x_range[1] - x_range[0]) * 0.15

    if is_cooling:
        # Arrow from high to low temperature
        arrow_start = x_range[0] + (x_range[1] - x_range[0]) * 0.25
        arrow_end = arrow_start - arrow_length
        ax1.annotate(
            "Cooling direction",
            xy=(arrow_end, y_pos),
            xytext=(arrow_start, y_pos),
            arrowprops=dict(facecolor="black", width=1.5, headwidth=8),
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
        )
    else:
        # Arrow from low to high temperature
        arrow_start = x_range[0] + (x_range[1] - x_range[0]) * 0.25
        arrow_end = arrow_start + arrow_length
        ax1.annotate(
            "Heating direction",
            xy=(arrow_end, y_pos),
            xytext=(arrow_start, y_pos),
            arrowprops=dict(facecolor="black", width=1.5, headwidth=8),
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
        )

    plt.tight_layout()

    # Save if needed
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def dilatometry_analysis_example(
    filenames=None, auto_detect_range=False, save_plots=False
):
    """
    Example of importing and analyzing dilatometry data.

    Args:
        filenames: List of filenames to analyze. If None, uses default files.
        auto_detect_range: If True, automatically determine analysis range
        save_plots: If True, save plots to files instead of displaying them
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
            print(f"\n{'='*80}")
            print(f"Analyzing file: {filename}")
            print(f"{'='*80}")

            # Import data
            data = dilatometry_importer(dilatometry_file_path)
            logger.info("Dilatometry data imported successfully.")

            temperature = data["temperature"]
            strain = data["relative_change"]

            # Detect if cooling or heating
            is_cooling = detect_segment_direction(temperature, strain)
            direction = "Cooling" if is_cooling else "Heating"
            print(f"Detected segment direction: {direction}")

            # Get analysis range (auto or from user)
            start_temp, end_temp = get_analysis_range(
                temperature, strain, auto_detect=auto_detect_range
            )

            # Make sure the range is appropriate for the direction
            if is_cooling and start_temp < end_temp:
                print(
                    "Warning: For cooling segments, start temperature should be higher than end temperature."
                )
                print("Swapping temperatures for proper analysis.")
                start_temp, end_temp = end_temp, start_temp
            elif not is_cooling and start_temp > end_temp:
                print(
                    "Warning: For heating segments, start temperature should be lower than end temperature."
                )
                print("Swapping temperatures for proper analysis.")
                start_temp, end_temp = end_temp, start_temp

            temperature_range, strain_range = analyze_range(
                temperature, strain, start_temp, end_temp
            )

            # Process and analyze data
            smooth_strain = smooth_data(strain_range)

            for method in ["lever", "tangent"]:
                logger.info(f"\nAnalyzing using {method} method:")

                # Use a smaller margin for finding inflection points in cooling segments
                # This helps identify the transformation points more accurately
                find_inflection_margin = 0.2 if is_cooling else 0.3

                # Perform analysis with appropriate margin
                results = analyze_dilatometry_curve(
                    temperature_range,
                    smooth_strain,
                    method=method,
                    margin_percent=0.2,
                    find_inflection_margin=find_inflection_margin,
                )

                # Calculate additional metrics
                metrics = get_transformation_metrics(results)

                # Print results
                print(f"\n{method.capitalize()} Method Results ({direction} segment):")
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

                # Plot results with direction indicator
                if save_plots:
                    save_path = (
                        f"{filename.split('.')[0]}_{method}_{direction.lower()}.png"
                    )
                    fig = plot_with_direction(
                        temperature_range,
                        strain_range,
                        smooth_strain,
                        results,
                        method,
                        save_path,
                    )
                else:
                    fig = plot_with_direction(
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
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots to files instead of displaying them",
    )

    args = parser.parse_args()

    if args.all:
        # Process all example files
        dilatometry_analysis_example(
            auto_detect_range=args.auto_range, save_plots=args.save
        )
    elif args.file:
        # Process specified files
        dilatometry_analysis_example(
            args.file, auto_detect_range=args.auto_range, save_plots=args.save
        )
    else:
        # Default behavior: process both example files
        dilatometry_analysis_example(
            auto_detect_range=args.auto_range, save_plots=args.save
        )
