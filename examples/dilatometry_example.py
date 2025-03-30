"""Example of importing and analyzing dilatometry data with enhanced visualization and validation."""

import argparse
import logging
import os
import warnings  # Import standard warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Configure warnings to be displayed
warnings.simplefilter("always", UserWarning)


from pkynetics.data_import import dilatometry_importer
from pkynetics.data_preprocessing import smooth_data
from pkynetics.result_visualization import plot_dilatometry_analysis
from pkynetics.technique_analysis import analyze_dilatometry_curve
from pkynetics.technique_analysis.utilities import (
    analyze_range,
    detect_segment_direction,
    get_analysis_summary,
    get_transformation_metrics,
    validate_temperature_range,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Assuming the script is in 'examples' folder, navigate up and then to src/pkynetics/data
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..")
)  # Adjust if script location changes
PKG_DATA_DIR = os.path.join(PROJECT_ROOT, "src", "pkynetics", "data")


def get_analysis_range(
    temperature: np.ndarray, strain: np.ndarray, auto_detect: bool = False
) -> Tuple[float, float]:
    """
    Get temperature range for analysis from user input or auto-detect,
    validating against data range and direction.

    Args:
        temperature: Full temperature array to show available range.
        strain: Strain data array (used for direction detection).
        auto_detect: If True, automatically determine analysis range.

    Returns:
        Tuple of (start_temp, end_temp) for analysis, ordered appropriately for the direction.

    Raises:
        ValueError: If auto-detection fails or user input is invalid after retries.
    """
    temp_min, temp_max = min(temperature), max(temperature)
    is_cooling = detect_segment_direction(temperature, strain)
    segment_type = "Cooling" if is_cooling else "Heating"

    print(f"\nAvailable temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
    print(f"Detected segment type: {segment_type}")

    if auto_detect:
        # Auto-detect a reasonable range (e.g., middle 60% of the temperature span)
        margin_fraction = 0.20  # Exclude 20% from each end
        temp_span = temp_max - temp_min
        margin_value = temp_span * margin_fraction

        if is_cooling:
            # For cooling: start near high temp, end near low temp
            start_temp = temp_max - margin_value
            end_temp = temp_min + margin_value
        else:
            # For heating: start near low temp, end near high temp
            start_temp = temp_min + margin_value
            end_temp = temp_max - margin_value

        # Validate the auto-detected range (should always be valid if derived correctly)
        is_valid, msg = validate_temperature_range(temperature, start_temp, end_temp)
        if not is_valid:
            # This shouldn't happen with the logic above, but as a safeguard:
            logger.error(f"Auto-detected range validation failed: {msg}")
            # Fallback to full range (might not be ideal for analysis but prevents crash)
            start_temp = temp_max if is_cooling else temp_min
            end_temp = temp_min if is_cooling else temp_max
            print(
                f"Warning: Auto-detection failed validation. Using full range: {start_temp:.1f}°C to {end_temp:.1f}°C"
            )
        else:
            print(
                f"Auto-detected analysis range: {start_temp:.1f}°C to {end_temp:.1f}°C"
            )

        return start_temp, end_temp

    # Manual input loop
    while True:
        try:
            input_start_str = input(
                f"Enter start temperature for analysis ({'higher for cooling' if is_cooling else 'lower for heating'}) (°C): "
            )
            input_start = float(input_start_str)
            input_end_str = input(
                f"Enter end temperature for analysis ({'lower for cooling' if is_cooling else 'higher for heating'}) (°C): "
            )
            input_end = float(input_end_str)

            # Validate the entered range using the utility function
            is_valid, msg = validate_temperature_range(
                temperature, input_start, input_end
            )

            if is_valid:
                print(f"Using analysis range: {input_start:.1f}°C to {input_end:.1f}°C")
                # Return the validated start and end temps
                return input_start, input_end
            else:
                # Print the specific validation error message
                print(f"Error: {msg}")
                print("Please try again.")
                continue  # Ask for input again

        except ValueError:
            print("Invalid input. Please enter numeric values for temperature.")
            continue  # Ask for input again
        except EOFError:
            raise ValueError("Analysis range input aborted.")


def plot_with_direction(
    temperature: np.ndarray,
    strain: np.ndarray,
    smooth_strain: Optional[np.ndarray],  # Allow for no smoothing
    results: dict,
    method: str,
    save_path: str = None,
) -> plt.Figure:
    """
    Create a plot using plot_dilatometry_analysis and add direction indicators.

    Args:
        temperature: Temperature data array for the analyzed range.
        strain: Raw strain data array for the analyzed range.
        smooth_strain: Smoothed strain data array (optional).
        results: Dictionary containing analysis results from analyze_dilatometry_curve.
        method: Analysis method name ('lever' or 'tangent').
        save_path: Path to save the figure (optional).

    Returns:
        matplotlib.figure.Figure: Complete figure with all plots.
    """
    # Use the standard plotting function, passing smooth_strain if available
    # The plotting function needs temperature, raw_strain, processed_strain, results, method_name
    processed_strain_for_plot = (
        smooth_strain if smooth_strain is not None else strain
    )  # Use raw if not smoothed

    fig = plot_dilatometry_analysis(
        temperature, strain, processed_strain_for_plot, results, method.capitalize()
    )

    # Add direction indicator to title/annotations
    is_cooling = results.get("is_cooling", False)
    direction = "Cooling" if is_cooling else "Heating"

    # Add direction to the main figure title or first subplot title
    try:
        ax1 = fig.axes[0]
        current_title = ax1.get_title()
        ax1.set_title(f"{current_title} ({direction} Segment)")
    except IndexError:
        fig.suptitle(
            f"Dilatometry Analysis ({method.capitalize()} Method) - {direction} Segment",
            fontsize=14,
        )

    # Add direction arrow annotation (optional, can be cluttered)
    try:
        ax1 = fig.axes[0]  # Target the first subplot (usually Strain vs Temp)
        x_range = ax1.get_xlim()
        y_range = ax1.get_ylim()

        # Position arrow near the bottom-left/right to avoid data overlap
        y_pos = y_range[0] + (y_range[1] - y_range[0]) * 0.10  # 10% from bottom
        arrow_length_ratio = 0.15  # Length relative to x-axis width

        if is_cooling:
            # Arrow points left (high to low temp)
            arrow_start_x = (
                x_range[1] - (x_range[1] - x_range[0]) * 0.05
            )  # Start near right edge
            arrow_end_x = arrow_start_x - (x_range[1] - x_range[0]) * arrow_length_ratio
            text_x = arrow_start_x  # Text near arrow start
            ha = "right"
        else:
            # Arrow points right (low to high temp)
            arrow_start_x = (
                x_range[0] + (x_range[1] - x_range[0]) * 0.05
            )  # Start near left edge
            arrow_end_x = arrow_start_x + (x_range[1] - x_range[0]) * arrow_length_ratio
            text_x = arrow_start_x  # Text near arrow start
            ha = "left"

        ax1.annotate(
            f"{direction} Direction",
            xy=(arrow_end_x, y_pos),  # Arrow points to this coordinate
            xytext=(text_x, y_pos),  # Text location
            arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5),
            ha=ha,
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        )
    except IndexError:
        pass  # No axes found to annotate

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )  # Adjust layout to prevent title overlap if suptitle used

    # Save if needed
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot to {save_path}: {e}")

    return fig


def dilatometry_analysis_example(
    filenames: Optional[List[str]] = None,
    auto_detect_range: bool = False,
    save_plots: bool = False,
    apply_smoothing: bool = True,
    analysis_method: str = "both",  # 'lever', 'tangent', or 'both'
    output_dir: str = ".",  # Directory to save plots
):
    """
    Example workflow for importing and analyzing dilatometry data.

    Args:
        filenames: List of filenames within PKG_DATA_DIR to analyze. If None, uses default files.
        auto_detect_range: If True, automatically determine analysis range.
        save_plots: If True, save plots to files in output_dir instead of displaying them.
        apply_smoothing: If True, smooth strain data before analysis.
        analysis_method: Which method(s) to run ('lever', 'tangent', 'both').
        output_dir: Directory where plots will be saved if save_plots is True.
    """
    # Default files if none specified
    if filenames is None:
        # Look for common dilatometry files in the data directory
        default_files = [
            f
            for f in os.listdir(PKG_DATA_DIR)
            if f.endswith((".asc", ".txt", ".csv"))
            and "dilatometry" in f.lower()
            or "enfriamiento" in f.lower()
        ]
        if not default_files:
            default_files = [
                "sample_dilatometry_data.asc",
                "ejemplo_enfriamiento.asc",
            ]  # Fallback defaults
        filenames = default_files
        print(f"No specific files provided. Using default/found files: {filenames}")

    # If a single string is passed, convert to list
    if isinstance(filenames, str):
        filenames = [filenames]

    # Ensure output directory exists
    if save_plots and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(
                f"Could not create output directory {output_dir}: {e}. Plots will not be saved."
            )
            save_plots = False  # Disable saving

    # Determine methods to run
    methods_to_run = []
    if analysis_method.lower() == "both":
        methods_to_run = ["lever", "tangent"]
    elif analysis_method.lower() in ["lever", "tangent"]:
        methods_to_run = [analysis_method.lower()]
    else:
        logger.error(
            f"Invalid analysis_method: '{analysis_method}'. Choose 'lever', 'tangent', or 'both'."
        )
        return

    # --- Process each file ---
    for filename in filenames:
        dilatometry_file_path = os.path.join(PKG_DATA_DIR, filename)

        if not os.path.exists(dilatometry_file_path):
            logger.error(f"File not found: {dilatometry_file_path}")
            print(f"\nAvailable files in {PKG_DATA_DIR}:")
            try:
                for file in os.listdir(PKG_DATA_DIR):
                    print(f"  - {file}")
            except FileNotFoundError:
                print(f"  Error: Could not list files in {PKG_DATA_DIR}")
            continue  # Skip to next file

        try:
            print(f"\n{'='*80}")
            print(f"Analyzing file: {filename}")
            print(f"{'='*80}")

            # 1. Import data
            data = dilatometry_importer(dilatometry_file_path)
            logger.info(f"Dilatometry data imported successfully from {filename}.")

            temperature_full = data["temperature"]
            strain_full = data["relative_change"]

            if len(temperature_full) == 0:
                logger.warning(f"No data found in file: {filename}. Skipping.")
                continue

            # 2. Get analysis range (auto or from user) - handles validation
            start_temp, end_temp = get_analysis_range(
                temperature_full, strain_full, auto_detect=auto_detect_range
            )

            # 3. Extract the data for the selected range - handles validation
            temperature_range, strain_range = analyze_range(
                temperature_full, strain_full, start_temp, end_temp
            )
            logger.info(
                f"Analysis range set from {start_temp:.1f}°C to {end_temp:.1f}°C ({len(temperature_range)} points)."
            )

            # 4. Preprocess: Smooth data if requested
            smooth_strain_range = None
            if apply_smoothing:
                try:
                    smooth_strain_range = smooth_data(strain_range)
                    logger.info("Strain data smoothed using Savitzky-Golay filter.")
                    processed_strain = smooth_strain_range
                except ValueError as e:
                    logger.warning(
                        f"Smoothing failed: {e}. Using raw strain data for analysis."
                    )
                    processed_strain = strain_range
            else:
                logger.info("Skipping smoothing. Using raw strain data for analysis.")
                processed_strain = strain_range  # Use raw strain if not smoothing

            # 5. Analyze data using selected method(s)
            for method in methods_to_run:
                logger.info(f"\n--- Analyzing using {method.capitalize()} method ---")

                try:
                    # Define parameters (can be exposed via argparse later)
                    # Sensible defaults, potentially adjusted based on is_cooling if needed
                    params = {
                        "margin_percent": (
                            None if method == "tangent" else 0.2
                        ),  # Let tangent find optimal if None
                        "find_inflection_margin": 0.3,  # Specific to lever method point finding
                        "min_points_fit": 10,
                        "min_r2_optimal_margin": 0.98,  # Slightly relaxed default
                        "deviation_threshold": None,  # Let tangent calculate
                    }

                    # Perform analysis
                    results = analyze_dilatometry_curve(
                        temperature=temperature_range,
                        strain=processed_strain,  # Use potentially smoothed data
                        method=method,
                        **params,
                    )

                    # Add original range data to results for plotting context if needed
                    results["temperature_full"] = temperature_full
                    results["strain_full"] = strain_full

                    # Calculate additional metrics
                    metrics = get_transformation_metrics(results)

                    # Print results summary
                    print(f"\n{method.capitalize()} Method Results:")
                    print(
                        get_analysis_summary(results)
                    )  # Includes fit quality and warnings for tangent

                    # Print additional metrics
                    print("\nAdditional Metrics:")
                    print(
                        f"  Temperature span (T_end - T_start): {metrics.get('temperature_span', float('nan')):.2f}°C"
                    )
                    print(
                        f"  Normalized mid position ((T_50% - T_start) / span): {metrics.get('normalized_mid_position', float('nan')):.3f}"
                    )
                    if "max_transformation_rate_per_degree" in metrics:
                        print(
                            f"  Max transformation rate: {metrics['max_transformation_rate_per_degree']:.3e} /°C at {metrics['temperature_at_max_rate']:.1f}°C"
                        )

                    # 6. Plot results
                    plot_filename = None
                    if save_plots:
                        base_name = os.path.splitext(filename)[0]
                        direction_label = (
                            "cooling" if results.get("is_cooling") else "heating"
                        )
                        plot_filename = os.path.join(
                            output_dir,
                            f"{base_name}_{method}_{direction_label}_analysis.png",
                        )

                    # Generate plot (passing raw strain_range for comparison, and processed_strain)
                    fig = plot_with_direction(
                        temperature=temperature_range,
                        strain=strain_range,  # Pass original (unsmoothed) range data
                        smooth_strain=smooth_strain_range,  # Pass smoothed data (or None)
                        results=results,
                        method=method,
                        save_path=plot_filename,
                    )

                    if not save_plots:
                        # fig.show() # Might block execution, use plt.show() after loop if preferred
                        plt.show(block=False)  # Show non-blocking

                except (ValueError, np.linalg.LinAlgError) as analysis_error:
                    logger.error(
                        f"Analysis using {method} method failed: {analysis_error}"
                    )
                    print(f"\nError during {method} analysis: {analysis_error}")
                    print(
                        "Check data quality, analysis range, and parameters (e.g., min_points_fit)."
                    )
                except Exception as unexpected_error:
                    logger.exception(
                        f"An unexpected error occurred during {method} analysis:"
                    )
                    print(
                        f"\nAn unexpected error occurred during {method} analysis: {unexpected_error}"
                    )

        except (ValueError, FileNotFoundError, ImportError) as import_or_setup_error:
            logger.error(f"Error processing file {filename}: {import_or_setup_error}")
            print(f"\nError processing file {filename}: {import_or_setup_error}")
        except Exception as e:
            # Catch-all for unexpected errors during the file processing loop
            logger.exception(
                f"An unexpected error occurred while processing {filename}:"
            )
            print(f"\nAn unexpected error occurred while processing {filename}: {e}")

        # Optional: pause between files if plots are shown non-blocking
        if not save_plots and len(filenames) > 1:
            input("Press Enter to continue to the next file...")
            plt.close("all")  # Close previous plots

    if not save_plots:
        print("\nDisplaying all generated plots. Close plot windows to exit.")
        plt.show()  # Show all non-blocking plots and block until closed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze dilatometry data using Pkynetics."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        nargs="+",  # Accept one or more filenames
        help="Specific data file(s) located in the pkynetics/data directory to analyze (e.g., sample_dilatometry_data.asc). If omitted, tries to find default files.",
    )
    parser.add_argument(
        "--auto-range",
        action="store_true",
        help="Automatically determine the temperature range for analysis.",
    )
    # parser.add_argument( # Removed --all, default behavior is now to process found files if --file omitted
    #     "--all",
    #     action="store_true",
    #     help="Process all example files found in the data directory.",
    # )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save analysis plots to files instead of displaying them.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_false",
        dest="apply_smoothing",  # Store False if specified, default is True
        help="Do not smooth the strain data before analysis.",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="both",
        choices=["lever", "tangent", "both"],
        help="Analysis method to use: 'lever', 'tangent', or 'both'. Default is 'both'.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="dilatometry_analysis_results",
        help="Directory to save output plots if --save is used. Default is 'dilatometry_analysis_results'.",
    )

    args = parser.parse_args()

    # Call the main analysis function
    dilatometry_analysis_example(
        filenames=args.file,  # Pass None if not provided, function handles default
        auto_detect_range=args.auto_range,
        save_plots=args.save,
        apply_smoothing=args.apply_smoothing,
        analysis_method=args.method,
        output_dir=args.output_dir,
    )

    print("\nDilatometry analysis example finished.")
