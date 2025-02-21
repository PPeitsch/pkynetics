"""Example of importing and analyzing dilatometry data with enhanced visualization."""

import os
import logging
from typing import Tuple
import numpy as np

from pkynetics.data_import import dilatometry_importer
from pkynetics.data_preprocessing import smooth_data
from pkynetics.technique_analysis import analyze_dilatometry_curve
from pkynetics.technique_analysis.utilities import (
    analyze_range,
    validate_temperature_range,
    get_analysis_summary,
    get_transformation_metrics, detect_segment_direction
)
from pkynetics.result_visualization import plot_dilatometry_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PKG_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'pkynetics', 'data')


def get_analysis_range(temperature: np.ndarray, strain: np.ndarray) -> Tuple[float, float]:
    """
    Get temperature range for analysis from user input.

    Args:
        temperature: Full temperature array to show available range
        strain: Strain data array

    Returns:
        Tuple of start and end temperatures for analysis
    """
    is_cooling = detect_segment_direction(temperature, strain)
    temp_min, temp_max = min(temperature), max(temperature)

    print(f"\nAvailable temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
    print(f"Segment type: {'Cooling' if is_cooling else 'Heating'}")

    while True:
        try:
            start_temp = float(input("Enter start temperature for analysis (°C): "))
            end_temp = float(input("Enter end temperature for analysis (°C): "))

            if validate_temperature_range(temperature, start_temp, end_temp):
                return start_temp, end_temp
            else:
                print("Invalid temperature range. Please ensure:")
                if is_cooling:
                    print("- Start temperature is greater than end temperature (cooling segment)")
                else:
                    print("- Start temperature is less than end temperature (heating segment)")
                print("- Temperatures are within the available range")

        except ValueError:
            print("Please enter valid numbers.")


def dilatometry_analysis_example():
    """Example of importing and analyzing dilatometry data."""
    dilatometry_file_path = os.path.join(PKG_DATA_DIR, 'ejemplo_enfriamiento.asc')

    try:
        # Import data
        data = dilatometry_importer(dilatometry_file_path)
        logger.info("Dilatometry data imported successfully.")

        temperature = data['temperature']
        strain = data['relative_change']

        # Get analysis range from user
        start_temp, end_temp = get_analysis_range(temperature, strain)
        temperature_range, strain_range = analyze_range(temperature, strain, start_temp, end_temp)

        # Process and analyze data
        smooth_strain = smooth_data(strain_range)

        for method in ['lever', 'tangent']:
            logger.info(f"\nAnalyzing using {method} method:")

            # Perform analysis
            results = analyze_dilatometry_curve(temperature_range, smooth_strain, method=method)

            # Calculate additional metrics
            metrics = get_transformation_metrics(results)

            # Print results
            print(f"\n{method.capitalize()} Method Results:")
            print(get_analysis_summary(results))
            print("\nAdditional Metrics:")
            print(f"Temperature range: {metrics['temperature_range']:.2f}°C")
            print(f"Normalized mid position: {metrics['normalized_mid_position']:.3f}")
            if 'max_transformation_rate' in metrics:
                print(f"Maximum transformation rate: {metrics['max_transformation_rate']:.3e} /°C")

            # Plot results
            fig = plot_dilatometry_analysis(temperature_range,
                                            strain_range,
                                            smooth_strain,
                                            results,
                                            method)
            fig.show()

    except Exception as e:
        logger.error(f"Error in dilatometry analysis: {str(e)}")
        raise


if __name__ == "__main__":
    dilatometry_analysis_example()
