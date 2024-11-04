"""Example of importing and analyzing dilatometry data with enhanced visualization."""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from data_import import dilatometry_importer
from data_preprocessing import smooth_data
from technique_analysis.dilatometry import analyze_dilatometry_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def plot_raw_and_smoothed(ax: plt.Axes, temperature: np.ndarray, strain: np.ndarray,
                          smooth_strain: np.ndarray, method: str) -> None:
    """Plot raw and smoothed dilatometry data."""
    ax.plot(temperature, strain, label='Raw data', alpha=0.5)
    ax.plot(temperature, smooth_strain, label='Smoothed data', color='r')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Relative Change')
    ax.set_title(f'Raw and Smoothed Dilatometry Data ({method.capitalize()} Method)')
    ax.legend()
    ax.grid(True)


def plot_transformation_points(ax: plt.Axes, temperature: np.ndarray, smooth_strain: np.ndarray,
                               results: Dict) -> None:
    """Plot strain data with transformation points and extrapolations."""
    ax.plot(temperature, smooth_strain, label='Strain')
    ax.plot(temperature, results['before_extrapolation'], '--', label='Before extrapolation')
    ax.plot(temperature, results['after_extrapolation'], '--', label='After extrapolation')

    # Add vertical lines and annotations for transformation points
    points = {
        'Start': ('start_temperature', 'green'),
        'End': ('end_temperature', 'red'),
        'Mid': ('mid_temperature', 'blue')
    }

    y_range = ax.get_ylim()
    text_y_positions = np.linspace(y_range[0], y_range[1], len(points) + 2)[1:-1]

    for (label, (temp_key, color)), y_pos in zip(points.items(), text_y_positions):
        temp = results[temp_key]
        ax.axvline(temp, color=color, linestyle='--', label=label)
        ax.annotate(f'{label}: {temp:.1f}°C',
                    xy=(temp, y_pos),
                    xytext=(10, 0), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                    ha='left', va='center')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Relative Change')
    ax.set_title('Dilatometry Curve with Transformation Points and Extrapolations')
    ax.legend()
    ax.grid(True)


def plot_lever_rule(ax: plt.Axes, temperature: np.ndarray, smooth_strain: np.ndarray,
                    results: Dict) -> None:
    """Plot lever rule representation."""
    ax.plot(temperature, smooth_strain, label='Strain')
    ax.plot(temperature, results['before_extrapolation'], '--', label='Before extrapolation')
    ax.plot(temperature, results['after_extrapolation'], '--', label='After extrapolation')

    mid_temp = results['mid_temperature']
    mid_strain = np.interp(mid_temp, temperature, smooth_strain)
    mid_before = np.interp(mid_temp, temperature, results['before_extrapolation'])
    mid_after = np.interp(mid_temp, temperature, results['after_extrapolation'])

    # Plot lever and add annotation
    ax.plot([mid_temp, mid_temp], [mid_before, mid_after], 'k-', label='Lever')
    ax.plot(mid_temp, mid_strain, 'ro', label='Mid point')
    ax.annotate(f'Mid point: {mid_temp:.1f}°C',
                xy=(mid_temp, mid_strain),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Relative Change')
    ax.set_title('Lever Rule Representation')
    ax.legend()
    ax.grid(True)


def plot_transformed_fraction(ax: plt.Axes, temperature: np.ndarray, results: Dict) -> None:
    """Plot transformed fraction vs temperature."""
    ax.plot(temperature, results['transformed_fraction'], label='Transformed Fraction')

    # Add points and annotations for transformation temperatures
    points = {
        'Start': ('start_temperature', 'green', 0.0),
        'Mid': ('mid_temperature', 'blue', 0.5),
        'End': ('end_temperature', 'red', 1.0)
    }

    for label, (temp_key, color, fraction) in points.items():
        temp = results[temp_key]
        ax.axvline(temp, color=color, linestyle='--', label=f'{label}')
        ax.plot(temp, fraction, 'o', color=color)
        ax.annotate(f'{label}: {temp:.1f}°C\n{fraction * 100:.1f}%',
                    xy=(temp, fraction),
                    xytext=(10, 0), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Transformed Fraction')
    ax.set_title('Transformed Fraction vs Temperature')
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True)


def plot_analysis_results(temperature: np.ndarray, strain: np.ndarray, smooth_strain: np.ndarray,
                          results: Dict, method: str) -> plt.Figure:
    """Create complete visualization of dilatometry analysis."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

    plot_raw_and_smoothed(ax1, temperature, strain, smooth_strain, method)
    plot_transformation_points(ax2, temperature, smooth_strain, results)
    plot_lever_rule(ax3, temperature, smooth_strain, results)
    plot_transformed_fraction(ax4, temperature, results)

    plt.tight_layout()
    return fig


def get_analysis_range(temperature: np.ndarray) -> Tuple[float, float]:
    """
    Get temperature range for analysis from user input.

    Args:
        temperature: Full temperature array to show available range

    Returns:
        Tuple of start and end temperatures for analysis
    """
    print(f"\nAvailable temperature range: {temperature.min():.1f}°C to {temperature.max():.1f}°C")
    while True:
        try:
            start_temp = float(input("Enter start temperature for analysis (°C): "))
            end_temp = float(input("Enter end temperature for analysis (°C): "))

            if start_temp >= end_temp:
                print("Start temperature must be less than end temperature.")
                continue

            if (start_temp < temperature.min() or start_temp > temperature.max() or
                    end_temp < temperature.min() or end_temp > temperature.max()):
                print("Temperatures must be within the available range.")
                continue

            return start_temp, end_temp

        except ValueError:
            print("Please enter valid numbers.")


def analyze_range(temperature: np.ndarray, strain: np.ndarray,
                  start_temp: float, end_temp: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and return data within specified temperature range.

    Args:
        temperature: Full temperature array
        strain: Full strain array
        start_temp: Start temperature for analysis
        end_temp: End temperature for analysis

    Returns:
        Tuple of temperature and strain arrays within specified range
    """
    mask = (temperature >= start_temp) & (temperature <= end_temp)
    return temperature[mask], strain[mask]


def dilatometry_analysis_example():
    """Example of importing and analyzing dilatometry data."""
    dilatometry_file_path = os.path.join(PROJECT_ROOT, 'data', 'sample_dilatometry_data.asc')

    try:
        # Import data
        data = dilatometry_importer(dilatometry_file_path)
        logger.info("Dilatometry data imported successfully.")

        temperature = data['temperature']
        strain = data['relative_change']

        # Get analysis range from user
        start_temp, end_temp = get_analysis_range(temperature)
        temperature_range, strain_range = analyze_range(temperature, strain, start_temp, end_temp)

        # Process and analyze data
        smooth_strain = smooth_data(strain_range)

        for method in ['lever', 'tangent']:
            logger.info(f"\nAnalyzing using {method} method:")
            results = analyze_dilatometry_curve(temperature_range, smooth_strain, method=method)

            print(f"\n{method.capitalize()} Method Results:")
            print(f"Start temperature: {results['start_temperature']:.2f}°C")
            print(f"End temperature: {results['end_temperature']:.2f}°C")
            print(f"Mid temperature: {results['mid_temperature']:.2f}°C")

            # Plot results
            plot_analysis_results(temperature_range,
                                  strain_range,
                                  smooth_strain,
                                  results, method)
            plt.show()

    except Exception as e:
        logger.error(f"Error in dilatometry analysis: {str(e)}")
        raise


if __name__ == "__main__":
    dilatometry_analysis_example()
