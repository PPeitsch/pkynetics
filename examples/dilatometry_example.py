import os
import logging
import matplotlib.pyplot as plt
import numpy as np

from data_import import dilatometry_importer
from data_preprocessing import smooth_data, analyze_dilatometry_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def dilatometry_analysis_example():
    """Example of importing and analyzing dilatometry data with enhanced visualizations."""
    dilatometry_file_path = os.path.join(PROJECT_ROOT, 'data', 'sample_dilatometry_data.asc')

    try:
        data = dilatometry_importer(dilatometry_file_path)
        logger.info("Dilatometry data imported successfully.")

        temperature = data['temperature']
        strain = data['relative_change']

        smooth_strain = smooth_data(strain, window_length=min(51, len(strain) - 2), polyorder=3)

        results = analyze_dilatometry_curve(temperature, smooth_strain)

        print(f"Start temperature: {results['start_temperature']:.2f}°C")
        print(f"End temperature: {results['end_temperature']:.2f}°C")
        print(f"Mid temperature: {results['mid_temperature']:.2f}°C")

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

        # Plot 1: Raw and smoothed strain data
        ax1.plot(temperature, strain, label='Raw data', alpha=0.5)
        ax1.plot(temperature, smooth_strain, label='Smoothed data', color='r')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Relative Change')
        ax1.set_title('Raw and Smoothed Dilatometry Data')
        ax1.legend()

        # Plot 2: Strain data with transformation points and extrapolation lines
        ax2.plot(temperature, smooth_strain, label='Strain')
        ax2.plot(temperature, results['before_extrapolation'], '--', label='Before extrapolation')
        ax2.plot(temperature, results['after_extrapolation'], '--', label='After extrapolation')
        ax2.axvline(results['start_temperature'], color='g', linestyle='--', label='Start')
        ax2.axvline(results['end_temperature'], color='r', linestyle='--', label='End')
        ax2.axvline(results['mid_temperature'], color='b', linestyle='--', label='Mid')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Relative Change')
        ax2.set_title('Dilatometry Curve with Transformation Points and Extrapolations')
        ax2.legend()

        # Plot 3: Lever rule representation
        ax3.plot(temperature, smooth_strain, label='Strain')
        ax3.plot(temperature, results['before_extrapolation'], '--', label='Before extrapolation')
        ax3.plot(temperature, results['after_extrapolation'], '--', label='After extrapolation')
        mid_temp = results['mid_temperature']
        mid_strain = np.interp(mid_temp, temperature, smooth_strain)
        mid_before = np.interp(mid_temp, temperature, results['before_extrapolation'])
        mid_after = np.interp(mid_temp, temperature, results['after_extrapolation'])
        ax3.plot([mid_temp, mid_temp], [mid_before, mid_after], 'k-', label='Lever')
        ax3.plot(mid_temp, mid_strain, 'ro', label='Mid point')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Relative Change')
        ax3.set_title('Lever Rule Representation')
        ax3.legend()

        # Plot 4: Transformed fraction
        ax4.plot(temperature, results['transformed_fraction'], label='Transformed Fraction')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Transformed Fraction')
        ax4.set_title('Transformed Fraction vs Temperature')
        ax4.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error in dilatometry analysis: {str(e)}")


if __name__ == "__main__":
    dilatometry_analysis_example()
