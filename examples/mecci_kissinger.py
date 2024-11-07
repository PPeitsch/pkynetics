import os
import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods.kissinger import kissinger_method
from data_import.dsc_importer import dsc_importer
from result_visualization.kinetic_plot import plot_kissinger
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from findpeaks import findpeaks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def separate_heating_cooling(temperature, time, heat_flow):
    """Separate heating and cooling portions of the DSC curve."""
    peak_index = np.argmax(temperature)
    return (
        temperature[:peak_index], time[:peak_index], heat_flow[:peak_index],
        temperature[peak_index:], time[peak_index:], heat_flow[peak_index:]
    )


def remove_isotherms(temperature, time, heat_flow, threshold=0.1):
    """Remove isothermal segments from the DSC data."""
    temp_rate = np.gradient(temperature, time)
    non_isothermal = np.abs(temp_rate) > threshold
    return temperature[non_isothermal], time[non_isothermal], heat_flow[non_isothermal]


def find_peak_temperature(temperature, heat_flow, temp_range, filename, exothermic=True):
    """Find the peak temperature."""
    logger.info(f"Starting peak detection for {filename}")
    mask = (temperature >= temp_range[0]) & (temperature <= temp_range[1])
    temp_segment = temperature[mask]
    hf_segment = heat_flow[mask]

    if len(temp_segment) < 20:
        logger.warning("Not enough data points in the selected range")
        return None

    peak_indices = plot_dsc_with_derivatives(temp_segment, hf_segment, filename, temp_range, exothermic=exothermic)

    if not peak_indices:
        logger.warning("No peaks found")
        return None

    # Select the peak with the highest heat flow
    peak_index = peak_indices[np.argmax(hf_segment[peak_indices])]

    logger.info(f"Peak detected at temperature {temp_segment[peak_index]:.2f} K")
    return temp_segment[peak_index]


def plot_dsc_curve(temperature, heat_flow, filename, temp_range=None, peak_temp=None):
    """Plot DSC curve for visualization."""
    plt.figure(figsize=(12, 6))
    plt.plot(temperature, heat_flow)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Flow')
    plt.title(f'DSC Curve - {filename}')
    plt.grid(True)

    if temp_range:
        plt.axvspan(temp_range[0], temp_range[1], color='yellow', alpha=0.3, label='Selected Range')

    if peak_temp:
        peak_index = np.abs(temperature - peak_temp).argmin()
        plt.plot(peak_temp, heat_flow[peak_index], 'ro', markersize=10, label='Detected Peak')

    plt.legend()
    plt.show()


def plot_full_dsc_curve(temperature, heat_flow, filename):
    """Plot full DSC curve for initial visualization."""
    plt.figure(figsize=(12, 6))
    plt.plot(temperature, heat_flow)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Flow')
    plt.title(f'Full DSC Curve - {filename}')
    plt.grid(True)
    plt.show()


def plot_dsc_with_derivatives(temperature, heat_flow, filename, temp_range=None, peak_temp=None, exothermic=True):
    """Plot DSC curve with its first derivative for visualization."""
    logger.info(f"Plotting DSC curve for {filename}")
    smoothed_hf = gaussian_filter1d(heat_flow, sigma=3)
    gradient = np.gradient(smoothed_hf, temperature)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    ax1.plot(temperature, heat_flow, label='Original')
    ax1.plot(temperature, smoothed_hf, label='Smoothed')
    ax1.set_ylabel('Heat Flow')
    ax1.set_title(f'DSC Curve - {filename}')

    ax2.plot(temperature, gradient)
    ax2.set_ylabel('Gradient')
    ax2.set_title('Gradient of Heat Flow')
    ax2.set_xlabel('Temperature (K)')

    for ax in (ax1, ax2):
        ax.grid(True)
        if temp_range:
            ax.axvspan(temp_range[0], temp_range[1], color='yellow', alpha=0.3, label='Selected Range')
        if peak_temp:
            ax.axvline(x=peak_temp, color='r', linestyle='--', label='Detected Peak')
            peak_index = np.abs(temperature - peak_temp).argmin()
            ax.plot(peak_temp, smoothed_hf[peak_index], 'ro', markersize=10)
        ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # Now that the figure is closed, we can use findpeaks
    fp = findpeaks(lookahead=5, interpolate=10, method='topology')
    try:
        if exothermic:
            results = fp.fit(smoothed_hf)
        else:
            results = fp.fit(-smoothed_hf)

        peak_indices = results['df']['peak'].values
        logger.info(f"Detected {len(peak_indices)} peaks in {filename}")
        return peak_indices
    except Exception as e:
        logger.error(f"Error in findpeaks for {filename}: {str(e)}")
        return []


def process_dsc_file(file_path, temp_range=None, exothermic=True):
    """Process a single DSC file and return peak temperature and heating rate."""
    logger.info(f"Processing file: {file_path}")
    data = dsc_importer(file_path=file_path, manufacturer='Setaram')

    # Separate heating portion
    temp_heating, time_heating, hf_heating, _, _, _ = separate_heating_cooling(
        data['temperature'], data['time'], data['heat_flow']
    )

    # Remove isothermal segments
    temp_heating, time_heating, hf_heating = remove_isotherms(temp_heating, time_heating, hf_heating)

    if temp_range:
        mask = (temp_heating >= temp_range[0]) & (temp_heating <= temp_range[1])
        temp_heating = temp_heating[mask]
        time_heating = time_heating[mask]
        hf_heating = hf_heating[mask]

        heating_rate = (temp_heating[-1] - temp_heating[0]) / (time_heating[-1] - time_heating[0]) * 60

        filename = os.path.basename(file_path)
        peak_temp = find_peak_temperature(temp_heating, hf_heating, temp_range, filename, exothermic)

        return peak_temp, heating_rate, temp_heating, time_heating, hf_heating
    else:
        return temp_heating, time_heating, hf_heating


def perform_kissinger_analysis(folder_path, temp_range, exothermic):
    """Perform Kissinger analysis on multiple DSC files in a folder."""
    peak_temperatures = []
    heating_rates = []
    total_files = len([f for f in os.listdir(folder_path) if f.endswith(".txt")])
    processed_files = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            processed_files += 1
            logger.info(f"Processing file {processed_files} of {total_files}: {filename}")
            file_path = os.path.join(folder_path, filename)
            try:
                peak_temp, heating_rate, temp_heating, _, hf_heating = process_dsc_file(file_path, temp_range, exothermic)
                if peak_temp:
                    peak_temperatures.append(peak_temp)
                    heating_rates.append(heating_rate)
                    logger.info(f"Processed {filename}: Peak Temperature = {peak_temp:.2f} K, Heating Rate = {heating_rate:.2f} K/min")
                    plot_dsc_with_derivatives(temp_heating, hf_heating, filename, temp_range, peak_temp, exothermic)
                else:
                    logger.warning(f"No peak found in {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")

    if peak_temperatures and heating_rates:
        try:
            e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(np.array(peak_temperatures), np.array(heating_rates))
            logger.info(f"Kissinger analysis results:")
            logger.info(f"Activation Energy: {e_a/1000:.2f} ± {se_e_a/1000:.2f} kJ/mol")
            logger.info(f"Pre-exponential factor: {a:.2e} min^-1")
            logger.info(f"R-squared: {r_squared:.4f}")
            plot_kissinger(np.array(peak_temperatures), np.array(heating_rates), e_a, a, r_squared)
        except Exception as e:
            logger.error(f"Error in Kissinger analysis: {str(e)}")
    else:
        logger.warning("Not enough data for Kissinger analysis")


def main():
    folder_path = input("Enter the path to the folder containing DSC files: ")

    # Plot a sample curve for user to select temperature range
    sample_file = next(f for f in os.listdir(folder_path) if f.endswith('.txt'))
    sample_path = os.path.join(folder_path, sample_file)
    temp, time, hf = process_dsc_file(sample_path)
    plot_full_dsc_curve(temp, hf, sample_file)

    # User selects temperature range
    temp_min = float(input("Enter minimum temperature for analysis (K): "))
    temp_max = float(input("Enter maximum temperature for analysis (K): "))
    temp_range = (temp_min, temp_max)

    # User specifies peak type
    peak_type = input("Is the peak exothermic or endothermic? (Enter 'exo' or 'endo'): ").lower()
    exothermic = peak_type == 'exo'

    # Plot the curve again with the selected range highlighted
    plot_dsc_with_derivatives(temp, hf, sample_file, temp_range, exothermic=exothermic)

    # Allow user to adjust peak detection parameters
    adjust = input("Do you want to adjust peak detection parameters? (y/n): ").lower()
    if adjust == 'y':
        prominence = float(input("Enter minimum peak prominence (default is 0.1): ") or 0.1)
        width = int(input("Enter minimum peak width (default is 5): ") or 5)

    # Perform Kissinger analysis
    perform_kissinger_analysis(folder_path, temp_range, exothermic)


if __name__ == "__main__":
    main()
