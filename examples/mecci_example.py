import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods.kissinger import kissinger_method
from data_import.dsc_importer import dsc_importer
from result_visualization.kinetic_plot import plot_kissinger
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


def manual_peak_selection(temperature, heat_flow, filename, temp_range):
    """Allow manual selection of peak temperature."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(temperature, heat_flow)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Heat Flow')
    ax.set_title(f'DSC Curve - {filename}\nClick to select peak, close window to skip')
    if temp_range:
        ax.axvspan(temp_range[0], temp_range[1], color='yellow', alpha=0.3, label='Selected Range')
    ax.legend()

    peak_temp = [None]
    selection_made = [False]

    def onclick(event):
        if event.xdata is not None:
            peak_temp[0] = event.xdata
            selection_made[0] = True
            ax.axvline(x=event.xdata, color='r', linestyle='--', label='Selected Peak')
            ax.legend()
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    while not selection_made[0]:
        plt.pause(0.1)
        if not plt.fignum_exists(fig.number):
            break

    plt.close(fig)

    if peak_temp[0] is not None:
        logger.info(f"Manually selected peak temperature: {peak_temp[0]:.2f} K")
    else:
        logger.warning("No peak temperature selected")

    return peak_temp[0]


def plot_full_dsc_curve(temperature, heat_flow, filename):
    """Plot full DSC curve for initial visualization."""
    plt.figure(figsize=(12, 6))
    plt.plot(temperature, heat_flow)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Flow')
    plt.title(f'Full DSC Curve - {filename}')
    plt.grid(True)
    plt.show()


def process_dsc_file(file_path, temp_range=None):
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
    peak_temp = manual_peak_selection(temp_heating, hf_heating, filename, temp_range)

    return peak_temp, heating_rate, temp_heating, hf_heating


def perform_kissinger_analysis(folder_path, temp_range):
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
                peak_temp, heating_rate, temp_heating, hf_heating = process_dsc_file(file_path, temp_range)
                if peak_temp:
                    peak_temperatures.append(peak_temp)
                    heating_rates.append(heating_rate)
                    logger.info(
                        f"Processed {filename}: Peak Temperature = {peak_temp:.2f} K, Heating Rate = {heating_rate:.2f} K/min")
                else:
                    logger.warning(f"No peak selected in {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")

    if peak_temperatures and heating_rates:
        try:
            e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(np.array(peak_temperatures), np.array(heating_rates))
            logger.info(f"Kissinger analysis results:")
            logger.info(f"Activation Energy: {e_a / 1000:.2f} ± {se_e_a / 1000:.2f} kJ/mol")
            logger.info(f"Pre-exponential factor: {a:.2e} min^-1")
            logger.info(f"R-squared: {r_squared:.4f}")
            plot_kissinger(np.array(peak_temperatures), np.array(heating_rates), e_a, a, r_squared)
        except Exception as e:
            logger.error(f"Error in Kissinger analysis: {str(e)}")
    else:
        logger.warning("Not enough data for Kissinger analysis")


def main():
    # folder_path = input("Enter the path to the folder containing DSC files: ")
    # folder_path = "C:/Users/Pablo/Desktop/Mercedes Duran/procesado/kissinger/MS"
    # folder_path = "C:/Users/Pablo/Desktop/Mercedes Duran/procesado/kissinger/ZAC"
    folder_path = "C:/Users/Pablo/Desktop/Mercedes Duran/procesado/kissinger/ZAC2"

    # Plot a sample curve for user to select temperature range
    sample_file = next(f for f in os.listdir(folder_path) if f.endswith('.txt'))
    sample_path = os.path.join(folder_path, sample_file)
    _, _, temp, hf = process_dsc_file(sample_path)
    plot_full_dsc_curve(temp, hf, sample_file)

    # User selects temperature range
    temp_min = float(input("Enter minimum temperature for analysis (K): "))
    temp_max = float(input("Enter maximum temperature for analysis (K): "))
    temp_range = (temp_min, temp_max)

    # Perform Kissinger analysis
    perform_kissinger_analysis(folder_path, temp_range)


if __name__ == "__main__":
    main()