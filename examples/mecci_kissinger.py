import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from data_import.dsc_importer import dsc_importer
from model_fitting_methods.kissinger import kissinger_method
from result_visualization.kinetic_plot import plot_kissinger


def separate_heating_cooling(temperature, time, heat_flow):
    """Separate heating and cooling portions of the DSC curve."""
    peak_index = np.argmax(temperature)
    return (
        temperature[:peak_index], time[:peak_index], heat_flow[:peak_index],
        temperature[peak_index:], time[peak_index:], heat_flow[peak_index:]
    )


def remove_isotherms(temperature, time, heat_flow, threshold=0.1):
    """
    Remove isothermal segments from the DSC data.

    Args:
    temperature (np.array): Temperature data
    time (np.array): Time data
    heat_flow (np.array): Heat flow data
    threshold (float): Minimum temperature change rate to consider non-isothermal (K/min)

    Returns:
    tuple: Cleaned temperature, time, and heat flow arrays
    """
    temp_rate = np.gradient(temperature, time)
    non_isothermal = np.abs(temp_rate) > threshold

    return temperature[non_isothermal], time[non_isothermal], heat_flow[non_isothermal]


def find_peak_temperature(temperature, heat_flow, filename):
    """Find the peak temperature based on the maximum of the heat flow."""
    # Smooth the heat flow data
    smoothed_hf = np.convolve(heat_flow, np.ones(5) / 5, mode='same')
    plot_dsc_curve(temperature=temperature, heat_flow=heat_flow, filename=filename+'smoothed')
    # Find peaks
    peaks, _ = find_peaks(smoothed_hf, height=0, distance=len(smoothed_hf) // 10)

    if len(peaks) == 0:
        return None

    # Return the temperature of the highest peak
    return temperature[peaks[np.argmax(smoothed_hf[peaks])]]


def plot_dsc_curve(temperature, heat_flow, filename, peak_temp=None):
    """Plot and save DSC curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(temperature, heat_flow)
    if peak_temp:
        plt.axvline(x=peak_temp, color='r', linestyle='--', label=f'Peak: {peak_temp:.2f} K')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Flow')
    plt.title(f'DSC Curve for {filename}')
    plt.legend()
    plt.savefig(f'dsc_curve_{filename}.png')
    plt.close()


def perform_kissinger_analysis(folder_path: str):
    peak_temperatures = []
    heating_rates = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            try:
                data = dsc_importer(file_path=file_path, manufacturer='Setaram')

                # Separate heating and cooling
                temp_heating, time_heating, hf_heating, _, _, _ = separate_heating_cooling(
                    data['temperature'], data['time'], data['heat_flow']
                )

                # Remove isothermal segments
                temp, time, hf = remove_isotherms(temp_heating, time_heating, hf_heating)

                # Ask user to select temperature range for analysis
                temp_min = float(input("Enter minimum temperature for analysis: "))
                temp_max = float(input("Enter maximum temperature for analysis: "))
                mask = (temp_heating >= temp_min) & (temp_heating <= temp_max)

                temp_analysis = temp_heating[mask]
                time_analysis = time_heating[mask]
                hf_analysis = hf_heating[mask]

                # Find peak temperature
                peak_temp = find_peak_temperature(temp_analysis, hf_analysis, filename)

                if peak_temp is not None:
                    # Calculate heating rate (use only the heating part)
                    heating_mask = np.gradient(temp) > 0
                    temp_heating = temp_analysis[heating_mask]
                    time_heating = time_analysis[heating_mask]
                    heating_rate = (temp_heating[-1] - temp_heating[0]) / (
                                time_heating[-1] - time_heating[0]) * 60  # Convert to K/min

                    peak_temperatures.append(peak_temp)
                    heating_rates.append(heating_rate)

                    print(
                        f"Processed {filename}: Peak Temperature = {peak_temp:.2f} K, Heating Rate = {heating_rate:.2f} K/min")

                    # Plot DSC curve
                    plot_dsc_curve(temp_analysis, hf_analysis, filename, peak_temp)
                else:
                    print(f"No peak found in {filename}")
                    # Plot DSC curve
                    plot_dsc_curve(temp_analysis, hf_analysis, filename)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    if len(peak_temperatures) > 1:
        try:
            e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(np.array(peak_temperatures), np.array(heating_rates))

            plot_kissinger(np.array(peak_temperatures), np.array(heating_rates), e_a, a, r_squared)

            print(f"\nKissinger Analysis Results:")
            print(f"Activation Energy (E_a) = {e_a / 1000:.2f} ± {se_e_a / 1000:.2f} kJ/mol")
            if np.isnan(a):
                print("Pre-exponential Factor (A) could not be determined")
            else:
                print(f"Pre-exponential Factor (A) = {a:.2e} min^-1")
            print(f"R-squared = {r_squared:.4f}")

            print("\nData used for Kissinger analysis:")
            print("Peak Temperatures (K):", peak_temperatures)
            print("Heating Rates (K/min):", heating_rates)
            print("1000/T (K^-1):", [1000 / T for T in peak_temperatures])
            print("ln(β/T^2):", [np.log(beta / T ** 2) for beta, T in zip(heating_rates, peak_temperatures)])
        except Exception as e:
            print(f"Error in Kissinger analysis: {str(e)}")
    else:
        print("\nInsufficient data for Kissinger analysis. At least two different heating rates are required.")


# Usage
folder_path = "C:/Users/Pablo/Desktop/Mercedes Duran/procesado/kissinger/MS"
perform_kissinger_analysis(folder_path)
