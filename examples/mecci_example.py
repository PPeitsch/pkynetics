import os
import numpy as np
import matplotlib.pyplot as plt
from model_fitting_methods import jmak_method, jmak_equation, fit_modified_jmak, modified_jmak_equation
from model_fitting_methods.kissinger import kissinger_method
from data_import.dsc_importer import dsc_importer
from data_preprocessing import calculate_transformed_fraction
from result_visualization.kinetic_plot import plot_kissinger, plot_jmak_results, plot_modified_jmak_results


def perform_kissinger_analysis(folder_path: str):
    """
    Perform Kissinger analysis on multiple DSC files in a folder.

    Args:
        folder_path (str): Path to the folder containing DSC files.
    """
    peak_temperatures = []
    heating_rates = []

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Adjust this if your files have a different extension
            file_path = os.path.join(folder_path, filename)

            try:
                # Import DSC data
                data = dsc_importer(file_path=file_path, manufacturer='Setaram')

                # Separate heating and cooling
                temp_heating = data['temperature'][:np.argmax(data['temperature'])]
                hf_heating = data['heat_flow'][:np.argmax(data['temperature'])]

                # Calculate heating rate
                time_heating = data['time'][:np.argmax(data['temperature'])]
                heating_rate = (temp_heating[-1] - temp_heating[0]) / (time_heating[-1] - time_heating[0])

                # Find peak temperature (you might need to adjust this depending on your data)
                peak_temp = temp_heating[np.argmax(np.gradient(hf_heating))]

                peak_temperatures.append(peak_temp)
                heating_rates.append(heating_rate)

                print(
                    f"Processed {filename}: Peak Temperature = {peak_temp:.2f} K, Heating Rate = {heating_rate:.2f} K/min")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Perform Kissinger analysis
    if len(peak_temperatures) > 1:
        e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(np.array(peak_temperatures), np.array(heating_rates))

        # Plot Kissinger results
        plot_kissinger(np.array(peak_temperatures), np.array(heating_rates), e_a, a, r_squared)

        print(f"\nKissinger Analysis Results:")
        print(f"Activation Energy (E_a) = {e_a / 1000:.2f} ± {se_e_a / 1000:.2f} kJ/mol")
        print(f"Pre-exponential Factor (A) = {a:.2e} min^-1")
        print(f"R-squared = {r_squared:.4f}")
    else:
        print("\nInsufficient data for Kissinger analysis. At least two different heating rates are required.")


def separate_heating_cooling(temperature, time, heat_flow):
    """Separate heating and cooling portions of the DSC curve."""
    peak_index = np.argmax(temperature)
    return (
        temperature[:peak_index], time[:peak_index], heat_flow[:peak_index],
        temperature[peak_index:], time[peak_index:], heat_flow[peak_index:]
    )


def plot_full_dsc_curve(temp_heating, hf_heating, temp_cooling, hf_cooling):
    """Plot the full DSC curve."""
    plt.figure(figsize=(12, 6))
    plt.plot(temp_heating, hf_heating, label='Heating')
    plt.plot(temp_cooling, hf_cooling, label='Cooling')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Flow')
    plt.title('Full DSC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    file_path = "C:/Users/Pablo/Desktop/Mercedes Duran/2024-JUL/MS/MAC250-MS20.txt"
    try:
        data = dsc_importer(file_path=file_path, manufacturer='Setaram')
    except Exception as e:
        print(f"Error importing Setaram data: {str(e)}")
        return

    # Separate heating and cooling
    temp_heating, time_heating, hf_heating, temp_cooling, time_cooling, hf_cooling = separate_heating_cooling(
        data['temperature'], data['time'], data['heat_flow']
    )

    # Plot full DSC curve
    plot_full_dsc_curve(temp_heating, hf_heating, temp_cooling, hf_cooling)

    # Ask user to select temperature range for analysis
    temp_min = float(input("Enter minimum temperature for analysis: "))
    temp_max = float(input("Enter maximum temperature for analysis: "))
    mask = (temp_heating >= temp_min) & (temp_heating <= temp_max)

    temp_analysis = temp_heating[mask]
    time_analysis = time_heating[mask]
    hf_analysis = hf_heating[mask]

    # Calculate heating rate
    heating_rate = (temp_analysis[-1] - temp_analysis[0]) / (time_analysis[-1] - time_analysis[0])

    # Calculate transformed fraction for the selected range
    transformed_fraction = calculate_transformed_fraction(hf_analysis, time_analysis)

    try:
        # Kissinger analysis
        t_p = temp_analysis[np.argmax(np.gradient(transformed_fraction))]  # Estimate peak temperature
        e_a, a, se_e_a, se_ln_a, r_squared_kissinger = kissinger_method(np.array([t_p]), np.array([heating_rate]))

        plot_kissinger(np.array([t_p]), np.array([heating_rate]), e_a, a, r_squared_kissinger)

        # JMAK analysis
        n_jmak, k_jmak, r_squared_jmak = jmak_method(time_analysis, transformed_fraction)
        fitted_curve_jmak = jmak_equation(time_analysis, k_jmak, n_jmak)

        plot_jmak_results(time_analysis, transformed_fraction, fitted_curve_jmak, n_jmak, k_jmak, r_squared_jmak)

        # Modified JMAK analysis
        k0, n_mod, r_squared_mod = fit_modified_jmak(temp_analysis, transformed_fraction, temp_analysis[0],
                                                     heating_rate, e_a)
        fitted_curve_mod = modified_jmak_equation(temp_analysis, k0, n_mod, e_a, temp_analysis[0], heating_rate)

        plot_modified_jmak_results(temp_analysis, transformed_fraction, fitted_curve_mod, k0, n_mod, e_a, r_squared_mod)

        # Print results
        print(
            f"Kissinger analysis: E_a = {e_a / 1000:.2f} ± {se_e_a / 1000:.2f} kJ/mol, A = {a:.2e} min^-1, R^2 = {r_squared_kissinger:.4f}")
        print(f"JMAK analysis: n = {n_jmak:.3f}, k = {k_jmak:.3e}, R^2 = {r_squared_jmak:.4f}")
        print(
            f"Modified JMAK analysis: k0 = {k0:.3e}, n = {n_mod:.3f}, E_a = {e_a / 1000:.2f} kJ/mol, R^2 = {r_squared_mod:.4f}")

        # Plot original DSC curve with analyzed region highlighted
        plt.figure(figsize=(10, 6))
        plt.plot(temp_heating, hf_heating, label='Heating')
        plt.plot(temp_cooling, hf_cooling, label='Cooling')
        plt.axvspan(temp_min, temp_max, color='yellow', alpha=0.3, label='Analyzed region')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Heat Flow')
        plt.title('DSC Curve with Analyzed Region')
        plt.legend()
        plt.show()

    except ValueError as e:
        print(f"Error in analysis: {str(e)}")


if __name__ == "__main__":
    main()
