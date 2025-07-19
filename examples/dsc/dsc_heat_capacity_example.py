"""
DSC Heat Capacity (Cp) Calculation Example
==========================================

This example demonstrates how to calculate the specific heat capacity (Cp) from
DSC data using both continuous and stepped-isothermal methods.

The workflow includes:
1.  Generating synthetic data for a continuous heating run.
2.  Calculating Cp using the three-step method on the continuous data.
3.  Generating synthetic data for a stepped-isothermal run.
4.  Calculating Cp using the `STEPPED` operation mode, which analyzes
    the stable isothermal segments.
5.  Visualizing the results of both methods.
"""

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.technique_analysis.dsc import CpCalculator, CpMethod, OperationMode


def generate_continuous_cp_data(
    temp: np.ndarray, mass: float, true_cp_func, noise_level: float = 0.05
) -> np.ndarray:
    """Generates a synthetic heat flow signal for a continuous Cp measurement."""
    heating_rate_K_per_s = 10.0 / 60.0  # 10 K/min
    base_hf = true_cp_func(temp) * mass * heating_rate_K_per_s
    noise = np.random.normal(0, noise_level, len(temp))
    return base_hf + noise


def generate_stepped_cp_data(
    temp_isotherms: list, true_cp_func, mass: float, noise_level: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a synthetic signal for a stepped-isothermal measurement."""
    heating_rate_K_per_min = 20.0
    isothermal_time_s = 30 * 60

    time_segments, temp_segments = [], []
    current_time = 0.0

    for i, target_temp in enumerate(temp_isotherms):
        # Isothermal hold
        hold_duration = isothermal_time_s
        hold_points = int(hold_duration / 5)
        time_segments.append(
            np.linspace(current_time, current_time + hold_duration, hold_points)
        )
        temp_segments.append(np.full(hold_points, target_temp))
        current_time += hold_duration

        # Heating ramp to next isotherm
        if i < len(temp_isotherms) - 1:
            next_target_temp = temp_isotherms[i + 1]
            ramp_duration = (next_target_temp - target_temp) / (
                heating_rate_K_per_min / 60.0
            )
            ramp_points = int(ramp_duration / 5)
            # Ensure at least 2 points for linspace
            if ramp_points > 1:
                time_segments.append(
                    np.linspace(current_time, current_time + ramp_duration, ramp_points)
                )
                temp_segments.append(
                    np.linspace(target_temp, next_target_temp, ramp_points)
                )
                current_time += ramp_duration

    # Concatenate and remove duplicate time points to prevent gradient errors
    full_time = np.concatenate(time_segments)
    full_temp = np.concatenate(temp_segments)

    _, unique_indices = np.unique(full_time, return_index=True)
    time = full_time[unique_indices]
    temperature = full_temp[unique_indices]

    # Calculate heat flow based on instantaneous heating rate
    rate_K_per_s = np.gradient(temperature, time)
    heat_flow = true_cp_func(temperature) * mass * rate_K_per_s
    noise = np.random.normal(0, noise_level, len(time))

    return time, temperature, heat_flow + noise


def main():
    """Main function to run the Cp calculation example."""
    print("--- DSC Heat Capacity (Cp) Calculation Example ---")
    np.random.seed(42)

    # --- Part 1: Continuous Method ---
    print("\n--- Running Continuous Three-Step Method ---")
    temp_cont = np.linspace(300, 700, 401)
    heating_rate_cont = 10.0
    sapphire_cp_func = lambda T: 1.0289 + 2.35e-4 * T
    sample_cp_func = lambda T: 1.5 + 0.002 * (T - 300)

    # Generate data
    blank_hf = generate_continuous_cp_data(temp_cont, 0, lambda T: 0)
    sapphire_mass, sample_mass = 25.0, 15.0
    sapphire_hf = generate_continuous_cp_data(
        temp_cont, sapphire_mass, sapphire_cp_func
    )
    sample_hf = generate_continuous_cp_data(temp_cont, sample_mass, sample_cp_func)

    # Calculate Cp
    calculator = CpCalculator()
    reference_data = {
        "temperature": temp_cont,
        "heat_flow": sapphire_hf - blank_hf,
        "mass": sapphire_mass,
        "cp": sapphire_cp_func(temp_cont),
    }
    cp_result_cont = calculator.calculate_cp(
        temperature=temp_cont,
        heat_flow=sample_hf - blank_hf,
        sample_mass=sample_mass,
        heating_rate=heating_rate_cont,
        method=CpMethod.THREE_STEP,
        reference_data=reference_data,
    )

    # --- Part 2: Stepped-Isothermal Method ---
    print("\n--- Running Stepped-Isothermal Method ---")
    isotherm_temps_C = np.arange(50, 701, 100)
    isotherm_temps_K = isotherm_temps_C + 273.15

    # Generate data for the sample
    time_step, temp_step, hf_step = generate_stepped_cp_data(
        isotherm_temps_K, sample_cp_func, sample_mass
    )

    # For simplicity, we assume the reference signal for the stepped method is just
    # its true value at the isotherm temperatures.
    ref_cp_at_isotherms = sapphire_cp_func(isotherm_temps_K)
    ref_signal_at_isotherms = (
        ref_cp_at_isotherms * sapphire_mass * (20.0 / 60.0)
    )  # Using ramp rate

    # We'll use a simplified reference for the stepped method calculation
    # In a real experiment, one would run the reference material through the same program
    # and extract the average heat flow from its stable regions.

    # To demonstrate the calculation, we'll use a simplified reference_data
    # where the Cp is known at the target temperatures.
    ref_data_stepped = {
        "temperature": temp_step,
        "heat_flow": np.interp(temp_step, isotherm_temps_K, ref_signal_at_isotherms),
        "mass": sapphire_mass,
        "cp": np.interp(temp_step, isotherm_temps_K, ref_cp_at_isotherms),
    }

    cp_result_step = calculator.calculate_cp(
        temperature=temp_step,
        heat_flow=hf_step,
        sample_mass=sample_mass,
        heating_rate=20.0,  # The ramp rate
        method=CpMethod.THREE_STEP,
        operation_mode=OperationMode.STEPPED,
        reference_data=ref_data_stepped,
    )

    # --- Visualization ---
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    # Plot continuous raw signals
    ax1.plot(temp_cont, sample_hf - blank_hf, label=f"Sample Signal (Continuous)")
    ax1.plot(temp_cont, sapphire_hf - blank_hf, label=f"Sapphire Signal (Continuous)")
    ax1.set_title("Raw DSC Signals for Continuous Method")
    ax1.set_ylabel("Heat Flow (mW)")
    ax1.legend()
    ax1.grid(True)

    # Plot stepped raw signal vs time and temp
    ax2.plot(time_step / 60, temp_step, label="Temperature Program")
    ax2.set_title("Stepped-Isothermal Program")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Temperature (K)")
    ax2.grid(True)

    ax3.plot(time_step / 60, hf_step, label="Sample Signal (Stepped)")
    ax3.set_title("Stepped-Isothermal Heat Flow")
    ax3.set_xlabel("Time (min)")
    ax3.set_ylabel("Heat Flow (mW)")
    ax3.grid(True)

    # Plot final Cp results
    ax4.plot(temp_cont, sample_cp_func(temp_cont), "k--", label="True Sample Cp")
    ax4.plot(
        cp_result_cont.temperature,
        cp_result_cont.specific_heat,
        label="Calculated Cp (Continuous)",
        alpha=0.8,
    )
    ax4.plot(
        cp_result_step.temperature,
        cp_result_step.specific_heat,
        "ro",
        markersize=8,
        label="Calculated Cp (Stepped)",
    )
    ax4.set_title("Final Calculated Specific Heat Capacity (Cp)")
    ax4.set_xlabel("Temperature (K)")
    ax4.set_ylabel("Cp (J/g·K)")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
