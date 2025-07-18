"""
DSC Heat Capacity (Cp) Calculation Example
==========================================

This example demonstrates how to calculate the specific heat capacity (Cp) from
DSC data using the three-step method. The workflow includes:
1.  Generating synthetic data for a blank run, a reference material (sapphire),
    and the sample.
2.  Using the CpCalculator to perform the calculation.
3.  Calibrating the instrument using the sapphire data.
4.  Applying the calibration to get the final Cp value.
5.  Visualizing all steps of the process.
"""

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.technique_analysis.dsc import CpCalculator, CpMethod


def generate_cp_data(
    temperature: np.ndarray, mass: float, true_cp_func, noise_level: float = 0.05
) -> np.ndarray:
    """Generates a synthetic heat flow signal for a Cp measurement."""
    heating_rate_K_per_s = 10.0 / 60.0  # 10 K/min
    # Heat Flow (mW) = Cp (J/gK) * mass (mg) * heating_rate (K/s)
    # Note: J/gK = mJ/mgK, so units are consistent
    base_heat_flow = true_cp_func(temperature) * mass * heating_rate_K_per_s
    noise = np.random.normal(0, noise_level, len(temperature))
    return base_heat_flow + noise


def main():
    """Main function to run the Cp calculation example."""
    print("--- DSC Heat Capacity (Cp) Calculation Example ---")

    # --- 1. Define experimental conditions and true Cp functions ---
    temperature = np.linspace(300, 600, 301)
    heating_rate_K_per_min = 10.0

    # True Cp for Sapphire (from NIST data, simplified)
    sapphire_cp_func = lambda T: 1.0289 + 2.35e-4 * T + 1.68e-7 * T**2

    # True Cp for our unknown sample (e.g., a fictional polymer)
    sample_cp_func = lambda T: 1.5 + 0.002 * (T - 300)

    # --- 2. Generate synthetic data for the three steps ---
    blank_heat_flow = generate_cp_data(
        temperature, mass=0, true_cp_func=lambda T: 0, noise_level=0.02
    )
    sapphire_mass = 25.0  # mg
    sapphire_heat_flow = generate_cp_data(temperature, sapphire_mass, sapphire_cp_func)
    sample_mass = 15.0  # mg
    sample_heat_flow = generate_cp_data(temperature, sample_mass, sample_cp_func)

    # The three-step method requires the heat flow signals to be subtracted
    # from the blank (or pan) measurement.
    sample_signal = sample_heat_flow - blank_heat_flow
    sapphire_signal = sapphire_heat_flow - blank_heat_flow

    # --- 3. Calculate Cp using the three-step method ---
    calculator = CpCalculator()

    reference_data = {
        "temperature": temperature,
        "heat_flow": sapphire_signal,
        "mass": sapphire_mass,
        "cp": sapphire_cp_func(temperature),
    }

    cp_result = calculator.calculate_cp(
        temperature=temperature,
        heat_flow=sample_signal,
        sample_mass=sample_mass,
        heating_rate=heating_rate_K_per_min,
        method=CpMethod.THREE_STEP,
        reference_data=reference_data,
        use_calibration=False,  # We first calculate without instrument calibration
    )

    print("\n--- Cp Calculation Results ---")
    print(f"Method: {cp_result.method.value}")
    print(f"Average calculated Cp: {np.mean(cp_result.specific_heat):.4f} J/(g·K)")

    # --- 4. Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(temperature, sample_signal, label=f"Sample Signal (mass={sample_mass}mg)")
    ax1.plot(
        temperature, sapphire_signal, label=f"Sapphire Signal (mass={sapphire_mass}mg)"
    )
    ax1.plot(temperature, blank_heat_flow, label="Blank Signal", linestyle="--")
    ax1.set_title("Raw DSC Signals (Corrected for Blank)")
    ax1.set_ylabel("Heat Flow (mW)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(temperature, sample_cp_func(temperature), "k--", label="True Sample Cp")
    ax2.plot(
        cp_result.temperature,
        cp_result.specific_heat,
        "o",
        markersize=4,
        label="Calculated Sample Cp",
    )
    ax2.fill_between(
        cp_result.temperature,
        cp_result.specific_heat - cp_result.uncertainty,
        cp_result.specific_heat + cp_result.uncertainty,
        alpha=0.3,
        label="Uncertainty",
    )
    ax2.set_title("Calculated Specific Heat Capacity (Cp)")
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Cp (J/g·K)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
