"""
Stepped-Isothermal Heat Capacity (Cp) Example
==============================================

This example demonstrates the three-step method (ASTM E1269 ratio with a
sapphire reference) for calculating Cp from a stepped-isothermal
experiment, on synthetic blank, reference and sample runs with a known Cp.

The workflow follows the standard procedure:
1.  Generate blank, reference (sapphire) and sample runs on the same program.
2.  Subtract the blank signal from both the sample and reference signals.
3.  Calculate Cp from the heat absorbed in each heating step (step method),
    using the masses and the known Cp of the reference material.
4.  Each key step is visualized with a dedicated plot.

For the same analysis on the bundled real Setaram runs, see
``stepped_cp_real_data_example.py``.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.technique_analysis.dsc import CpCalculator, CpMethod, OperationMode

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_analysis(
    title: str,
    sample_data,
    ref_data,
    blank_data,
    sample_mass,
    ref_mass,
    ref_cp_func,
    true_cp_func=None,
):
    """Runs the full Cp analysis and plotting workflow."""
    logger.info(f"\n--- Starting Analysis: {title} ---")

    # --- Step 1: Blank Subtraction ---
    corrected_sample_hf = sample_data["heat_flow"] - blank_data["heat_flow"]
    corrected_ref_hf = ref_data["heat_flow"] - blank_data["heat_flow"]

    plt.figure(figsize=(12, 7))
    plt.title(f"{title}: Step 1 - Blank-Subtracted Signals")
    plt.plot(sample_data["temperature"], corrected_sample_hf, label="Sample - Blank")
    plt.plot(ref_data["temperature"], corrected_ref_hf, label="Reference - Blank")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Corrected DSC Signal (a.u.)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    # --- Step 2: Perform Cp Calculation ---
    calculator = CpCalculator()
    reference_for_calc = {
        "temperature": ref_data["temperature"],
        "heat_flow": ref_data["heat_flow"],
        "mass": ref_mass,
        "cp": ref_cp_func(ref_data["temperature"]),
    }

    cp_result = calculator.calculate_cp(
        temperature=sample_data["temperature"],
        heat_flow=sample_data["heat_flow"],
        sample_mass=sample_mass,
        heating_rate=20.0,
        method=CpMethod.THREE_STEP,
        operation_mode=OperationMode.STEPPED,
        reference_data=reference_for_calc,
        time=sample_data["time"],
        blank_heat_flow=blank_data["heat_flow"],
    )

    # --- Step 3: Plot Final Result ---
    plt.figure(figsize=(12, 7))
    plt.title(f"{title}: Final Calculated Cp")
    if true_cp_func:
        plt.plot(
            sample_data["temperature"],
            true_cp_func(sample_data["temperature"]),
            "k--",
            label="True Cp",
        )

    if len(cp_result.temperature) > 100:
        plt.plot(
            cp_result.temperature,
            cp_result.specific_heat,
            "-",
            color="red",
            linewidth=2,
            label="Calculated Cp",
        )
    else:
        plt.plot(
            cp_result.temperature,
            cp_result.specific_heat,
            "ro",
            markersize=8,
            label="Calculated Cp (Stepped)",
        )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Cp (J/g·K)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.ylim(bottom=0)  # Cp should not be negative
    plt.show()


def generate_synthetic_stepped_data(cp_func, mass: float, noise_level: float = 0.03):
    """Generates synthetic data for a stepped program."""
    temp_isotherms_K = np.arange(100, 501, 100) + 273.15
    rate_K_per_s = 20.0 / 60.0
    isothermal_time_s = 10 * 60
    time_segs, temp_segs = [], []
    current_time = 0.0
    for i, temp in enumerate(temp_isotherms_K):
        if i > 0:
            prev_temp = temp_isotherms_K[i - 1]
            duration = (temp - prev_temp) / rate_K_per_s
            points = int(duration / 2)
            time_segs.append(np.linspace(current_time, current_time + duration, points))
            temp_segs.append(np.linspace(prev_temp, temp, points))
            current_time += duration
        points = int(isothermal_time_s / 2)
        time_segs.append(
            np.linspace(current_time, current_time + isothermal_time_s, points)
        )
        temp_segs.append(np.full(points, temp))
        current_time += isothermal_time_s

    full_time, full_temp = np.concatenate(time_segs), np.concatenate(temp_segs)
    _, unique_indices = np.unique(full_time, return_index=True)
    time, temp = full_time[unique_indices], full_temp[unique_indices]

    rate = np.gradient(temp, time, edge_order=2)
    hf = cp_func(temp) * mass * rate
    noise = np.random.normal(0, noise_level, len(time))
    return {"time": time, "temperature": temp, "heat_flow": hf + noise}


def main():
    """Run the synthetic stepped Cp example."""
    np.random.seed(42)

    ref_cp_func_known = lambda T: 1.0289 + 2.35e-4 * T

    sample_mass_synth, ref_mass_synth = 15.0, 25.0
    true_cp_func_synth = lambda T: 1.5 + 0.002 * (T - (100 + 273.15))

    blank_synth = generate_synthetic_stepped_data(lambda T: 0.0, 0)
    ref_synth = generate_synthetic_stepped_data(ref_cp_func_known, ref_mass_synth)
    sample_synth = generate_synthetic_stepped_data(
        true_cp_func_synth, sample_mass_synth
    )

    run_analysis(
        "Synthetic Data",
        sample_synth,
        ref_synth,
        blank_synth,
        sample_mass_synth,
        ref_mass_synth,
        ref_cp_func_known,
        true_cp_func=true_cp_func_synth,
    )


if __name__ == "__main__":
    main()
