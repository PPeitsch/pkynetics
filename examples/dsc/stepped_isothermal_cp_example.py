"""
Stepped-Isothermal Heat Capacity (Cp) Example
==============================================

This example demonstrates the three-step method for calculating Cp from a
stepped-isothermal experiment, using both real data from files and
synthetic data.

The workflow follows the standard procedure:
1.  Load data for blank, reference (sapphire), and sample runs.
2.  Subtract the blank signal from both the sample and reference signals.
3.  Calculate the final Cp using the corrected signals, masses, and the
    known Cp of the reference material.
4.  Each key step is visualized with a dedicated plot.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.data_import import CustomImporter
from pkynetics.technique_analysis.dsc import CpCalculator, CpMethod, OperationMode

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_data_file_path(filename: str) -> Path:
    """Constructs a robust path to the heat capacity data files."""
    try:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir
        while (
            not (project_root / "src").exists() and project_root.parent != project_root
        ):
            project_root = project_root.parent
        if not (project_root / "src").exists():
            raise FileNotFoundError("Could not determine project root.")
        file_path = (
            project_root / "src" / "pkynetics" / "data" / "heat_capacity" / filename
        )
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found at expected path: {file_path}"
            )
        return file_path
    except Exception as e:
        logger.error(f"Error finding data file for {filename}: {e}")
        raise


def load_real_cp_data(filename: str) -> dict:
    """Loads a real Cp experiment file using CustomImporter."""
    path = get_data_file_path(filename)
    logger.info(f"Loading real data from: {path}")
    importer = CustomImporter(
        file_path=str(path),
        column_names=["index", "time", "furnace_temp", "temperature_c", "tg", "heat_flow"],
        separator=";",
        decimal=".",
        skiprows=14,
        encoding="utf-16le",
    )
    data = importer.import_data()
    data["temperature"] = data.pop("temperature_c") + 273.15
    return data


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
        "blank_heat_flow": blank_data["heat_flow"],
    }

    cp_result = calculator.calculate_cp(
        temperature=sample_data["temperature"],
        heat_flow=sample_data["heat_flow"],
        sample_mass=sample_mass,
        heating_rate=20.0,
        method=CpMethod.THREE_STEP,
        operation_mode=OperationMode.STEPPED,
        reference_data=reference_for_calc,
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
    """Main function to run both real and synthetic Cp examples."""
    np.random.seed(42)

    # --- Part 1: Real Data Analysis ---
    sample_mass_real, ref_mass_real = 20.8, 25.3
    ref_cp_func_known = lambda T: 1.0289 + 2.35e-4 * T

    try:
        blank_real = load_real_cp_data("zero.txt")
        ref_real = load_real_cp_data("sapphire.txt")
        sample_real = load_real_cp_data("sample.txt")
        run_analysis(
            "Real Data",
            sample_real,
            ref_real,
            blank_real,
            sample_mass_real,
            ref_mass_real,
            ref_cp_func_known,
        )
    except Exception as e:
        logger.error(f"Could not complete real data analysis: {e}")

    # --- Part 2: Synthetic Data Analysis ---
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
