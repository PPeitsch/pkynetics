"""
Stepped-Isothermal Heat Capacity (Cp) Example
==============================================

This example demonstrates how to calculate the specific heat capacity (Cp) from
a stepped-isothermal DSC experiment, using both real data from files and
generated synthetic data.

The workflow for each part (real and synthetic) includes:
1.  Loading or generating data for blank, reference (sapphire), and sample runs.
2.  Plotting the raw signals to visualize the input data.
3.  Using the `CpCalculator` with `operation_mode=OperationMode.STEPPED`.
4.  Plotting the final calculated Cp result.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.data_import import dsc_importer
from pkynetics.technique_analysis.dsc import CpCalculator, CpMethod, OperationMode

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_data_file_path(filename: str) -> Path:
    """Constructs a robust path to the heat capacity data files."""
    try:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir
        while not (project_root / "src").exists() and project_root.parent != project_root:
            project_root = project_root.parent

        if not (project_root / "src").exists():
            raise FileNotFoundError("Could not determine project root.")

        file_path = project_root / "src" / "pkynetics" / "data" / "heat_capacity" / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at expected path: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error determining data file path for {filename}: {e}")
        raise


def load_cp_data(filename: str) -> dict:
    """Loads a Cp experiment file."""
    path = get_data_file_path(filename)
    logger.info(f"Loading data from: {path}")
    return dsc_importer(file_path=str(path), manufacturer="Setaram")


def plot_raw_signals(blank, reference, sample, title: str):
    """Plots the three raw heat flow signals."""
    plt.figure(figsize=(12, 7))
    plt.title(title)
    plt.plot(blank["temperature"], blank["heat_flow"], label="Blank (Empty Pan)")
    plt.plot(reference["temperature"], reference["heat_flow"], label="Reference (Sapphire)")
    plt.plot(sample["temperature"], sample["heat_flow"], label="Sample")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Heat Flow (mW)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()


def plot_cp_result(cp_result, true_cp_func=None, title: str = "Calculated Specific Heat Capacity (Cp)"):
    """Plots the final calculated Cp."""
    plt.figure(figsize=(12, 7))
    plt.title(title)
    if true_cp_func:
        plt.plot(
            cp_result.temperature, true_cp_func(cp_result.temperature), "k--", label="True Cp"
        )
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
    plt.show()


def run_real_data_analysis():
    """Analyzes Cp from real data files."""
    logger.info("\n--- Analyzing Real Data from Files ---")

    # 1. Load data
    blank_data = load_cp_data("zero.txt")
    sapphire_data = load_cp_data("sapphire.txt")
    sample_data = load_cp_data("sample.txt")

    # 2. Plot raw signals
    plot_raw_signals(blank_data, sapphire_data, sample_data, "Raw Signals from Real Data Files")

    # 3. Perform analysis
    sapphire_mass, sample_mass = 25.3, 20.8  # Masses for these specific files
    sapphire_cp_func = lambda T: 1.0289 + 2.35e-4 * T

    calculator = CpCalculator()
    reference_data = {
        "temperature": sapphire_data["temperature"],
        "heat_flow": sapphire_data["heat_flow"],
        "mass": sapphire_mass,
        "cp": sapphire_cp_func(sapphire_data["temperature"]),
        "blank_heat_flow": blank_data["heat_flow"],
    }

    cp_result = calculator.calculate_cp(
        temperature=sample_data["temperature"],
        heat_flow=sample_data["heat_flow"],
        sample_mass=sample_mass,
        heating_rate=20.0,
        method=CpMethod.THREE_STEP,
        operation_mode=OperationMode.STEPPED,
        reference_data=reference_data,
    )

    # 4. Plot final result
    plot_cp_result(cp_result, title="Calculated Cp from Real Data")


def generate_synthetic_stepped_data(cp_func, mass: float, noise_level: float = 0.03):
    """Generates synthetic data for a stepped program."""
    temp_isotherms_K = np.arange(100, 501, 100) + 273.15
    heating_rate_K_per_s = 20.0 / 60.0
    isothermal_time_s = 10 * 60
    time_segments, temp_segments = [], []
    current_time = 0.0

    for i, target_temp in enumerate(temp_isotherms_K):
        if i > 0:
            prev_temp = temp_isotherms_K[i - 1]
            ramp_duration = (target_temp - prev_temp) / heating_rate_K_per_s
            time_segments.append(np.linspace(current_time, current_time + ramp_duration, int(ramp_duration / 2)))
            temp_segments.append(np.linspace(prev_temp, target_temp, int(ramp_duration / 2)))
            current_time += ramp_duration

        time_segments.append(np.linspace(current_time, current_time + isothermal_time_s, int(isothermal_time_s / 2)))
        temp_segments.append(np.full(int(isothermal_time_s / 2), target_temp))
        current_time += isothermal_time_s

    # FIX: Ensure time is strictly monotonic before calculating gradient
    full_time = np.concatenate(time_segments)
    full_temp = np.concatenate(temp_segments)
    _, unique_indices = np.unique(full_time, return_index=True)
    time = full_time[unique_indices]
    temperature = full_temp[unique_indices]

    rate_K_per_s = np.gradient(temperature, time, edge_order=2)
    heat_flow = cp_func(temperature) * mass * rate_K_per_s
    noise = np.random.normal(0, noise_level, len(time))
    return {"time": time, "temperature": temperature, "heat_flow": heat_flow + noise}


def run_synthetic_data_analysis():
    """Analyzes Cp from generated synthetic data."""
    logger.info("\n--- Analyzing Synthetic Data ---")

    # 1. Generate data
    sapphire_cp_func = lambda T: 1.0289 + 2.35e-4 * T
    sample_cp_func = lambda T: 1.5 + 0.002 * (T - (100 + 273.15))
    sapphire_mass, sample_mass = 25.0, 15.0

    blank_data = generate_synthetic_stepped_data(lambda T: 0.0, 0)
    sapphire_data = generate_synthetic_stepped_data(sapphire_cp_func, sapphire_mass)
    sample_data = generate_synthetic_stepped_data(sample_cp_func, sample_mass)

    # 2. Plot raw signals
    plot_raw_signals(blank_data, sapphire_data, sample_data, "Raw Signals from Synthetic Data")

    # 3. Perform analysis
    calculator = CpCalculator()
    reference_data = {
        "temperature": sapphire_data["temperature"],
        "heat_flow": sapphire_data["heat_flow"],
        "mass": sapphire_mass,
        "cp": sapphire_cp_func(sapphire_data["temperature"]),
        "blank_heat_flow": blank_data["heat_flow"],
    }

    cp_result = calculator.calculate_cp(
        temperature=sample_data["temperature"],
        heat_flow=sample_data["heat_flow"],
        sample_mass=sample_mass,
        heating_rate=20.0,
        method=CpMethod.THREE_STEP,
        operation_mode=OperationMode.STEPPED,
        reference_data=reference_data,
    )

    # 4. Plot final result
    plot_cp_result(cp_result, true_cp_func=sample_cp_func, title="Calculated Cp from Synthetic Data")


def main():
    """Main function to run both real and synthetic Cp examples."""
    np.random.seed(42)

    # Run analysis for real data from files
    run_real_data_analysis()

    # Run analysis for generated synthetic data
    run_synthetic_data_analysis()


if __name__ == "__main__":
    main()