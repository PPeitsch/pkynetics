"""Specific heat capacity example: step method with blank, sapphire and sample.

The three Setaram runs share one stepped temperature program (isotherms
at 100, 200, 300, 400 and 500 degC joined by 5 K/min ramps). The heat
absorbed in each heating step, relative to the isothermal levels, gives
the mean Cp over the step.

- Single-step (sample - blank): needs only the sample mass, but carries the
  instrument's temperature-dependent sensitivity unless calibrated.
- Three-step (ASTM E1269 ratio with sapphire): cancels the sensitivity, but
  needs the sapphire mass, which is not recorded in the file header. Set
  SAPPHIRE_MASS to run it.
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.data_import import dsc_importer
from pkynetics.technique_analysis.dsc import CpCalculator, CpMethod, plot_cp

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "pkynetics", "data", "heat_capacity"
)
SAMPLE_MASS = 58.30  # mg, aluminium sample, from the file header
SAPPHIRE_MASS: Optional[float] = None  # mg, not in the file header


def load(name: str) -> Dict[str, np.ndarray]:
    """Load a Setaram run in K and s."""
    data = dsc_importer(os.path.join(DATA_DIR, f"{name}.txt"), manufacturer="Setaram")
    return {
        "time": data["time"],  # s
        "temperature": data["sample_temperature"] + 273.15,
        "heat_flow": data["heat_flow"],  # mW, endothermic negative
    }


def main() -> None:
    blank, sapphire, sample = load("zero"), load("sapphire"), load("sample")

    # Setaram: endothermic heat flow is negative
    calculator = CpCalculator(exo_up=True)

    single = calculator.calculate_cp(
        sample["temperature"],
        sample["heat_flow"],
        SAMPLE_MASS,
        heating_rate=5.0,
        method=CpMethod.SINGLE_STEP,
        operation_mode="stepped",
        time=sample["time"],
        blank_heat_flow=blank["heat_flow"],
    )

    print("Single-step Cp (blank-corrected, uncalibrated):")
    for (t_from, t_to), cp, u in zip(
        single.metadata["step_temperatures"],
        single.specific_heat,
        single.uncertainty,
    ):
        print(
            f"  {t_from - 273.15:6.1f} -> {t_to - 273.15:6.1f} degC: "
            f"{cp:.3f} +/- {u:.3f} J/(g K)"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_cp(ax, single, celsius=True)

    if SAPPHIRE_MASS is None:
        print("\nThree-step Cp skipped: set SAPPHIRE_MASS (mg) to compute it.")
    else:
        three = calculator.calculate_cp(
            sample["temperature"],
            sample["heat_flow"],
            SAMPLE_MASS,
            heating_rate=5.0,
            method=CpMethod.THREE_STEP,
            operation_mode="stepped",
            time=sample["time"],
            blank_heat_flow=blank["heat_flow"],
            reference_data={
                "heat_flow": sapphire["heat_flow"],
                "mass": SAPPHIRE_MASS,
                "temperature": sapphire["temperature"],
            },
        )
        print("\nThree-step Cp (sapphire reference):")
        for temp, cp, u in zip(
            three.temperature, three.specific_heat, three.uncertainty
        ):
            print(f"  {temp - 273.15:6.1f} degC: {cp:.3f} +/- {u:.3f} J/(g K)")
        plot_cp(ax, three, celsius=True)

    ax.set_title("Specific heat capacity, step method")
    plt.show()


if __name__ == "__main__":
    main()
