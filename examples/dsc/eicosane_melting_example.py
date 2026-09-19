"""DSC analysis example: melting of eicosane (TA Instruments data).

Imports a TA Universal Analysis export, selects the heating segment of the
heat/cool cycle, corrects the baseline and characterizes the melting peak
(extrapolated onset, peak temperature, enthalpy of fusion).

Sample (from the file header): eicosane, 9.00 mg, 1 K/min, "Exotherm Up".
Literature for comparison: melting ~36.4-36.8 degC, enthalpy of fusion
~247 J/g.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.data_import import dsc_importer
from pkynetics.technique_analysis.dsc import (
    DataValidator,
    DSCAnalyzer,
    DSCExperiment,
    ThermalEventDetector,
    plot_dsc_analysis,
)

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
    "pkynetics",
    "data",
    "dsc",
    "sample_dsc_tainstruments.txt",
)
SAMPLE_MASS = 9.00  # mg, "Size" in the file header


def main() -> None:
    data = dsc_importer(DATA_FILE)  # TA format detected from the header

    # DSC analysis works in K and s; TA exports use degC and min
    time = data["time"] * 60
    temperature = data["temperature"] + 273.15
    heat_flow = data["heat_flow"]

    # The run is a heat/cool cycle: analyze the heating segment
    program = DataValidator.detect_temperature_program(temperature, time)
    print(f"Temperature program: {program['type']}")
    for segment in program["segments"]:
        print(
            f"  {segment['type']:<10} {segment['rate']:6.2f} K/min, "
            f"points {segment['start_idx']}-{segment['end_idx']}"
        )
    heating = next(s for s in program["segments"] if s["type"] == "heating")
    region = slice(heating["start_idx"], heating["end_idx"])

    # Skip the start-up transient (first 3 K) and the turning point
    selected = np.zeros(len(temperature), dtype=bool)
    selected[region] = True
    selected &= temperature > temperature[region][0] + 3
    selected &= temperature < temperature[region].max() - 0.5

    experiment = DSCExperiment(
        temperature=temperature[selected],
        heat_flow=heat_flow[selected],
        time=time[selected],
        mass=SAMPLE_MASS,
        heating_rate=heating["rate"],
        sample_name="Eicosane",
    )

    # The file header states "Exotherm Up"
    analyzer = DSCAnalyzer(
        experiment, event_detector=ThermalEventDetector(exo_up=True)
    )
    results = analyzer.analyze(baseline_method="polynomial", degree=2)

    print(f"\nBaseline: {results['baseline']['type']}")
    for peak in results["peaks"]:
        print(
            f"\n{peak.type.capitalize()} peak\n"
            f"  Onset (extrapolated): {peak.onset_temperature - 273.15:.2f} degC\n"
            f"  Peak:                 {peak.peak_temperature - 273.15:.2f} degC\n"
            f"  Endset:               {peak.endset_temperature - 273.15:.2f} degC\n"
            f"  FWHM:                 {peak.peak_width:.2f} K\n"
            f"  Enthalpy:             {peak.enthalpy:.1f} J/g"
        )

    events = results["events"]
    print(
        f"\nEvents: {len(events['glass_transitions'])} glass transition(s), "
        f"{len(events['crystallization'])} crystallization(s), "
        f"{len(events['melting'])} melting event(s)"
    )

    plot_dsc_analysis(analyzer, celsius=True)
    plt.show()


if __name__ == "__main__":
    main()
