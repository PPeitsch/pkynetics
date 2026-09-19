"""Tests for the complete DSC analysis workflow (DSCAnalyzer)."""

import numpy as np
import pytest

from pkynetics.technique_analysis.dsc import DSCAnalyzer, DSCExperiment
from pkynetics.technique_analysis.dsc.thermal_events import ThermalEventDetector

HEATING_RATE = 10.0  # K/min
MASS = 5.0  # mg


def gaussian(x, center, amplitude, width):
    return amplitude * np.exp(-(((x - center) / width) ** 2))


def enthalpy(amplitude, width):
    """J/g of a Gaussian peak at HEATING_RATE and MASS."""
    return amplitude * width * np.sqrt(np.pi) / (HEATING_RATE / 60) / MASS


@pytest.fixture
def experiment():
    """10 K/min run with a sloped baseline, cold crystallization and melting."""
    time = np.arange(0, 1800.0, 1.0)
    temperature = 300 + HEATING_RATE / 60 * time
    baseline = 0.2 + 0.002 * (temperature - 300)
    heat_flow = (
        baseline
        + gaussian(temperature, 400, -0.8, 8)  # exothermic
        + gaussian(temperature, 500, 1.5, 6)  # endothermic
    )
    return DSCExperiment(
        temperature=temperature,
        heat_flow=heat_flow,
        time=time,
        mass=MASS,
        sample_name="synthetic",
    )


def test_heating_rate_from_data(experiment):
    np.testing.assert_allclose(experiment.heating_rate, HEATING_RATE)


def test_analyze(experiment):
    analyzer = DSCAnalyzer(experiment)
    results = analyzer.analyze(baseline_method="linear")

    assert set(results) == {"peaks", "events", "baseline"}
    assert results["baseline"]["type"] == "linear"
    np.testing.assert_allclose(analyzer.baseline[:50], experiment.heat_flow[:50])

    # Upward peaks of the corrected signal: the melting peak
    assert len(results["peaks"]) == 1
    peak = results["peaks"][0]
    assert abs(peak.peak_temperature - 500) < 0.5
    np.testing.assert_allclose(peak.enthalpy, enthalpy(1.5, 6), rtol=0.02)

    events = results["events"]
    assert len(events["melting"]) == 1
    assert len(events["crystallization"]) == 1
    np.testing.assert_allclose(
        events["melting"][0].enthalpy, enthalpy(1.5, 6), rtol=0.02
    )
    np.testing.assert_allclose(
        events["crystallization"][0].enthalpy, -enthalpy(0.8, 8), rtol=0.02
    )
    assert events["melting"][0].baseline_subtracted


def test_analyze_auto_baseline(experiment):
    results = DSCAnalyzer(experiment).analyze()
    assert results["baseline"]["type"] in ("linear", "polynomial")
    assert len(results["events"]["melting"]) == 1


def test_analyze_exo_up(experiment):
    """Inverted data with an exo-up event detector gives the same events."""
    inverted = DSCExperiment(
        temperature=experiment.temperature,
        heat_flow=-experiment.heat_flow,
        time=experiment.time,
        mass=MASS,
    )
    analyzer = DSCAnalyzer(inverted, event_detector=ThermalEventDetector(exo_up=True))
    events = analyzer.analyze(baseline_method="linear")["events"]

    assert [round(e.peak_temperature) for e in events["melting"]] == [500]
    assert [round(e.peak_temperature) for e in events["crystallization"]] == [400]


def test_cooling_enthalpy_positive(experiment):
    """On cooling (negative heating rate) enthalpy magnitudes stay positive."""
    cooling = DSCExperiment(
        temperature=experiment.temperature[::-1].copy(),
        heat_flow=experiment.heat_flow[::-1].copy(),
        time=experiment.time,
        mass=MASS,
    )
    assert cooling.heating_rate < 0
    peaks = DSCAnalyzer(cooling).analyze(baseline_method="linear")["peaks"]
    assert len(peaks) == 1
    np.testing.assert_allclose(peaks[0].enthalpy, enthalpy(1.5, 6), rtol=0.02)


def test_experiment_validation():
    with pytest.raises(ValueError, match="same length"):
        DSCExperiment(
            temperature=np.arange(10.0),
            heat_flow=np.zeros(9),
            time=np.arange(10.0),
            mass=1.0,
        )
    with pytest.raises(ValueError, match="mass"):
        DSCExperiment(
            temperature=np.arange(10.0),
            heat_flow=np.zeros(10),
            time=np.arange(10.0),
            mass=0.0,
        )
