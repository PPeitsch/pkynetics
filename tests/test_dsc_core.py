"""Tests for the complete DSC analysis workflow (DSCAnalyzer)."""

import numpy as np
import pytest

from pkynetics.technique_analysis.dsc import DSCAnalyzer, DSCExperiment
from pkynetics.technique_analysis.dsc.core import _step_region
from pkynetics.technique_analysis.dsc.thermal_events import ThermalEventDetector
from pkynetics.technique_analysis.dsc.types import GlassTransition

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

    # Peaks in both directions, by temperature
    assert [p.type for p in results["peaks"]] == ["exothermic", "endothermic"]
    exo, endo = results["peaks"]
    assert abs(exo.peak_temperature - 400) < 0.5
    assert abs(endo.peak_temperature - 500) < 0.5
    np.testing.assert_allclose(exo.enthalpy, enthalpy(0.8, 8), rtol=0.02)
    np.testing.assert_allclose(endo.enthalpy, enthalpy(1.5, 6), rtol=0.02)

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
    peaks = analyzer.peaks
    assert [(p.type, round(p.peak_temperature)) for p in peaks] == [
        ("exothermic", 400),
        ("endothermic", 500),
    ]


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
    melting = [p for p in peaks if p.type == "endothermic"]
    assert len(melting) == 1
    np.testing.assert_allclose(melting[0].enthalpy, enthalpy(1.5, 6), rtol=0.02)


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


@pytest.fixture
def polymer_experiment():
    """Run with a glass transition (a step) before the two peaks."""
    time = np.arange(0, 1800.0, 1.0)
    temperature = 300 + HEATING_RATE / 60 * time
    heat_flow = (
        0.2
        + 0.002 * (temperature - 300)
        + 0.5 / (1 + np.exp(-(temperature - 355) / 5.0))  # glass transition
        + gaussian(temperature, 420, -0.8, 8)  # cold crystallization
        + gaussian(temperature, 500, 1.5, 6)  # melting
    )
    return DSCExperiment(
        temperature=temperature,
        heat_flow=heat_flow,
        time=time,
        mass=MASS,
        sample_name="polymer",
    )


def test_glass_transition_does_not_create_a_peak(polymer_experiment):
    """The step is fitted around, not through: no spurious peak, right areas."""
    analyzer = DSCAnalyzer(
        polymer_experiment, event_detector=ThermalEventDetector(exo_up=False)
    )
    results = analyzer.analyze(baseline_method="linear")

    assert results["baseline"]["type"].startswith("stepped")
    assert len(results["peaks"]) == 2

    crystallization, melting = results["peaks"]
    assert crystallization.type == "exothermic"
    assert melting.type == "endothermic"
    np.testing.assert_allclose(crystallization.peak_temperature, 420, atol=2)
    np.testing.assert_allclose(melting.peak_temperature, 500, atol=2)
    np.testing.assert_allclose(crystallization.enthalpy, enthalpy(0.8, 8), rtol=0.1)
    np.testing.assert_allclose(melting.enthalpy, enthalpy(1.5, 6), rtol=0.1)


def test_glass_transition_still_reported(polymer_experiment):
    """The stepped baseline removes the step, so the raw-curve Tg is kept."""
    analyzer = DSCAnalyzer(
        polymer_experiment, event_detector=ThermalEventDetector(exo_up=False)
    )
    results = analyzer.analyze(baseline_method="linear")

    transitions = results["events"]["glass_transitions"]
    assert len(transitions) == 1
    np.testing.assert_allclose(transitions[0].midpoint_temperature, 355, atol=5)
    assert transitions[0].onset_temperature < transitions[0].endpoint_temperature


def test_single_baseline_across_the_step_is_worse(polymer_experiment):
    """detect_steps=False keeps the old behaviour: the step becomes a peak."""
    analyzer = DSCAnalyzer(
        polymer_experiment, event_detector=ThermalEventDetector(exo_up=False)
    )
    results = analyzer.analyze(baseline_method="linear", detect_steps=False)

    assert results["baseline"]["type"] == "linear"
    peaks = results["peaks"]
    assert len(peaks) > 2
    assert any(abs(peak.peak_temperature - 355) < 25 for peak in peaks)


def test_curve_without_step_is_unchanged(experiment):
    """No glass transition: a single baseline, as before."""
    analyzer = DSCAnalyzer(
        experiment, event_detector=ThermalEventDetector(exo_up=False)
    )
    results = analyzer.analyze(baseline_method="linear")

    assert results["baseline"]["type"] == "linear"
    assert len(results["peaks"]) == 2


def glass_transition(onset, endpoint):
    return GlassTransition(
        onset_temperature=onset,
        midpoint_temperature=(onset + endpoint) / 2,
        endpoint_temperature=endpoint,
        delta_cp=0.3,
        width=endpoint - onset,
    )


@pytest.mark.parametrize(
    "transition, expected",
    [
        (None, None),
        (glass_transition(340.0, 360.0), (340.0, 360.0)),
        (glass_transition(np.nan, 360.0), None),  # detector could not bracket it
        (glass_transition(340.0, np.nan), None),
        (glass_transition(360.0, 340.0), None),  # endpoint below onset
    ],
)
def test_step_region_of_a_transition(transition, expected):
    assert _step_region(transition) == expected


def test_unusable_transition_falls_back_to_one_baseline(experiment, monkeypatch):
    """A transition without a usable range must not produce step regions."""
    detector = ThermalEventDetector(exo_up=False)
    monkeypatch.setattr(
        detector,
        "detect_glass_transition",
        lambda *args, **kwargs: glass_transition(np.nan, np.nan),
    )
    analyzer = DSCAnalyzer(experiment, event_detector=detector)

    results = analyzer.analyze(baseline_method="linear")

    assert results["baseline"]["type"] == "linear"


# D21: the enthalpy of a peak of known area is recovered exactly
@pytest.mark.parametrize("dh_true", [50.0, 250.0, 500.0])
def test_enthalpy_is_exact_on_a_peak_of_known_area(dh_true):
    """analyze() returns the enthalpy the peak was built to have.

    Recorded against the eicosane example, where the reported enthalpy of
    fusion (286 J/g) sits ~16 % above the literature value (~247 J/g). The
    integration is not what is off: numerically integrating that file by
    hand gives ~283 J/g for any sensible pair of limits, and here the
    pipeline recovers a synthetic peak to better than 1 %. The discrepancy
    belongs to the measurement (the file's own TempCal is off by 1.3 K at
    the indium point, and the onset comes out ~1 K low to match), not to
    this code.
    """
    beta, mass = 1.0, 9.0  # K/min, mg
    temperature = np.linspace(283.15, 343.15, 6000)
    time = (temperature - temperature[0]) / (beta / 60)

    peak = gaussian(temperature, 310.15, 1.0, np.sqrt(2.0))
    # Scale so that the integral is exactly dh_true * mass (in mJ)
    peak *= dh_true * mass / np.trapezoid(peak, time)
    heat_flow = 0.02 * (temperature - temperature[0]) - 1.0 + peak

    experiment = DSCExperiment(
        temperature=temperature,
        heat_flow=heat_flow,
        time=time,
        mass=mass,
        heating_rate=beta,
        sample_name="synthetic",
    )
    results = DSCAnalyzer(
        experiment, event_detector=ThermalEventDetector(exo_up=True)
    ).analyze(baseline_method="polynomial", degree=2)

    assert len(results["peaks"]) == 1
    assert results["peaks"][0].enthalpy == pytest.approx(dh_true, rel=0.01)
