"""Smoke tests for DSC plotting functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pkynetics.technique_analysis.dsc import (  # noqa: E402
    CpCalculator,
    DSCAnalyzer,
    DSCExperiment,
    plot_cp,
    plot_dsc_analysis,
    plot_dsc_curve,
    plot_thermal_events,
)


@pytest.fixture
def analyzer():
    time = np.arange(0, 1800.0)
    temperature = 300 + time / 6
    heat_flow = (
        0.1
        - 0.8 * np.exp(-(((temperature - 400) / 8) ** 2))
        + 1.5 * np.exp(-(((temperature - 500) / 6) ** 2))
    )
    analyzer = DSCAnalyzer(
        DSCExperiment(temperature, heat_flow, time, mass=5.0, sample_name="test")
    )
    analyzer.analyze(baseline_method="linear")
    return analyzer


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_dsc_curve(analyzer):
    fig, ax = plt.subplots()
    exp = analyzer.experiment
    plot_dsc_curve(
        ax, exp.temperature, exp.heat_flow, analyzer.baseline, analyzer.peaks
    )
    labels = [line.get_label() for line in ax.get_lines()]
    assert "Heat flow" in labels and "Baseline" in labels
    assert ax.get_ylabel() == "Heat flow (mW, endo ↑)"
    assert any("ΔH" in text.get_text() for text in ax.texts)


def test_plot_thermal_events_celsius(analyzer):
    fig, ax = plt.subplots()
    exp = analyzer.experiment
    plot_thermal_events(
        ax, exp.temperature, analyzer.corrected_heat_flow, analyzer.events, celsius=True
    )
    texts = " ".join(text.get_text() for text in ax.texts)
    assert "126.9" in texts  # crystallization at 400 K in degC
    assert "226.9" in texts  # melting at 500 K in degC
    assert ax.get_xlabel() == "Temperature (°C)"


@pytest.mark.parametrize("stepped", [False, True])
def test_plot_cp(stepped):
    calculator = CpCalculator()
    if stepped:
        time = np.arange(0, 5400.0)
        temperature = np.interp(time, [0, 1800, 2400, 5400], [350, 350, 400, 400])
        heat_flow = 10.0 * np.gradient(temperature, time)
        result = calculator.calculate_cp(
            temperature,
            heat_flow,
            10.0,
            5.0,
            method="single_step",
            operation_mode="stepped",
            time=time,
        )
    else:
        temperature = np.linspace(350, 450, 500)
        result = calculator.calculate_cp(
            temperature, np.full(500, 10 / 6), 10.0, 10.0, method="single_step"
        )

    fig, ax = plt.subplots()
    plot_cp(ax, result)
    assert ax.get_ylabel() == "Specific heat capacity (J/(g·K))"
    assert ax.get_legend() is not None


def test_plot_dsc_analysis(analyzer):
    fig = plot_dsc_analysis(analyzer)
    assert len(fig.axes) == 2

    unanalyzed = DSCAnalyzer(analyzer.experiment)
    with pytest.raises(ValueError, match="analyze"):
        plot_dsc_analysis(unanalyzed)


def test_plot_glass_transition():
    """Tg is marked when present (detected on the raw curve)."""
    from pkynetics.technique_analysis.dsc import ThermalEventDetector

    temperature = np.linspace(300, 450, 1000)
    heat_flow = 0.3 / (1 + np.exp(-(temperature - 370) / 3))
    events = ThermalEventDetector().detect_events(temperature, heat_flow)

    fig, ax = plt.subplots()
    plot_thermal_events(ax, temperature, heat_flow, events)
    assert any(text.get_text().startswith("Tg = 370") for text in ax.texts)
