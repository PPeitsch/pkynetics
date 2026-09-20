"""Smoke tests for the plotting helpers.

These functions draw and call ``plt.show()``, so what is worth checking is
that they run over realistic inputs, put something on the axes and label
it. The module sat at ~10 % coverage, which is how the signature drift
behind K9 (``plot_horowitz_metzger``) went unnoticed.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pkynetics.result_visualization import (  # noqa: E402
    plot_activation_energy_vs_conversion,
    plot_arrhenius,
    plot_coats_redfern,
    plot_conversion_vs_temperature,
    plot_derivative_thermogravimetry,
    plot_dilatometry_analysis,
    plot_freeman_carroll,
    plot_horowitz_metzger,
    plot_jmak_results,
    plot_kissinger,
    plot_lever_rule,
    plot_modified_jmak_results,
    plot_raw_and_smoothed,
    plot_transformation_points,
    plot_transformed_fraction,
)
from pkynetics.technique_analysis.dilatometry import (  # noqa: E402
    analyze_dilatometry_curve,
)


@pytest.fixture(autouse=True)
def _no_blocking_show(monkeypatch):
    """Keep plt.show() from blocking, and close what each test opened."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


@pytest.fixture
def dilatometry_results():
    temperature = np.linspace(600.0, 900.0, 500)
    sigmoid = 1 / (1 + np.exp(-(temperature - 750.0) / 10.0))
    strain = 1e-5 * (temperature - 600.0) - 2e-3 * sigmoid
    results = analyze_dilatometry_curve(temperature, strain, method="lever")
    return temperature, strain, results


# Kinetic plots
def test_plot_arrhenius():
    temperatures = np.linspace(500.0, 700.0, 20)
    rate_constants = 1e10 * np.exp(-120000 / (8.314 * temperatures))

    plot_arrhenius(temperatures, rate_constants, 120000.0, 1e10)

    ax = plt.gca()
    assert ax.get_xlabel() == "1/T (K^-1)"
    assert len(ax.lines) == 2  # data and fit


def test_plot_conversion_vs_temperature():
    temperatures = [np.linspace(500.0, 700.0, 50)] * 3
    conversions = [np.linspace(0.01, 0.99, 50)] * 3
    heating_rates = [5.0, 10.0, 20.0]

    plot_conversion_vs_temperature(temperatures, conversions, heating_rates)

    ax = plt.gca()
    assert len(ax.lines) == 3
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "5.0 K/min",
        "10.0 K/min",
        "20.0 K/min",
    ]


def test_plot_derivative_thermogravimetry():
    temperatures = [np.linspace(500.0, 700.0, 50)] * 2
    conversions = [np.linspace(0.01, 0.99, 50)] * 2

    plot_derivative_thermogravimetry(temperatures, conversions, [5.0, 10.0])

    assert len(plt.gca().lines) == 2


def test_plot_activation_energy_vs_conversion():
    conversions = np.linspace(0.1, 0.9, 9)
    energies = np.full(9, 120.0)

    plot_activation_energy_vs_conversion(conversions, energies, "KAS")

    assert "KAS" in plt.gca().get_title()


def test_plot_kissinger():
    t_p = np.array([420.0, 435.0, 450.0])  # degC
    beta = np.array([5.0, 10.0, 20.0]) / 60  # degC/s

    plot_kissinger(t_p, beta, 120000.0, 1e10, 0.999)

    ax = plt.gcf().axes[0]
    assert len(ax.collections) == 1  # the scatter of experimental points
    assert len(ax.lines) == 1  # the theoretical line


def test_plot_kissinger_without_a_fit():
    """NaN parameters must annotate instead of drawing a line."""
    t_p = np.array([420.0, 435.0])
    beta = np.array([5.0, 10.0]) / 60

    plot_kissinger(t_p, beta, float("nan"), float("nan"), float("nan"))

    ax = plt.gcf().axes[0]
    assert len(ax.lines) == 0
    assert "Insufficient data" in ax.texts[0].get_text()


def test_plot_jmak_results():
    time = np.linspace(0.1, 100.0, 100)
    fitted = 1 - np.exp(-((0.01 * time) ** 2))

    plot_jmak_results(time, fitted, fitted, 2.0, 0.01, 0.999)

    assert len(plt.gcf().axes) == 2


def test_plot_modified_jmak_results():
    temperature = np.linspace(600.0, 900.0, 100)
    fitted = 1 / (1 + np.exp(-(temperature - 750.0) / 10.0))

    plot_modified_jmak_results(temperature, fitted, fitted, 1e5, 2.0, 120000.0, 0.999)

    assert len(plt.gcf().axes) >= 1


# Model-specific plots
def test_plot_coats_redfern():
    x = np.linspace(1.2, 1.8, 50)
    y = -12.0 * x + 5.0

    plot_coats_redfern(x, y, x, y, 120000.0, 1e10, 0.999)

    assert len(plt.gca().lines) >= 1


def test_plot_freeman_carroll():
    x = np.linspace(-0.001, 0.0, 50)
    y = -14000.0 * x + 1.0

    plot_freeman_carroll(x, y, x, y, 120000.0, 1.0, 0.999)

    assert len(plt.gca().lines) >= 1


def test_plot_horowitz_metzger_takes_the_heating_rate():
    """Regression for K9: the wrapper must forward the heating rate."""
    temperature = np.linspace(400.0, 800.0, 500)
    alpha = np.clip(1 / (1 + np.exp(-(temperature - 600.0) / 20.0)), 1e-3, 1 - 1e-3)

    fig, ax = plot_horowitz_metzger(temperature, alpha, 10.0)

    assert ax.get_xlabel().startswith("θ")
    assert len(ax.lines) >= 1
    plt.close(fig)


# Dilatometry plots
def test_plot_raw_and_smoothed(dilatometry_results):
    temperature, strain, _ = dilatometry_results
    _, ax = plt.subplots()

    plot_raw_and_smoothed(ax, temperature, strain, strain, "lever")

    assert len(ax.lines) == 2
    assert "Lever" in ax.get_title()


def test_plot_transformation_points(dilatometry_results):
    temperature, strain, results = dilatometry_results
    _, ax = plt.subplots()

    plot_transformation_points(ax, temperature, strain, results)

    assert len(ax.lines) >= 1
    assert ax.get_legend() is not None


def test_plot_lever_rule(dilatometry_results):
    temperature, strain, results = dilatometry_results
    _, ax = plt.subplots()

    plot_lever_rule(ax, temperature, strain, results)

    assert len(ax.lines) >= 1


def test_plot_transformed_fraction_marks_the_three_points(dilatometry_results):
    temperature, _, results = dilatometry_results
    _, ax = plt.subplots()

    plot_transformed_fraction(ax, temperature, results)

    # One vertical line per transformation point
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"Start", "Mid", "End"} <= set(labels)
    assert ax.get_ylim() == (-0.1, 1.1)


def test_plot_transformed_fraction_flips_the_ends_when_cooling(dilatometry_results):
    """On cooling the fraction starts at 1 and ends at 0."""
    temperature, _, results = dilatometry_results
    results = dict(results, is_cooling=True)
    _, ax = plt.subplots()

    plot_transformed_fraction(ax, temperature, results)

    annotations = [t.get_text() for t in ax.texts]
    assert any(text.startswith("Start") and "100.0%" in text for text in annotations)
    assert any(text.startswith("End") and "0.0%" in text for text in annotations)


def test_plot_dilatometry_analysis_builds_four_panels(dilatometry_results):
    temperature, strain, results = dilatometry_results

    fig = plot_dilatometry_analysis(temperature, strain, strain, results, "lever")

    assert len(fig.axes) == 4
    plt.close(fig)
