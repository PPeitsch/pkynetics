"""Tests for DilatometryAnalyzer.

The analyzer is a second way into the same analysis, so most of what matters
is that it agrees with the free functions rather than drifting from them.
"""

import numpy as np
import pytest

from pkynetics.data import load_dilatometry_heating
from pkynetics.technique_analysis import DilatometryAnalyzer
from pkynetics.technique_analysis.dilatometry import (
    analyze_dilatometry_curve,
    find_transformation_limits,
)

TRANSFORMATION_CENTRE = 750.0
TRANSFORMATION_WIDTH = 10.0


@pytest.fixture
def curve():
    temperature = np.linspace(600.0, 900.0, 1000)
    sigmoid = 1 / (
        1 + np.exp(-(temperature - TRANSFORMATION_CENTRE) / TRANSFORMATION_WIDTH)
    )
    return temperature, 1e-5 * (temperature - 600.0) - 2e-3 * sigmoid


@pytest.fixture
def real_curve():
    data = load_dilatometry_heating()
    return np.asarray(data["temperature"]), np.asarray(data["relative_change"])


@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_it_matches_the_free_function(curve, method):
    """The defaults of the analyzer are the defaults of the function."""
    temperature, strain = curve

    from_class = DilatometryAnalyzer().analyze(temperature, strain, method=method)
    from_function = analyze_dilatometry_curve(temperature, strain, method=method)

    assert from_class["start_temperature"] == from_function["start_temperature"]
    assert from_class["end_temperature"] == from_function["end_temperature"]
    assert from_class["method"] == from_function["method"]


def test_the_limits_agree_with_the_temperatures_reported(curve):
    """Regression: the limits used to be located a second time, with the
    analyzer's own margin, which could disagree with what the analysis ran on.
    They are read back off the result instead."""
    temperature, strain = curve
    analyzer = DilatometryAnalyzer()

    result = analyzer.analyze(temperature, strain)

    start_idx, end_idx = analyzer.limits
    assert temperature[start_idx] == pytest.approx(result["start_temperature"])
    assert temperature[end_idx] == pytest.approx(result["end_temperature"])


def test_nothing_is_kept_before_analysing():
    analyzer = DilatometryAnalyzer()

    assert analyzer.temperature is None
    assert analyzer.limits is None
    assert analyzer.result is None
    assert analyzer.is_cooling is None


def test_the_inputs_are_kept(curve):
    temperature, strain = curve
    analyzer = DilatometryAnalyzer()

    analyzer.analyze(temperature, strain)

    assert analyzer.temperature is not None
    assert np.array_equal(analyzer.temperature, temperature)
    assert np.array_equal(analyzer.strain, strain)
    assert analyzer.is_cooling is False
    assert analyzer.result is not None


def test_the_settings_are_passed_through(curve):
    """One name per concept: limits_margin is what the functions call `margin`,
    `find_inflection_margin` and `limits_margin`."""
    temperature, strain = curve

    from_class = DilatometryAnalyzer(limits_margin=0.15).find_limits(
        temperature, strain
    )
    from_function = find_transformation_limits(temperature, strain, margin=0.15)

    assert from_class == from_function


def test_find_limits_detects_the_direction(curve):
    """Cooling is read off the data when it is not given."""
    temperature, strain = curve
    cooling_t, cooling_s = temperature[::-1], strain[::-1]
    analyzer = DilatometryAnalyzer()

    start_idx, end_idx = analyzer.find_limits(cooling_t, cooling_s)

    assert start_idx < end_idx
    assert cooling_t[start_idx] > cooling_t[end_idx]


def test_one_analyzer_serves_several_curves(curve, real_curve):
    """The settings are configuration, the results are per call: analysing a
    second curve replaces the state rather than mixing with it."""
    analyzer = DilatometryAnalyzer()

    analyzer.analyze(*curve)
    first = analyzer.limits

    analyzer.analyze(*real_curve)

    assert analyzer.limits != first
    assert analyzer.temperature is not None
    assert len(analyzer.temperature) == len(real_curve[0])


def test_an_unsupported_method_is_rejected(curve):
    with pytest.raises(ValueError, match="Unsupported method"):
        DilatometryAnalyzer().analyze(*curve, method="parabolic")
