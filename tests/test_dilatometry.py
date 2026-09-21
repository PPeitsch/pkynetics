"""Tests for dilatometry curve analysis.

Two curves are used. The synthetic one is two linear thermal-expansion
segments joined by a sigmoidal transformation of known centre and width,
which lets the results be asserted tightly. The real one is the Zry-4
heating run fetched by ``pkynetics.data``, which has the curved baselines
and the noise the synthetic curve lacks — it is what showed that locating
the limits by deviation from an extrapolated tangent could not work
(issue #94).
"""

import numpy as np
import pytest

from pkynetics.data import load_dilatometry_cooling, load_dilatometry_heating
from pkynetics.technique_analysis.dilatometry import (
    analyze_dilatometry_curve,
    calculate_transformed_fraction_lever,
    find_optimal_margin,
    find_transformation_limits,
)
from pkynetics.technique_analysis.utilities import (
    analyze_range,
    detect_segment_direction,
    estimate_heating_rate,
    get_analysis_summary,
    get_transformation_metrics,
    validate_temperature_range,
)

# Transformation centred at 750 degC; the sigmoid is within 1 % of its
# limits by +-46 K, so the transformation really runs over ~705-795 degC
TRANSFORMATION_CENTRE = 750.0
TRANSFORMATION_WIDTH = 10.0


def dilatometry_curve(n_points=1000, cooling=False):
    """Two linear segments joined by a sigmoidal transformation."""
    temperature = np.linspace(600.0, 900.0, n_points)
    sigmoid = 1 / (
        1 + np.exp(-(temperature - TRANSFORMATION_CENTRE) / TRANSFORMATION_WIDTH)
    )
    strain = 1e-5 * (temperature - 600.0) - 2e-3 * sigmoid
    if cooling:
        return temperature[::-1], strain[::-1]
    return temperature, strain


# The Zry-4 alpha->beta contraction in the shipped sample, read off the
# local slope of the curve: it flattens from ~855 degC, is frankly negative
# between 870 and 918, and is back on a linear expansion by ~935 degC
REAL_TRANSFORMATION = (855.0, 935.0)


@pytest.fixture
def curve():
    return dilatometry_curve()


@pytest.fixture
def real_cooling_curve():
    """A cooling run, windowed onto the region with a linear baseline on
    either side of the transformation."""
    data = load_dilatometry_cooling()
    return analyze_range(
        np.asarray(data["temperature"]),
        np.asarray(data["relative_change"]),
        1040.0,
        700.0,
    )


@pytest.fixture
def real_curve():
    """The Zry-4 dilatometry run, fetched on demand."""
    data = load_dilatometry_heating()
    return np.asarray(data["temperature"]), np.asarray(data["relative_change"])


# What the analysis gets right
@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_midpoint_temperature_is_accurate(curve, method):
    """The midpoint lands on the centre of the transformation."""
    temperature, strain = curve

    result = analyze_dilatometry_curve(temperature, strain, method=method)

    assert result["mid_temperature"] == pytest.approx(TRANSFORMATION_CENTRE, abs=1.0)


@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_transformed_fraction_runs_from_zero_to_one(curve, method):
    """The transformed fraction is monotonic and spans the full range."""
    temperature, strain = curve

    result = analyze_dilatometry_curve(temperature, strain, method=method)
    fraction = result["transformed_fraction"]

    assert len(fraction) == len(temperature)
    assert fraction[0] == pytest.approx(0.0, abs=0.05)
    assert fraction[-1] == pytest.approx(1.0, abs=0.05)
    # Monotonic to within the numerical noise of the extrapolated tangents
    assert np.min(np.diff(fraction)) > -1e-3


@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_result_carries_the_inputs_and_the_method(curve, method):
    temperature, strain = curve

    result = analyze_dilatometry_curve(temperature, strain, method=method)

    assert result["method"] == method
    assert result["is_cooling"] is False
    np.testing.assert_array_equal(result["temperature"], temperature)
    np.testing.assert_array_equal(result["strain"], strain)
    assert len(result["before_extrapolation"]) == len(temperature)
    assert len(result["after_extrapolation"]) == len(temperature)


@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_cooling_curve_is_detected_and_analysed(method):
    """Reversing the ramp flags cooling and keeps the midpoint."""
    temperature, strain = dilatometry_curve(cooling=True)

    result = analyze_dilatometry_curve(temperature, strain, method=method)

    assert result["is_cooling"] is True
    assert result["mid_temperature"] == pytest.approx(TRANSFORMATION_CENTRE, abs=1.0)


def test_tangent_method_reports_fit_quality(curve):
    temperature, strain = curve

    result = analyze_dilatometry_curve(temperature, strain, method="tangent")

    assert "fit_quality" in result


# The transformation limits (issue #94)
@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_limits_bracket_the_synthetic_transformation(curve, method):
    """The limits land on the ~705-795 degC transformation, not on the data range."""
    temperature, strain = curve

    result = analyze_dilatometry_curve(temperature, strain, method=method)

    assert result["start_temperature"] == pytest.approx(705.0, abs=5.0)
    assert result["end_temperature"] == pytest.approx(795.0, abs=5.0)


@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_limits_stay_inside_the_data(curve, method):
    """Regression for #94: the limits used to be the first and last point."""
    temperature, strain = curve

    result = analyze_dilatometry_curve(temperature, strain, method=method)

    assert result["start_temperature"] > temperature[0]
    assert result["end_temperature"] < temperature[-1]


@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_limits_bracket_the_real_transformation(real_curve, method):
    """The limits find the alpha->beta contraction in the shipped Zry-4 run."""
    temperature, strain = real_curve
    expected_start, expected_end = REAL_TRANSFORMATION

    result = analyze_dilatometry_curve(temperature, strain, method=method)

    assert result["start_temperature"] == pytest.approx(expected_start, abs=20.0)
    assert result["end_temperature"] == pytest.approx(expected_end, abs=20.0)
    assert result["start_temperature"] > temperature[0]
    assert result["end_temperature"] < temperature[-1]


def test_both_methods_agree_on_the_limits(real_curve):
    """The methods differ in how they get the fraction, not in where the
    transformation is, so they share the limit detection."""
    temperature, strain = real_curve

    lever = analyze_dilatometry_curve(temperature, strain, method="lever")
    tangent = analyze_dilatometry_curve(temperature, strain, method="tangent")

    assert lever["start_temperature"] == pytest.approx(tangent["start_temperature"])
    assert lever["end_temperature"] == pytest.approx(tangent["end_temperature"])


def test_find_transformation_limits_returns_indices_in_array_order(real_curve):
    temperature, strain = real_curve

    start_idx, end_idx = find_transformation_limits(temperature, strain)

    assert 0 < start_idx < end_idx < len(temperature) - 1


def test_limits_on_a_real_cooling_run(real_cooling_curve):
    """The transformation on cooling runs from ~945 down to ~760 degC; the
    limits have to sit inside it, in cooling order, and not on the edges."""
    temperature, strain = real_cooling_curve

    result = analyze_dilatometry_curve(temperature, strain, method="lever")

    assert result["is_cooling"] is True
    assert result["start_temperature"] > result["end_temperature"]
    assert 760.0 < result["end_temperature"] < result["start_temperature"] < 945.0
    assert result["start_temperature"] < temperature[0]
    assert result["end_temperature"] > temperature[-1]


def test_a_spike_does_not_move_the_limits(curve):
    """Regression: a single bad point used to set the scale for everything,
    because the limits came from the peak of the derivative."""
    temperature, strain = curve
    clean = find_transformation_limits(temperature, strain)

    spiked = strain.copy()
    spiked[50] += 1e-3  # Half the size of the whole transformation

    assert find_transformation_limits(temperature, spiked) == clean


@pytest.mark.network
def test_a_curved_baseline_is_reported():
    """The full cooling run has a transformation inside its own baseline
    window, which the analysis cannot see past but does report."""
    data = load_dilatometry_cooling()
    temperature = np.asarray(data["temperature"])
    strain = np.asarray(data["relative_change"])

    with pytest.warns(UserWarning, match="baseline window is not linear"):
        find_transformation_limits(temperature, strain, is_cooling=True)


def test_find_transformation_limits_rejects_an_impossible_fraction(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="deviation_fraction"):
        find_transformation_limits(temperature, strain, deviation_fraction=1.5)


def test_find_transformation_limits_needs_enough_points():
    temperature = np.linspace(600.0, 900.0, 10)

    with pytest.raises(ValueError, match="Insufficient data points"):
        find_transformation_limits(temperature, temperature * 1e-5)


def test_find_transformation_limits_rejects_an_empty_baseline():
    """A margin so small it leaves a single point cannot define a baseline."""
    temperature, strain = dilatometry_curve(n_points=25)

    with pytest.raises(ValueError, match="fewer than 2 points"):
        find_transformation_limits(temperature, strain, margin=0.01)


def test_limits_fall_back_to_the_raw_signal_when_smoothing_fails(curve):
    """A polyorder wider than the data is reported, not swallowed: the
    limits still come back, located on the unsmoothed signal."""
    temperature, strain = curve

    with pytest.warns(UserWarning, match="Smoothing the strain failed"):
        start_idx, end_idx = find_transformation_limits(
            temperature, strain, polyorder=len(temperature)
        )

    assert 0 < start_idx < end_idx < len(temperature) - 1


def test_a_wider_deviation_fraction_narrows_the_limits(curve):
    """The fraction is the knob: more of the excursion counted as baseline
    means a tighter bracket."""
    temperature, strain = curve

    narrow = find_transformation_limits(temperature, strain, deviation_fraction=0.01)
    wide = find_transformation_limits(temperature, strain, deviation_fraction=0.20)

    assert narrow[0] < wide[0]
    assert narrow[1] > wide[1]


# Error handling
def test_mismatched_arrays_raise(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="same length"):
        analyze_dilatometry_curve(temperature[:-1], strain)


def test_insufficient_points_raise():
    temperature = np.linspace(600.0, 900.0, 10)

    with pytest.raises(ValueError, match="Insufficient data points"):
        analyze_dilatometry_curve(temperature, temperature * 1e-5)


def test_unsupported_method_raises(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="Unsupported method"):
        analyze_dilatometry_curve(temperature, strain, method="spline")


# Supporting functions
def test_find_optimal_margin_returns_a_usable_margin(curve):
    temperature, strain = curve

    margin = find_optimal_margin(temperature, strain, is_cooling=False)

    assert 0.0 < margin < 0.5


def test_find_optimal_margin_prefers_the_widest_acceptable_margin(curve):
    """A narrower window always fits a line better, so picking the best R2
    would use the least baseline available (issue #94)."""
    temperature, strain = curve

    margin = find_optimal_margin(temperature, strain, is_cooling=False, min_r2=0.99)

    assert margin == pytest.approx(0.4)


def test_calculate_transformed_fraction_lever(curve):
    temperature, strain = curve

    fraction, before, after = calculate_transformed_fraction_lever(
        temperature, strain, start_temp=705.0, end_temp=795.0, margin_percent=0.2
    )

    assert len(fraction) == len(temperature)
    assert fraction[0] == pytest.approx(0.0, abs=0.05)
    assert fraction[-1] == pytest.approx(1.0, abs=0.05)
    assert len(before) == len(after) == len(temperature)


def test_detect_segment_direction():
    heating = np.linspace(600.0, 900.0, 100)

    assert detect_segment_direction(heating) is False
    assert detect_segment_direction(heating[::-1]) is True
    # Too few points to tell
    assert detect_segment_direction(np.array([700.0])) is False


def test_detect_segment_direction_ignores_nans():
    temperature = np.linspace(600.0, 900.0, 100)
    temperature[10:20] = np.nan

    assert detect_segment_direction(temperature) is False


def test_analyze_range_slices_a_heating_segment(curve):
    temperature, strain = curve

    sliced_temp, sliced_strain = analyze_range(temperature, strain, 700.0, 800.0)

    assert sliced_temp.min() >= 700.0
    assert sliced_temp.max() <= 800.0
    assert len(sliced_temp) == len(sliced_strain)


def test_analyze_range_rejects_temperatures_outside_the_data(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="within data range"):
        analyze_range(temperature, strain, 500.0, 800.0)


def test_analyze_range_rejects_a_reversed_range_for_heating(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="must be less than"):
        analyze_range(temperature, strain, 800.0, 700.0)


def test_analyze_range_rejects_a_reversed_range_for_cooling():
    temperature, strain = dilatometry_curve(cooling=True)

    with pytest.raises(ValueError, match="must be greater than"):
        analyze_range(temperature, strain, 700.0, 800.0)


def test_validate_temperature_range(curve):
    temperature, _ = curve

    valid, _ = validate_temperature_range(temperature, 700.0, 800.0)
    assert valid is True

    invalid, message = validate_temperature_range(temperature, 100.0, 800.0)
    assert invalid is False
    assert message


def test_estimate_heating_rate_with_and_without_time(curve):
    temperature, _ = curve
    # 10 degC/min over the 300 K ramp
    time = np.linspace(0.0, 30.0 * 60.0, len(temperature))

    assert estimate_heating_rate(temperature, time) == pytest.approx(10.0, rel=1e-6)
    # Without a time base the rate is not knowable
    assert np.isnan(estimate_heating_rate(temperature))
    assert np.isnan(estimate_heating_rate(temperature[:1]))

    with pytest.raises(ValueError, match="same length"):
        estimate_heating_rate(temperature, time[:-1])


def test_get_transformation_metrics(curve):
    temperature, strain = curve
    result = analyze_dilatometry_curve(temperature, strain, method="lever")

    metrics = get_transformation_metrics(result)

    assert metrics["temperature_span"] == pytest.approx(
        result["end_temperature"] - result["start_temperature"], rel=1e-9
    )


def test_get_analysis_summary_mentions_the_temperatures(curve):
    temperature, strain = curve
    result = analyze_dilatometry_curve(temperature, strain, method="lever")

    summary = get_analysis_summary(result)

    assert isinstance(summary, str)
    assert f"{result['mid_temperature']:.1f}" in summary
