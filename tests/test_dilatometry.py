"""Tests for dilatometry curve analysis (L1).

The synthetic curve is the shape the analysis is meant to handle: two
linear thermal-expansion segments joined by a sigmoidal transformation of
known centre and width. It lets the correct results be asserted tightly,
and it is what shows that the transformation *limits* are not yet right —
see ``test_transformation_start_is_far_too_early`` and
``test_tangent_method_returns_the_whole_data_range``, both xfail.
"""

import numpy as np
import pytest

from pkynetics.technique_analysis.dilatometry import (
    analyze_dilatometry_curve,
    calculate_transformed_fraction_lever,
    find_optimal_margin,
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


@pytest.fixture
def curve():
    return dilatometry_curve()


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


# L1: the transformation limits
@pytest.mark.xfail(
    strict=True,
    reason="L1: the lever start is ~15 K below the transformation, and the "
    "end ~15 K above. Needs real data to fix, not just a tighter threshold.",
)
def test_transformation_start_is_far_too_early(curve):
    """The lever limits should bracket the ~705-795 degC transformation."""
    temperature, strain = curve

    result = analyze_dilatometry_curve(temperature, strain, method="lever")

    assert result["start_temperature"] == pytest.approx(705.0, abs=5.0)
    assert result["end_temperature"] == pytest.approx(795.0, abs=5.0)


@pytest.mark.xfail(
    strict=True,
    reason="L1: the tangent method returns the first and last temperature of "
    "the data, i.e. it does not locate the transformation limits at all.",
)
def test_tangent_method_returns_the_whole_data_range(curve):
    temperature, strain = curve

    result = analyze_dilatometry_curve(temperature, strain, method="tangent")

    assert result["start_temperature"] > temperature[0]
    assert result["end_temperature"] < temperature[-1]


def test_lever_limits_currently_overshoot_by_about_15_kelvin(curve):
    """Pins today's behaviour so a change to it is deliberate.

    Paired with the xfail above: that one says where the limits should be,
    this one says where they are.
    """
    temperature, strain = curve

    result = analyze_dilatometry_curve(temperature, strain, method="lever")

    assert result["start_temperature"] == pytest.approx(690.0, abs=2.0)
    assert result["end_temperature"] == pytest.approx(810.0, abs=2.0)


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
