"""Tests for dilatometry curve analysis.

Two curves are used. The synthetic one is two linear thermal-expansion
segments joined by a sigmoidal transformation of known centre and width,
which lets the results be asserted tightly. The real one is the Zry-4
heating run fetched by ``pkynetics.data``, which has the curved baselines
and the noise the synthetic curve lacks — it is what showed that locating
the limits by deviation from an extrapolated tangent could not work
(issue #94).
"""

import warnings

import numpy as np
import pytest

from pkynetics.data import load_dilatometry_cooling, load_dilatometry_heating
from pkynetics.technique_analysis.dilatometry import (
    DetectionContext,
    DilatometryAnalyzer,
    TransformationLimits,
    _strain_derivative,
    analyze_dilatometry_curve,
    available_detectors,
    calculate_transformed_fraction_lever,
    derivative_limits,
    find_optimal_margin,
    find_transformation_limits,
    get_detector,
    max_backward_step,
    tangent_method,
)
from pkynetics.technique_analysis.dilatometry.detection.adaptive import stable_margin
from pkynetics.technique_analysis.dilatometry.detection.statistical import (
    _residual_structure,
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


# The Zry-4 alpha->beta transformation in the shipped heating run, which the
# instrument file records as a DIL805 quenching dilatometer ramp at 10 K/s.
#
# Two different things were being asked of one number here, so they are now two.
#
# PHYSICAL_TRANSFORMATION is the published range. At 10 K/s the transformation
# is not at equilibrium: the start shifts upwards with heating rate up to about
# 500 K/s, and the end stays near 960 degC. Reported measurement uncertainty on
# these temperatures is about +/-10 K, and the range itself moves with oxygen
# content and heat treatment, so it is wide on purpose -- it says the detector
# is finding the alpha->beta transformation and not some other feature.
#
# MEASURED_ONSET is what this detector reports on this file, to 0.5 K. It is a
# regression guard, not a physical claim: 839 degC is the foot of the
# contraction, where the derivative leaves the baseline scatter (2-5 % of the
# peak excursion) and falls monotonically from there. The ~855 degC that earlier
# versions of this test used is where the flattening becomes *visible* on a
# plot, which is an eyeball reading of the same event, not a better one.
PHYSICAL_TRANSFORMATION = (810.0, 980.0)
MEASURED_ONSET = (839.0, 936.4)


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
def test_limits_fall_in_the_published_transformation_range(real_curve, method):
    """The limits land on the alpha->beta transformation, not another feature."""
    temperature, strain = real_curve
    low, high = PHYSICAL_TRANSFORMATION

    result = analyze_dilatometry_curve(temperature, strain, method=method)

    assert low <= result["start_temperature"] <= high
    assert low <= result["end_temperature"] <= high
    assert result["start_temperature"] < result["end_temperature"]
    assert result["start_temperature"] > temperature[0]
    assert result["end_temperature"] < temperature[-1]


@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_limits_match_the_measured_onset(real_curve, method):
    """Regression guard: this detector on this file, to half a kelvin."""
    temperature, strain = real_curve
    expected_start, expected_end = MEASURED_ONSET

    result = analyze_dilatometry_curve(temperature, strain, method=method)

    assert result["start_temperature"] == pytest.approx(expected_start, abs=0.5)
    assert result["end_temperature"] == pytest.approx(expected_end, abs=0.5)


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
    """The beta->alpha transformation on cooling runs from ~945 down to ~760
    degC. Both ends are read off the curve, so they are matched to within 20 K,
    the same tolerance as the heating run."""
    temperature, strain = real_cooling_curve

    result = analyze_dilatometry_curve(temperature, strain, method="lever")

    assert result["is_cooling"] is True
    assert result["start_temperature"] > result["end_temperature"]
    assert result["start_temperature"] == pytest.approx(945.0, abs=20.0)
    assert result["end_temperature"] == pytest.approx(760.0, abs=20.0)
    assert result["start_temperature"] < temperature[0]
    assert result["end_temperature"] > temperature[-1]


def test_the_derivative_is_taken_against_temperature_not_index(curve):
    """A ramp that slows down partway through must not look like a change in
    dS/dT: the local fit differentiates against the index, so it has to be
    divided by dT/di. Resampling the same curve onto an uneven grid cannot
    move the limits."""
    temperature, strain = curve
    even = find_transformation_limits(temperature, strain)

    # Same curve, sampled twice as densely below 750 degC as above it
    dense = np.linspace(600.0, 750.0, 900)
    sparse = np.linspace(750.0, 900.0, 450)[1:]
    uneven_t = np.concatenate([dense, sparse])
    uneven_s = np.interp(uneven_t, temperature, strain)

    start_idx, end_idx = find_transformation_limits(uneven_t, uneven_s)

    assert uneven_t[start_idx] == pytest.approx(temperature[even[0]], abs=5.0)
    assert uneven_t[end_idx] == pytest.approx(temperature[even[1]], abs=5.0)


def test_a_stalled_ramp_does_not_become_the_peak_excursion():
    """Where the temperature holds, dT/di is zero and dS/dT is undefined. Those
    points are interpolated over instead of becoming an infinite excursion that
    every threshold is then measured against."""
    ramp = np.linspace(600.0, 900.0, 600)
    hold = np.full(120, 900.0)  # An isothermal hold at the end of the ramp
    temperature = np.concatenate([ramp, hold])
    strain = 1e-5 * (temperature - 600.0) - 2e-3 * (
        1 / (1 + np.exp(-(temperature - TRANSFORMATION_CENTRE) / TRANSFORMATION_WIDTH))
    )

    start_idx, end_idx = find_transformation_limits(temperature, strain)

    assert np.isfinite(temperature[start_idx])
    assert temperature[start_idx] == pytest.approx(705.0, abs=15.0)
    assert temperature[end_idx] == pytest.approx(795.0, abs=15.0)


def test_a_flat_temperature_is_reported():
    """With no ramp at all there are no baseline windows to speak of, which is
    caught with a clearer message than anything the derivative could give."""
    temperature = np.full(100, 800.0)
    strain = np.linspace(0.0, 1e-3, 100)

    with pytest.raises(ValueError, match="Temperature range is too small"):
        find_transformation_limits(temperature, strain)


def test_the_derivative_falls_back_to_differencing_without_a_window():
    """Too few points for a local fit: dS/dT is differenced directly."""
    temperature = np.linspace(600.0, 900.0, 40)
    strain = 2e-5 * (temperature - 600.0)

    derivative = _strain_derivative(temperature, strain, 0, 2)

    assert derivative == pytest.approx(np.full(40, 2e-5))


def test_a_temperature_that_never_moves_gives_no_derivative():
    """dT/di is exactly zero throughout, so there is nothing to divide by."""
    temperature = np.zeros(50)
    strain = np.linspace(0.0, 1e-3, 50)

    with pytest.warns(UserWarning, match="temperature does not change"):
        derivative = _strain_derivative(temperature, strain, 5, 2)

    assert np.all(derivative == 0.0)


def test_the_derivative_stays_finite_on_a_flat_temperature():
    """dS/dT is undefined there, but it may not come back as inf or nan: the
    callers measure thresholds against the peak of this array."""
    temperature = np.full(100, 800.0)
    strain = np.linspace(0.0, 1e-3, 100)

    derivative = _strain_derivative(temperature, strain, 5, 2)

    assert np.all(np.isfinite(derivative))


def test_the_limits_are_not_dominated_by_the_sampling_rate(real_cooling_curve):
    """Regression: with dS/dT differenced point to point, the scatter of the
    baseline grew as the spacing shrank. On this run, recorded every 0.25 K, it
    reached 7 % of the peak excursion and set the threshold, pulling the limits
    ~30 and ~110 K inside the transformation (915-870 against ~945-760)."""
    temperature, strain = real_cooling_curve

    start_idx, end_idx = find_transformation_limits(
        temperature, strain, is_cooling=True
    )

    assert temperature[start_idx] > 930.0
    assert temperature[end_idx] < 790.0


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


def test_limits_fall_back_to_the_raw_signal_when_the_local_fit_fails(curve):
    """A polyorder wider than the data is reported, not swallowed: the
    limits still come back, from a plain difference of the raw signal."""
    temperature, strain = curve

    with pytest.warns(UserWarning, match="local fit for the derivative failed"):
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


# --- The types the pieces come back as (#103.1) ---------------------------


def test_transformation_limits_still_unpack_as_a_pair(real_curve):
    """The named tuple is what makes the new type non-breaking."""
    temperature, strain = real_curve

    limits = find_transformation_limits(temperature, strain)
    start_idx, end_idx = limits

    assert (limits.start_idx, limits.end_idx) == (start_idx, end_idx)
    assert limits == (start_idx, end_idx)


def test_fit_quality_is_a_dataclass_but_the_result_carries_a_mapping(real_curve):
    """The dataclass is internal; callers index the same keys as before."""
    temperature, strain = real_curve

    result = analyze_dilatometry_curve(temperature, strain, method="tangent")
    fit_quality = result["fit_quality"]

    assert isinstance(fit_quality, dict)
    assert set(fit_quality) == {
        "r2_start",
        "r2_end",
        "margin_used",
        "deviation_fraction",
        "warnings",
    }


# --- The transformed fraction is left raw and measured (#103.3) -----------


@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_result_reports_how_far_the_fraction_goes_backwards(real_curve, method):
    temperature, strain = real_curve

    result = analyze_dilatometry_curve(temperature, strain, method=method)

    # Measured on the shipped heating run; reported, not smoothed away.
    assert result["max_backward_step"] == pytest.approx(0.0103, abs=5e-4)


def test_max_backward_step_is_zero_for_a_rising_fraction():
    assert max_backward_step(np.linspace(0.0, 1.0, 50)) == 0.0


def test_max_backward_step_measures_the_worst_single_step():
    fraction = np.array([0.0, 0.5, 0.3, 0.9, 0.85, 1.0])

    assert max_backward_step(fraction) == pytest.approx(0.2)


def test_max_backward_step_handles_a_degenerate_fraction():
    assert max_backward_step(np.array([0.5])) == 0.0


def noisy_curve(scale=2e-5, n_points=1000):
    """The synthetic transformation with enough noise to push the fraction
    backwards. No example run needed, so this covers the warning without a
    network marker."""
    temperature = np.linspace(600.0, 900.0, n_points)
    sigmoid = 1 / (
        1 + np.exp(-(temperature - TRANSFORMATION_CENTRE) / TRANSFORMATION_WIDTH)
    )
    strain = 1e-5 * (temperature - 600.0) - 2e-3 * sigmoid
    return temperature, strain + np.random.normal(0.0, scale, n_points)


def test_tangent_warns_when_the_fraction_steps_too_far_backwards():
    """The threshold is what turns a measured backward step into a warning."""
    temperature, strain = noisy_curve()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = tangent_method(
            temperature,
            strain,
            is_cooling=False,
            margin_percent=0.2,
            max_backward_step_warn=0.01,
        )

    backward_step = result["max_backward_step"]
    assert backward_step > 0.01
    messages = [w for w in result["fit_quality"]["warnings"] if "backwards" in w]
    assert len(messages) == 1
    assert f"{backward_step:.2%}" in messages[0]


def test_tangent_stays_quiet_when_the_backward_step_is_under_the_threshold():
    """Same curve, same backward step: only the threshold changes."""
    temperature, strain = noisy_curve()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = tangent_method(
            temperature,
            strain,
            is_cooling=False,
            margin_percent=0.2,
            max_backward_step_warn=0.5,
        )

    assert result["max_backward_step"] > 0.0
    assert not [w for w in result["fit_quality"]["warnings"] if "backwards" in w]


# --- Detection is its own axis (#26) --------------------------------------


def test_the_default_detector_is_what_the_module_already_did(curve):
    """`detection` defaults to the behaviour that predates the parameter."""
    temperature, strain = curve

    default = find_transformation_limits(temperature, strain)
    explicit = find_transformation_limits(temperature, strain, detection="derivative")

    assert default == explicit


def test_available_detectors_lists_what_detection_accepts():
    assert "derivative" in available_detectors()
    assert available_detectors() == sorted(available_detectors())


def test_an_unknown_detector_says_which_ones_exist(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="Unknown detection method"):
        find_transformation_limits(temperature, strain, detection="offset_typo")

    with pytest.raises(ValueError, match="derivative"):
        find_transformation_limits(temperature, strain, detection="offset_typo")


def test_the_detector_name_is_case_insensitive(curve):
    temperature, strain = curve

    assert find_transformation_limits(
        temperature, strain, detection="DERIVATIVE"
    ) == find_transformation_limits(temperature, strain)


@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_detection_and_method_are_chosen_independently(curve, method):
    """The two axes cross: any detector goes with either method."""
    temperature, strain = curve

    result = analyze_dilatometry_curve(
        temperature, strain, method=method, detection="derivative"
    )

    assert result["method"] == method
    assert result["parameters"]["detection"] == "derivative"


def test_the_analyzer_carries_the_detector_too(curve):
    temperature, strain = curve

    analyzer = DilatometryAnalyzer(detection="derivative")
    result = analyzer.analyze(temperature, strain)

    assert analyzer.detection == "derivative"
    assert result["parameters"]["detection"] == "derivative"


def test_a_detector_can_be_called_on_its_own(curve):
    """The registry hands back something usable directly."""
    temperature, strain = curve
    context = DetectionContext(
        temperature=temperature,
        strain=strain,
        is_cooling=False,
        margin=0.2,
        window_length=25,
        polyorder=2,
        baseline_min_r2=0.99,
    )

    detector = get_detector("derivative")

    assert detector(context) == derivative_limits(context)


# --- The offset detector (#26) --------------------------------------------


def test_offset_is_registered_and_selectable(curve):
    temperature, strain = curve

    assert "offset" in available_detectors()
    limits = find_transformation_limits(temperature, strain, detection="offset")

    assert 0 < limits.start_idx < limits.end_idx < len(temperature) - 1


def test_offset_brackets_conservatively_on_a_straight_baseline(curve):
    """On a synthetic curve the offset sits inside the derivative's bracket.

    That is the method, not an error: it answers where the curve has departed
    measurably, not where the departure begins.
    """
    temperature, strain = curve

    derivative = find_transformation_limits(temperature, strain)
    offset = find_transformation_limits(temperature, strain, detection="offset")

    assert offset.start_idx > derivative.start_idx
    assert offset.end_idx < derivative.end_idx


def test_a_larger_offset_tightens_the_bracket(curve):
    """Over the useful range. Past roughly half the excursion the relation
    breaks down -- the threshold approaches the peak of the departure and the
    longest run over it stops being the transformation -- which is well past
    any offset worth using."""
    temperature, strain = curve

    wide = find_transformation_limits(
        temperature, strain, detection="offset", offset_fraction=0.02
    )
    narrow = find_transformation_limits(
        temperature, strain, detection="offset", offset_fraction=0.20
    )

    assert narrow.start_idx > wide.start_idx
    assert narrow.end_idx < wide.end_idx


def test_offset_warns_when_it_cannot_clear_the_baseline_curvature(real_curve):
    """The known failure of the method, measured and reported.

    The heating run's initial baseline fits a line at R2 = 0.999 and still
    departs from it by 1.3 % of the transformation excursion, so a 2 % offset
    is measuring the bow of the baseline. Left unwarned it reports the start
    at ~722 degC against a real ~839.
    """
    temperature, strain = real_curve

    with pytest.warns(UserWarning, match="not clear of the baseline"):
        limits = find_transformation_limits(
            temperature, strain, detection="offset", offset_fraction=0.02
        )

    assert temperature[limits.start_idx] < 800.0  # the drift the warning is about


def test_offset_rejects_a_fraction_outside_the_unit_interval(curve):
    temperature, strain = curve

    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="offset_fraction"):
            find_transformation_limits(
                temperature, strain, detection="offset", offset_fraction=bad
            )


# --- The second-derivative detector (#26) ---------------------------------


def test_second_derivative_finds_the_curvature_extrema(curve):
    """Checked against the closed form, not against an eyeballed number.

    For a logistic sigmoid of width w centred on T0, d2S/dT2 is extremal at
    T0 +/- w*ln(2 + sqrt(3)). With the fixture's 750 degC and w = 10 that is
    736.8 and 763.2.
    """
    temperature, strain = curve
    offset = TRANSFORMATION_WIDTH * np.log(2 + np.sqrt(3))

    limits = find_transformation_limits(
        temperature, strain, detection="second_derivative"
    )

    assert temperature[limits.start_idx] == pytest.approx(
        TRANSFORMATION_CENTRE - offset, abs=2.0
    )
    assert temperature[limits.end_idx] == pytest.approx(
        TRANSFORMATION_CENTRE + offset, abs=2.0
    )


def test_second_derivative_brackets_inside_the_derivative_detector(curve):
    """It marks maximum curvature, which is inside the feet of the
    transformation, not at them."""
    temperature, strain = curve

    derivative = find_transformation_limits(temperature, strain)
    curvature = find_transformation_limits(
        temperature, strain, detection="second_derivative"
    )

    assert derivative.start_idx < curvature.start_idx
    assert curvature.end_idx < derivative.end_idx


def test_second_derivative_warns_on_lopsided_curvature():
    """One extremum much smaller than the other is not a transformation."""
    temperature = np.linspace(600.0, 900.0, 1000)
    # A ramp that bends once and never comes back: one extremum, no pair.
    strain = (
        1e-5 * (temperature - 600.0) + 1e-8 * np.maximum(temperature - 750.0, 0.0) ** 2
    )

    with pytest.warns(UserWarning):
        find_transformation_limits(temperature, strain, detection="second_derivative")


def test_second_derivative_rejects_a_prominence_outside_the_unit_interval(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="prominence_fraction"):
        find_transformation_limits(
            temperature,
            strain,
            detection="second_derivative",
            prominence_fraction=1.5,
        )


@pytest.mark.parametrize(
    "detection", ["derivative", "offset", "second_derivative", "double_tangent"]
)
@pytest.mark.parametrize("method", ["lever", "tangent"])
def test_every_detector_crosses_with_every_method(curve, detection, method):
    temperature, strain = curve

    result = analyze_dilatometry_curve(
        temperature, strain, method=method, detection=detection
    )

    assert result["parameters"]["detection"] == detection
    assert result["start_temperature"] < result["end_temperature"]


# --- What the detectors do when there is nothing to find ------------------


@pytest.mark.parametrize("detection", ["offset", "second_derivative"])
@pytest.mark.parametrize("shape", ["constant", "straight"])
def test_a_curve_without_a_transformation_is_reported_as_such(detection, shape):
    """A run that never leaves its baseline has nothing to bracket.

    The comparison inside is against the scale of the strain, not against
    zero: on a perfectly straight run the departure and the curvature are
    floating-point residue, which is not zero, and without the scale both
    detectors returned invented limits without a word.
    """
    temperature = np.linspace(600.0, 900.0, 500)
    strain = (
        np.full_like(temperature, 0.5)
        if shape == "constant"
        else 1e-5 * (temperature - 600.0)
    )

    with pytest.warns(UserWarning, match="no transformation|numerical noise"):
        limits = find_transformation_limits(temperature, strain, detection=detection)

    assert limits == (75, 425)  # the search interval it falls back to


@pytest.mark.parametrize(
    "detection", ["derivative", "offset", "second_derivative", "double_tangent"]
)
def test_no_detector_cries_wolf_on_a_real_transformation(real_curve, detection):
    """The guard above must not fire on a curve that does transform."""
    temperature, strain = real_curve

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        find_transformation_limits(temperature, strain, detection=detection)

    assert not [
        w
        for w in raised
        if "no transformation" in str(w.message) or "numerical noise" in str(w.message)
    ]


def test_offset_needs_enough_points_in_each_baseline(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="fewer than 2 points"):
        find_transformation_limits(
            temperature, strain, detection="offset", margin=0.0005
        )


def test_second_derivative_needs_interior_points_to_look_at(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="too few interior points"):
        find_transformation_limits(
            temperature, strain, detection="second_derivative", margin=1.0
        )


# --- The double-tangent detector (#26) ------------------------------------


def test_double_tangent_is_registered_and_selectable(curve):
    temperature, strain = curve

    assert "double_tangent" in available_detectors()
    limits = find_transformation_limits(temperature, strain, detection="double_tangent")

    assert 0 < limits.start_idx < limits.end_idx < len(temperature) - 1


def test_double_tangent_reproduces_the_reading_off_the_plot(real_curve):
    """It recovers the number the reference value used to be.

    #103 settled that the 855-935 degC the tests used to assert was where the
    flattening becomes *visible* on a chart, not where the transformation
    begins -- the derivative detector puts the foot at 839. The double-tangent
    construction is that chart reading formalised: extend both baselines,
    draw the tangent through the steepest part, take the crossings. It lands
    on 860-926, which is the visual reading recovered from the data rather
    than from someone's eye.
    """
    temperature, strain = real_curve

    limits = find_transformation_limits(temperature, strain, detection="double_tangent")

    assert temperature[limits.start_idx] == pytest.approx(860.0, abs=5.0)
    assert temperature[limits.end_idx] == pytest.approx(926.0, abs=5.0)


def test_double_tangent_barely_moves_with_its_window(curve):
    """Its one parameter is not a threshold, and it shows.

    Doubling the tangent window moves the limits by a couple of kelvin; the
    offset detector moves by tens over a comparable change.
    """
    temperature, strain = curve

    narrow = find_transformation_limits(
        temperature, strain, detection="double_tangent", tangent_window_fraction=0.05
    )
    wide = find_transformation_limits(
        temperature, strain, detection="double_tangent", tangent_window_fraction=0.10
    )

    assert abs(temperature[narrow.start_idx] - temperature[wide.start_idx]) < 3.0
    assert abs(temperature[narrow.end_idx] - temperature[wide.end_idx]) < 3.0


def test_double_tangent_says_so_when_the_lines_never_cross():
    """A straight run has no steepest part, so the three lines are parallel."""
    temperature = np.linspace(600.0, 900.0, 500)
    strain = 1e-5 * (temperature - 600.0)

    with pytest.warns(UserWarning, match="parallel"):
        limits = find_transformation_limits(
            temperature, strain, detection="double_tangent"
        )

    assert limits == (75, 425)


def test_double_tangent_rejects_a_window_outside_the_unit_interval(curve):
    temperature, strain = curve

    for bad in (0.0, 1.0, 2.0):
        with pytest.raises(ValueError, match="tangent_window_fraction"):
            find_transformation_limits(
                temperature,
                strain,
                detection="double_tangent",
                tangent_window_fraction=bad,
            )


def test_double_tangent_needs_enough_points_in_each_baseline(curve):
    temperature, strain = curve

    with pytest.raises(ValueError, match="fewer than 2 points"):
        find_transformation_limits(
            temperature, strain, detection="double_tangent", margin=0.0005
        )


# --- The statistical detector (#26) ---------------------------------------


def noisy_sigmoid(scale=2e-5, n_points=1000):
    """The synthetic transformation with real noise on it, so that the
    residuals of a baseline are scatter rather than structure."""
    temperature = np.linspace(600.0, 900.0, n_points)
    sigmoid = 1 / (
        1 + np.exp(-(temperature - TRANSFORMATION_CENTRE) / TRANSFORMATION_WIDTH)
    )
    strain = 1e-5 * (temperature - 600.0) - 2e-3 * sigmoid
    return temperature, strain + np.random.normal(0.0, scale, n_points)


def test_statistical_is_registered_and_selectable():
    temperature, strain = noisy_sigmoid()

    assert "statistical" in available_detectors()
    limits = find_transformation_limits(temperature, strain, detection="statistical")

    assert 0 < limits.start_idx < limits.end_idx < len(temperature) - 1


def test_statistical_works_on_the_data_it_is_for():
    """A noisy run is where "is this more than noise" is the right question."""
    temperature, strain = noisy_sigmoid()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # nothing to warn about
        limits = find_transformation_limits(
            temperature, strain, detection="statistical"
        )

    assert temperature[limits.start_idx] == pytest.approx(720.0, abs=10.0)
    assert temperature[limits.end_idx] == pytest.approx(782.0, abs=10.0)


def test_statistical_detects_a_transformation_tail_on_a_clean_curve(curve):
    """Sensitive is not accurate, and the detector says which it is being.

    With no noise the baseline residual scatter is ~3e-8 while the tail of the
    sigmoid is already 2e-7 at 658 degC. That is a real six-sigma departure
    and 0.017 % of the excursion: correctly flagged, physically irrelevant.
    """
    temperature, strain = curve

    with pytest.warns(UserWarning, match="bowed, not scattered"):
        limits = find_transformation_limits(
            temperature, strain, detection="statistical"
        )

    # Far outside the 705-795 the other detectors bracket.
    assert temperature[limits.start_idx] < 700.0
    assert temperature[limits.end_idx] > 800.0


def test_statistical_reports_a_bowed_baseline(real_curve):
    """The assumption the method rests on, tested rather than assumed.

    It does not fix the drift that limits `offset`, which was the obvious
    reason to expect something of it: on this run it reports 717 degC where
    `offset` reports 739 and the foot is at 839.
    """
    temperature, strain = real_curve

    with pytest.warns(UserWarning, match="bowed, not scattered"):
        limits = find_transformation_limits(
            temperature, strain, detection="statistical"
        )

    assert temperature[limits.start_idx] < 800.0


def test_the_runs_test_separates_scatter_from_structure():
    """Near zero for noise, far below it for a bow."""
    temperature = np.linspace(600.0, 900.0, 400)
    mask = np.ones_like(temperature, dtype=bool)
    straight = 1e-5 * (temperature - 600.0)

    noisy = straight + np.random.normal(0.0, 1e-6, len(temperature))
    bowed = straight + 1e-9 * (temperature - 750.0) ** 2

    assert abs(_residual_structure(temperature, noisy, mask)) < 3.0
    assert _residual_structure(temperature, bowed, mask) < -3.0


def test_statistical_needs_three_points_for_a_variance():
    temperature, strain = noisy_sigmoid()

    with pytest.raises(ValueError, match="fewer than 3 points"):
        find_transformation_limits(
            temperature, strain, detection="statistical", margin=0.002
        )


def test_statistical_rejects_an_impossible_confidence():
    temperature, strain = noisy_sigmoid()

    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="confidence"):
            find_transformation_limits(
                temperature, strain, detection="statistical", confidence=bad
            )


def test_statistical_rejects_a_non_positive_structure_limit():
    temperature, strain = noisy_sigmoid()

    with pytest.raises(ValueError, match="max_residual_structure"):
        find_transformation_limits(
            temperature,
            strain,
            detection="statistical",
            max_residual_structure=0.0,
        )


def test_the_runs_test_stays_quiet_when_it_cannot_be_applied():
    """A perfect fit leaves no residual signs to count, and an inapplicable
    test must not raise a false alarm."""
    temperature = np.linspace(600.0, 900.0, 400)
    mask = np.ones_like(temperature, dtype=bool)

    assert _residual_structure(temperature, 1e-5 * (temperature - 600.0), mask) == 0.0


def test_statistical_says_so_when_nothing_clears_the_interval():
    """At a confidence that wide, no departure is surprising any more."""
    temperature, strain = noisy_sigmoid()

    with pytest.warns(UserWarning, match="never leaves"):
        limits = find_transformation_limits(
            temperature,
            strain,
            detection="statistical",
            confidence=0.9999999999999999,
        )

    assert limits == (150, 850)  # the search interval it falls back to


# --- Choosing the margin from the curve (#103.4, via #26) -----------------


def test_the_margin_has_a_dead_zone_this_is_not_hypothetical(real_curve):
    """The reason `margin="auto"` exists, asserted so it stays true.

    On the heating run the derivative detector gives 839-936 degC for margins
    from 0.18 to 0.25 and collapses to a 15 K bracket for 0.15 to 0.17. Both
    look equally reasonable from outside, and the baselines fit a line just as
    well inside the dead zone as outside it.
    """
    temperature, strain = real_curve

    with pytest.warns(UserWarning, match="wrong order"):
        collapsed = find_transformation_limits(temperature, strain, margin=0.16)
    healthy = find_transformation_limits(temperature, strain, margin=0.20)

    collapsed_width = abs(
        temperature[collapsed.end_idx] - temperature[collapsed.start_idx]
    )
    healthy_width = abs(temperature[healthy.end_idx] - temperature[healthy.start_idx])

    assert collapsed_width < 25.0
    assert healthy_width > 90.0


def test_auto_margin_steps_over_the_dead_zone(real_curve):
    temperature, strain = real_curve

    limits = find_transformation_limits(temperature, strain, margin="auto")

    assert temperature[limits.start_idx] == pytest.approx(839.0, abs=3.0)
    assert temperature[limits.end_idx] == pytest.approx(936.0, abs=3.0)


@pytest.mark.parametrize(
    "detection", ["derivative", "offset", "second_derivative", "double_tangent"]
)
def test_auto_margin_works_for_every_detector(curve, detection):
    temperature, strain = curve

    limits = find_transformation_limits(
        temperature, strain, margin="auto", detection=detection
    )

    assert 0 < limits.start_idx < limits.end_idx < len(temperature) - 1


def test_auto_margin_is_case_insensitive_and_rejects_anything_else(curve):
    temperature, strain = curve

    assert find_transformation_limits(
        temperature, strain, margin="AUTO"
    ) == find_transformation_limits(temperature, strain, margin="auto")

    with pytest.raises(ValueError, match="margin must be a fraction or 'auto'"):
        find_transformation_limits(temperature, strain, margin="widest")


def test_the_widest_plateau_wins_not_the_first():
    """Two plateaus, and the choice is the wider one."""
    limits_a = TransformationLimits(10, 90)
    limits_b = TransformationLimits(30, 70)
    answers = {
        0.10: limits_a,
        0.11: limits_a,
        0.12: limits_b,
        0.13: limits_b,
        0.14: limits_b,
    }

    margin, limits = stable_margin(
        lambda m: answers[round(m, 2)], n_total=100, candidates=sorted(answers)
    )

    assert limits == limits_b
    assert margin == pytest.approx(0.13)


def test_a_margin_that_does_not_fit_is_skipped_not_fatal():
    """A candidate that leaves too little data is one fewer candidate."""
    good = TransformationLimits(10, 90)

    def run(margin):
        if margin < 0.12:
            raise ValueError("too little data")
        return good

    margin, limits = stable_margin(
        run, n_total=100, candidates=[0.10, 0.11, 0.12, 0.13, 0.14]
    )

    assert limits == good
    assert margin >= 0.12


def test_no_usable_margin_at_all_is_an_error():
    def run(margin):
        raise ValueError("too little data")

    with pytest.raises(ValueError, match="No margin between"):
        stable_margin(run, n_total=100, candidates=[0.10, 0.20, 0.30])


def test_auto_margin_says_so_when_nothing_is_stable():
    """Every margin its own answer means there is no stable choice."""
    answers = {
        margin: TransformationLimits(10 + i * 20, 90 + i * 20)
        for i, margin in enumerate((0.10, 0.11, 0.12))
    }

    with pytest.warns(UserWarning, match="no stable choice"):
        stable_margin(
            lambda m: answers[round(m, 2)], n_total=100, candidates=sorted(answers)
        )


def test_exploring_does_not_leak_the_warnings_of_margins_it_discarded(real_curve):
    """The dead zone warns loudly, and `auto` tries it. The caller should only
    hear about the margin actually chosen."""
    temperature, strain = real_curve

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        find_transformation_limits(temperature, strain, margin="auto")

    assert not [w for w in raised if "wrong order" in str(w.message)]
