"""Tests for DSC baseline correction functionality."""

import numpy as np
import pytest
from numpy.typing import NDArray

from pkynetics.technique_analysis.dsc.baseline import BaselineCorrector, BaselineResult


# Test utilities
def generate_test_data(
    n_points: int = 1000, temp_range: tuple = (300, 500)
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate temperature and baseline data."""
    temperature = np.linspace(temp_range[0], temp_range[1], n_points)
    baseline = 0.001 * (temperature - temp_range[0])  # Linear baseline
    return temperature, baseline


def generate_test_peak(
    temperature: NDArray[np.float64], center: float, amplitude: float, width: float
) -> NDArray[np.float64]:
    """Generate a Gaussian peak."""
    return amplitude * np.exp(-(((temperature - center) / width) ** 2))


@pytest.fixture
def baseline_corrector():
    """Create BaselineCorrector instance."""
    return BaselineCorrector()


@pytest.fixture
def simple_data():
    """Generate simple test data with linear baseline."""
    temperature, baseline = generate_test_data()
    # Add single peak
    peak = generate_test_peak(temperature, 400, 1.0, 20.0)
    heat_flow = baseline + peak
    return {
        "temperature": temperature,
        "heat_flow": heat_flow,
        "baseline": baseline,
        "peak": peak,
    }


@pytest.fixture
def complex_data():
    """Generate complex test data with non-linear baseline."""
    temperature, _ = generate_test_data()
    # Create non-linear baseline
    baseline = 0.001 * (temperature - 300) + 0.00001 * (temperature - 300) ** 2
    # Add multiple peaks, leaving the first/last 10% free of events
    peak1 = generate_test_peak(temperature, 360, 0.8, 15.0)
    peak2 = generate_test_peak(temperature, 430, 1.2, 20.0)
    heat_flow = baseline + peak1 + peak2
    return {
        "temperature": temperature,
        "heat_flow": heat_flow,
        "baseline": baseline,
        "peaks": [peak1, peak2],
    }


# Basic functionality tests
def test_linear_baseline(baseline_corrector, simple_data):
    """Test linear baseline correction."""
    result = baseline_corrector.correct(
        simple_data["temperature"], simple_data["heat_flow"], method="linear"
    )

    assert isinstance(result, BaselineResult)
    np.testing.assert_allclose(result.baseline, simple_data["baseline"], atol=1e-2)
    assert "slope" in result.parameters
    assert "intercept" in result.parameters


def test_polynomial_baseline(baseline_corrector, complex_data):
    """Test polynomial baseline correction."""
    result = baseline_corrector.correct(
        complex_data["temperature"],
        complex_data["heat_flow"],
        method="polynomial",
        degree=2,
    )

    assert isinstance(result, BaselineResult)
    np.testing.assert_allclose(result.baseline, complex_data["baseline"], atol=1e-2)
    assert "coefficients" in result.parameters
    assert len(result.parameters["coefficients"]) == 3  # degree 2 + 1


def test_spline_baseline(baseline_corrector, complex_data):
    """Test spline baseline correction."""
    result = baseline_corrector.correct(
        complex_data["temperature"], complex_data["heat_flow"], method="spline"
    )

    assert isinstance(result, BaselineResult)
    assert "smoothing" in result.parameters
    assert "n_knots" in result.parameters
    assert result.quality_metrics["smoothness"] > 0


def test_asymmetric_baseline(baseline_corrector, complex_data):
    """Test asymmetric least squares baseline correction."""
    result = baseline_corrector.correct(
        complex_data["temperature"], complex_data["heat_flow"], method="asymmetric"
    )

    assert isinstance(result, BaselineResult)
    assert "lambda" in result.parameters
    assert "p" in result.parameters


def test_rubberband_baseline(baseline_corrector, simple_data):
    """Test rubberband baseline correction."""
    result = baseline_corrector.correct(
        simple_data["temperature"], simple_data["heat_flow"], method="rubberband"
    )

    assert isinstance(result, BaselineResult)
    assert "n_hull_points" in result.parameters
    # Baseline should be below or equal to data points (up to smoothing error)
    assert np.all(result.baseline <= simple_data["heat_flow"] + 1e-6)
    # ...and follow the linear baseline instead of cutting over the peak
    np.testing.assert_allclose(result.baseline, simple_data["baseline"], atol=1e-2)


def test_auto_baseline(baseline_corrector, complex_data):
    """Test automatic baseline method selection."""
    result = baseline_corrector.correct(
        complex_data["temperature"], complex_data["heat_flow"], method="auto"
    )

    assert isinstance(result, BaselineResult)
    assert result.method in ["linear", "polynomial"]
    assert "method" not in result.parameters
    # Within the overlap of the two peak tails in the valley between them
    np.testing.assert_allclose(result.baseline, complex_data["baseline"], atol=0.05)


# Region detection and optimization tests
def test_find_quiet_regions(baseline_corrector, simple_data):
    """Test quiet region detection."""
    regions = baseline_corrector._find_quiet_regions(
        simple_data["temperature"], simple_data["heat_flow"]
    )

    assert len(regions) > 0
    for start, end in regions:
        assert start < end
        assert start >= simple_data["temperature"][0]
        assert end <= simple_data["temperature"][-1]


def test_optimize_baseline(baseline_corrector, complex_data):
    """Test baseline optimization."""
    result = baseline_corrector.optimize_baseline(
        complex_data["temperature"], complex_data["heat_flow"]
    )

    assert isinstance(result, BaselineResult)
    assert result.regions is not None
    assert len(result.regions) > 0
    assert result.quality_metrics["baseline_rmse"] > 0


# Quality metrics tests
def test_quality_metrics(baseline_corrector, simple_data):
    """Test quality metrics calculation."""
    result = baseline_corrector.correct(
        simple_data["temperature"], simple_data["heat_flow"], method="linear"
    )

    metrics = result.quality_metrics
    assert "total_correction" in metrics
    assert "smoothness" in metrics
    assert metrics["total_correction"] > 0
    assert metrics["smoothness"] >= 0


def test_baseline_evaluation(baseline_corrector, simple_data):
    """Test baseline quality evaluation."""
    result = baseline_corrector.correct(
        simple_data["temperature"], simple_data["heat_flow"], method="linear"
    )

    score = baseline_corrector._evaluate_baseline_quality(result)
    assert isinstance(score, float)
    assert score >= 0


# Error handling tests
def test_invalid_method(baseline_corrector, simple_data):
    """Test handling of invalid correction method."""
    with pytest.raises(ValueError, match="Unknown baseline method"):
        baseline_corrector.correct(
            simple_data["temperature"],
            simple_data["heat_flow"],
            method="invalid_method",
        )


def test_mismatched_arrays(baseline_corrector):
    """Test handling of mismatched array lengths."""
    temperature = np.array([1, 2, 3])
    heat_flow = np.array([1, 2])

    with pytest.raises(ValueError, match="must have same length"):
        baseline_corrector.correct(temperature, heat_flow)


def test_insufficient_data(baseline_corrector):
    """Test handling of insufficient data points."""
    temperature = np.array([1, 2, 3])
    heat_flow = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="Data length must be at least"):
        baseline_corrector.correct(temperature, heat_flow)


# Integration tests
def test_full_baseline_workflow(baseline_corrector, complex_data):
    """Test complete baseline correction workflow."""
    # Find optimal regions
    regions = baseline_corrector._find_quiet_regions(
        complex_data["temperature"], complex_data["heat_flow"]
    )

    # Apply correction with regions
    result = baseline_corrector.correct(
        complex_data["temperature"],
        complex_data["heat_flow"],
        method="polynomial",
        regions=regions,
        degree=2,
    )

    # Verify results
    assert isinstance(result, BaselineResult)
    assert result.regions == regions
    assert len(result.baseline) == len(complex_data["heat_flow"])
    assert all(metric > 0 for metric in result.quality_metrics.values())


def test_baseline_reproducibility(baseline_corrector, simple_data):
    """Test reproducibility of baseline correction."""
    result1 = baseline_corrector.correct(
        simple_data["temperature"], simple_data["heat_flow"], method="linear"
    )

    result2 = baseline_corrector.correct(
        simple_data["temperature"], simple_data["heat_flow"], method="linear"
    )

    np.testing.assert_array_equal(result1.baseline, result2.baseline)
    assert result1.parameters == result2.parameters


def test_quiet_regions_avoid_peaks(baseline_corrector, complex_data):
    """Peak apexes are locally flat but must not be taken as baseline."""
    temperature = complex_data["temperature"]
    peak_signal = sum(complex_data["peaks"])
    regions = baseline_corrector._find_quiet_regions(
        temperature, complex_data["heat_flow"]
    )

    for start, end in regions:
        mask = (temperature >= start) & (temperature <= end)
        # Only the overlapping peak tails between the two peaks remain
        assert np.max(peak_signal[mask]) < 0.05


def test_spline_baseline_accuracy(baseline_corrector, complex_data):
    """Spline through the detected quiet regions recovers the baseline."""
    result = baseline_corrector.correct(
        complex_data["temperature"], complex_data["heat_flow"], method="spline"
    )
    np.testing.assert_allclose(result.baseline, complex_data["baseline"], atol=0.05)


def test_asymmetric_baseline_accuracy(baseline_corrector, simple_data):
    """ALS keeps the peak and does not depend on the number of points.

    ALS is approximate (~5% of the peak height here); the regression guarded
    against is the baseline absorbing the whole peak (error ~1.0).
    """
    result = baseline_corrector.correct(
        simple_data["temperature"], simple_data["heat_flow"], method="asymmetric"
    )
    np.testing.assert_allclose(result.baseline, simple_data["baseline"], atol=0.075)

    # Same curve sampled 20x more densely: same baseline, reasonable memory
    temperature = np.linspace(300, 500, 20000)
    baseline = 0.001 * (temperature - 300)
    heat_flow = baseline + generate_test_peak(temperature, 400, 1.0, 20.0)
    result = baseline_corrector.correct(temperature, heat_flow, method="asymmetric")
    np.testing.assert_allclose(result.baseline, baseline, atol=0.075)


# Stepped baselines (glass transition)
def stepped_signal(step_center: float = 400.0, step_height: float = 0.5):
    """Sloped baseline with a step, and a peak well after it."""
    temperature, baseline = generate_test_data(n_points=1000, temp_range=(300, 500))
    step = step_height / (1 + np.exp(-(temperature - step_center) / 3.0))
    peak = generate_test_peak(temperature, 460, 1.0, 6)
    return temperature, baseline + step + peak, baseline, step


def test_stepped_baseline_follows_the_step(baseline_corrector):
    """Each side is fitted on its own, so the step stays in the baseline."""
    temperature, heat_flow, baseline, step = stepped_signal()

    result = baseline_corrector.correct(
        temperature, heat_flow, method="linear", step_regions=[(385.0, 415.0)]
    )

    assert result.method.startswith("stepped")
    assert result.parameters["steps"] == [(385.0, 415.0)]
    assert len(result.parameters["segments"]) == 2

    # The baseline reproduces slope plus step, away from the peak
    expected = baseline + step
    before = temperature < 380
    after = (temperature > 420) & (temperature < 440)
    np.testing.assert_allclose(result.baseline[before], expected[before], atol=0.02)
    np.testing.assert_allclose(result.baseline[after], expected[after], atol=0.02)


def test_single_baseline_misses_the_step(baseline_corrector):
    """Without step_regions one fit crosses the step and leaves a residue."""
    temperature, heat_flow, baseline, step = stepped_signal()

    result = baseline_corrector.correct(temperature, heat_flow, method="linear")

    expected = baseline + step
    after = (temperature > 420) & (temperature < 440)
    residual = np.max(np.abs(result.baseline[after] - expected[after]))
    assert residual > 0.1  # half the step height, give or take


def test_stepped_baseline_keeps_the_peak(baseline_corrector):
    """The corrected signal is flat outside the peak and keeps its area."""
    temperature, heat_flow, _, _ = stepped_signal()

    result = baseline_corrector.correct(
        temperature, heat_flow, method="linear", step_regions=[(385.0, 415.0)]
    )

    quiet = (temperature > 420) & (temperature < 440)
    assert np.max(np.abs(result.corrected_data[quiet])) < 0.02
    peak_area = np.trapezoid(result.corrected_data, temperature)
    np.testing.assert_allclose(peak_area, 1.0 * 6 * np.sqrt(np.pi), rtol=0.1)


def test_step_regions_need_data_on_at_least_one_side(baseline_corrector):
    temperature, heat_flow, _, _ = stepped_signal()

    with pytest.raises(ValueError):
        baseline_corrector.correct(
            temperature,
            heat_flow,
            method="linear",
            step_regions=[(float(temperature[0]), float(temperature[-1]))],
        )


def test_stepped_baseline_with_explicit_regions(baseline_corrector):
    """Given regions are clipped to each segment; a segment without any
    falls back to the method's own choice."""
    temperature, heat_flow, baseline, step = stepped_signal()

    result = baseline_corrector.correct(
        temperature,
        heat_flow,
        method="linear",
        # The second region spans the step; the third lies beyond the peak
        regions=[(300.0, 340.0), (360.0, 430.0), (480.0, 500.0)],
        step_regions=[(385.0, 415.0)],
    )

    assert result.method.startswith("stepped")
    first, second = result.parameters["segments"]
    assert first["range"] == (300.0, 385.0)
    assert second["range"] == (415.0, 500.0)

    expected = baseline + step
    before = temperature < 380
    after = (temperature > 420) & (temperature < 440)
    np.testing.assert_allclose(result.baseline[before], expected[before], atol=0.02)
    np.testing.assert_allclose(result.baseline[after], expected[after], atol=0.02)


def test_stepped_baseline_ignores_regions_outside_a_segment(baseline_corrector):
    """A segment whose regions all fall outside it still gets a baseline."""
    temperature, heat_flow, baseline, step = stepped_signal()

    result = baseline_corrector.correct(
        temperature,
        heat_flow,
        method="linear",
        regions=[(300.0, 340.0)],  # only inside the first segment
        step_regions=[(385.0, 415.0)],
    )

    assert len(result.parameters["segments"]) == 2
    before = temperature < 380
    np.testing.assert_allclose(
        result.baseline[before], (baseline + step)[before], atol=0.02
    )


# Error paths of the fitting methods
def test_linear_baseline_needs_two_points(baseline_corrector, simple_data):
    temperature, heat_flow = simple_data["temperature"], simple_data["heat_flow"]
    empty_region = [(float(temperature[0]) - 50, float(temperature[0]) - 40)]

    with pytest.raises(ValueError, match="Not enough points"):
        baseline_corrector.correct(
            temperature, heat_flow, method="linear", regions=empty_region
        )


def test_polynomial_baseline_needs_more_points_than_the_degree(
    baseline_corrector, simple_data
):
    temperature, heat_flow = simple_data["temperature"], simple_data["heat_flow"]
    two_points = [(float(temperature[0]), float(temperature[2]))]

    with pytest.raises(ValueError, match="Not enough points"):
        baseline_corrector.correct(
            temperature, heat_flow, method="polynomial", regions=two_points, degree=3
        )


def test_auto_baseline_needs_points(baseline_corrector):
    """Fewer points than parameters: no candidate model can be fitted."""
    temperature = np.array([300.0, 310.0])
    heat_flow = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Not enough points"):
        baseline_corrector._auto_baseline(temperature, heat_flow, [(300.0, 310.0)])


def test_optimize_baseline_without_quiet_regions(baseline_corrector, monkeypatch):
    temperature, baseline = generate_test_data()
    monkeypatch.setattr(baseline_corrector, "_find_quiet_regions", lambda *a, **k: [])

    with pytest.raises(ValueError, match="No quiet regions"):
        baseline_corrector.optimize_baseline(temperature, baseline)
