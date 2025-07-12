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
    # Add multiple peaks
    peak1 = generate_test_peak(temperature, 350, 0.8, 15.0)
    peak2 = generate_test_peak(temperature, 450, 1.2, 25.0)
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
    np.testing.assert_allclose(result.baseline, simple_data["baseline"], rtol=5e-2)
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
    np.testing.assert_allclose(result.baseline, complex_data["baseline"], rtol=5e-2)
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
    # Baseline should be below or equal to data points
    assert np.all(result.baseline <= simple_data["heat_flow"])


def test_auto_baseline(baseline_corrector, complex_data):
    """Test automatic baseline method selection."""
    result = baseline_corrector.correct(
        complex_data["temperature"], complex_data["heat_flow"], method="auto"
    )

    assert isinstance(result, BaselineResult)
    assert result.method in ["linear", "polynomial", "spline", "asymmetric"]


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
