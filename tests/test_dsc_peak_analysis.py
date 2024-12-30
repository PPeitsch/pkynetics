"""Tests for DSC peak analysis functionality."""

import numpy as np
import pytest
from numpy.typing import NDArray

from pkynetics.technique_analysis.dsc.peak_analysis import PeakAnalyzer


# Test utilities
def generate_gaussian_peak(
    temperature: NDArray[np.float64], center: float, amplitude: float, width: float
) -> NDArray[np.float64]:
    """Generate a Gaussian peak for testing."""
    return amplitude * np.exp(-(((temperature - center) / width) ** 2))


def generate_multiple_peaks(
    temperature: NDArray[np.float64],
    centers: list[float],
    amplitudes: list[float],
    widths: list[float],
) -> NDArray[np.float64]:
    """Generate multiple overlapping peaks."""
    heat_flow = np.zeros_like(temperature)
    for center, amplitude, width in zip(centers, amplitudes, widths):
        heat_flow += generate_gaussian_peak(temperature, center, amplitude, width)
    return heat_flow


# Fixtures
@pytest.fixture
def peak_analyzer():
    """Create PeakAnalyzer instance."""
    return PeakAnalyzer()


@pytest.fixture
def simple_peak_data():
    """Generate single peak data."""
    temperature = np.linspace(300, 500, 1000)
    center = 400
    amplitude = 1.0
    width = 20.0
    heat_flow = generate_gaussian_peak(temperature, center, amplitude, width)
    peak_idx = np.argmax(heat_flow)
    return {
        "temperature": temperature,
        "heat_flow": heat_flow,
        "peak_idx": peak_idx,
        "expected_center": center,
        "expected_onset": center - width * np.sqrt(np.log(2)),
        "expected_endset": center + width * np.sqrt(np.log(2)),
        "expected_width": 2 * width * np.sqrt(np.log(2)),
        "expected_height": amplitude,
    }


@pytest.fixture
def overlapping_peaks_data():
    """Generate overlapping peaks data."""
    temperature = np.linspace(300, 500, 1000)
    centers = [380, 420]
    amplitudes = [1.0, 0.8]
    widths = [15.0, 18.0]
    heat_flow = generate_multiple_peaks(temperature, centers, amplitudes, widths)
    return {
        "temperature": temperature,
        "heat_flow": heat_flow,
        "n_peaks": len(centers),
        "expected_centers": centers,
        "expected_amplitudes": amplitudes,
        "expected_widths": widths,
    }


@pytest.fixture
def noisy_peak_data():
    """Generate peak data with noise."""
    temperature = np.linspace(300, 500, 1000)
    heat_flow = generate_gaussian_peak(temperature, 400, 1.0, 20.0)
    noise = np.random.normal(0, 0.05, size=len(temperature))
    noisy_heat_flow = heat_flow + noise
    peak_idx = np.argmax(noisy_heat_flow)
    return {
        "temperature": temperature,
        "heat_flow": noisy_heat_flow,
        "peak_idx": peak_idx,
        "original_heat_flow": heat_flow,
    }


# Tests for basic peak detection
def test_find_peaks_simple(peak_analyzer, simple_peak_data):
    """Test peak detection for a single clean peak."""
    peaks = peak_analyzer.find_peaks(
        simple_peak_data["temperature"], simple_peak_data["heat_flow"]
    )

    assert len(peaks) == 1
    peak = peaks[0]

    np.testing.assert_allclose(
        peak.peak_temperature, simple_peak_data["expected_center"], rtol=1e-2
    )
    np.testing.assert_allclose(
        peak.peak_height, simple_peak_data["expected_height"], rtol=1e-2
    )


def test_find_peaks_with_noise(peak_analyzer, noisy_peak_data):
    """Test peak detection with noisy data."""
    peaks = peak_analyzer.find_peaks(
        noisy_peak_data["temperature"], noisy_peak_data["heat_flow"]
    )

    assert len(peaks) == 1
    # Peak should be detected within 5K of the true peak
    assert abs(peaks[0].peak_temperature - 400) < 5


def test_find_peaks_overlapping(peak_analyzer, overlapping_peaks_data):
    """Test detection of overlapping peaks."""
    peaks = peak_analyzer.find_peaks(
        overlapping_peaks_data["temperature"], overlapping_peaks_data["heat_flow"]
    )

    assert len(peaks) == overlapping_peaks_data["n_peaks"]
    peak_temps = [p.peak_temperature for p in peaks]
    np.testing.assert_allclose(
        sorted(peak_temps),
        sorted(overlapping_peaks_data["expected_centers"]),
        rtol=1e-2,
    )


# Tests for peak characteristics
def test_onset_calculation(peak_analyzer, simple_peak_data):
    """Test onset temperature calculation."""
    onset_temp = peak_analyzer._calculate_onset(
        simple_peak_data["temperature"],
        simple_peak_data["heat_flow"],
        simple_peak_data["peak_idx"],
    )

    np.testing.assert_allclose(
        onset_temp, simple_peak_data["expected_onset"], rtol=1e-2
    )


def test_endset_calculation(peak_analyzer, simple_peak_data):
    """Test endset temperature calculation."""
    endset_temp = peak_analyzer._calculate_endset(
        simple_peak_data["temperature"],
        simple_peak_data["heat_flow"],
        simple_peak_data["peak_idx"],
    )

    np.testing.assert_allclose(
        endset_temp, simple_peak_data["expected_endset"], rtol=1e-2
    )


def test_peak_width_calculation(peak_analyzer, simple_peak_data):
    """Test peak width calculation."""
    width = peak_analyzer._calculate_peak_width(
        simple_peak_data["temperature"],
        simple_peak_data["heat_flow"],
        simple_peak_data["peak_idx"],
    )

    np.testing.assert_allclose(width, simple_peak_data["expected_width"], rtol=1e-2)


# Tests for peak deconvolution
def test_peak_deconvolution(peak_analyzer, overlapping_peaks_data):
    """Test deconvolution of overlapping peaks."""
    peak_params, fitted_curve = peak_analyzer.deconvolute_peaks(
        overlapping_peaks_data["temperature"],
        overlapping_peaks_data["heat_flow"],
        overlapping_peaks_data["n_peaks"],
    )

    assert len(peak_params) == overlapping_peaks_data["n_peaks"]

    # Check centers of deconvoluted peaks
    centers = sorted([p["center"] for p in peak_params])
    np.testing.assert_allclose(
        centers, sorted(overlapping_peaks_data["expected_centers"]), rtol=1e-2
    )

    # Check that fitted curve approximates original data
    rmse = np.sqrt(np.mean((fitted_curve - overlapping_peaks_data["heat_flow"]) ** 2))
    assert rmse < 0.1


# Tests for error handling
def test_invalid_peak_index(peak_analyzer, simple_peak_data):
    """Test handling of invalid peak index."""
    with pytest.raises(IndexError):
        peak_analyzer._calculate_onset(
            simple_peak_data["temperature"],
            simple_peak_data["heat_flow"],
            len(simple_peak_data["temperature"]),
        )


def test_empty_data(peak_analyzer):
    """Test handling of empty data."""
    with pytest.raises(ValueError):
        peak_analyzer.find_peaks(np.array([]), np.array([]))


def test_mismatched_arrays(peak_analyzer):
    """Test handling of mismatched array lengths."""
    with pytest.raises(ValueError):
        peak_analyzer.find_peaks(np.array([1, 2, 3]), np.array([1, 2]))


# Integration tests
def test_full_peak_analysis(peak_analyzer, simple_peak_data):
    """Test complete peak analysis workflow."""
    # Find peaks
    peaks = peak_analyzer.find_peaks(
        simple_peak_data["temperature"], simple_peak_data["heat_flow"]
    )

    assert len(peaks) == 1
    peak = peaks[0]

    # Check all peak characteristics
    assert abs(peak.peak_temperature - simple_peak_data["expected_center"]) < 1
    assert abs(peak.onset_temperature - simple_peak_data["expected_onset"]) < 5
    assert abs(peak.endset_temperature - simple_peak_data["expected_endset"]) < 5
    assert abs(peak.peak_height - simple_peak_data["expected_height"]) < 0.1
    assert peak.peak_area > 0
    assert peak.enthalpy > 0


def test_baseline_correction_impact(peak_analyzer, simple_peak_data):
    """Test peak analysis with baseline correction."""
    # Add sloped baseline
    slope = 0.001
    baseline = slope * simple_peak_data["temperature"]
    heat_flow_with_baseline = simple_peak_data["heat_flow"] + baseline

    # Analyze with and without baseline correction
    peaks_no_baseline = peak_analyzer.find_peaks(
        simple_peak_data["temperature"], heat_flow_with_baseline
    )

    peaks_with_baseline = peak_analyzer.find_peaks(
        simple_peak_data["temperature"], heat_flow_with_baseline, baseline=baseline
    )

    # Peak temperature should be the same
    assert (
        abs(
            peaks_no_baseline[0].peak_temperature
            - peaks_with_baseline[0].peak_temperature
        )
        < 1
    )

    # Areas should be different
    assert peaks_no_baseline[0].peak_area != peaks_with_baseline[0].peak_area
