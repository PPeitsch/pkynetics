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
        # Extrapolated onset/endset (ISO 11357-1): the tangent at the
        # inflection point (center -/+ width/sqrt(2)) meets the baseline at
        # center -/+ sqrt(2) * width
        "expected_onset": center - np.sqrt(2) * width,
        "expected_endset": center + np.sqrt(2) * width,
        # Full width at half maximum
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
    noise = np.random.default_rng(0).normal(0, 0.05, size=len(temperature))
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


def test_find_peaks_noise_does_not_create_spurious_peaks(peak_analyzer):
    """No realization of 5 % noise on a single peak yields a second peak.

    The absolute prominence/height defaults (0.1 and 0.05 mW) are below what
    a noise fluctuation reaches on this signal, so detection used to depend
    on the draw: about 1 % of seeds found a second, spurious peak. The floors
    now scale with the noise left after smoothing.
    """
    temperature = np.linspace(300, 500, 1000)
    clean = generate_gaussian_peak(temperature, 400, 1.0, 20.0)

    counts = []
    for seed in range(200):
        noise = np.random.default_rng(seed).normal(0, 0.05, size=len(temperature))
        peaks = peak_analyzer.find_peaks(temperature, clean + noise)
        counts.append(len(peaks))

    assert set(counts) == {1}, f"spurious detections: {sorted(set(counts))}"


def test_find_peaks_still_detects_a_peak_just_above_the_noise(peak_analyzer):
    """Raising the floor with the noise does not swallow genuine small peaks."""
    temperature = np.linspace(300, 500, 1000)
    # Modest peak: it clears the absolute 0.1 mW prominence floor, and the
    # noise-scaled floor has to leave it alone too
    heat_flow = generate_gaussian_peak(temperature, 400, 0.3, 20.0)
    noise = np.random.default_rng(1).normal(0, 0.005, size=len(temperature))

    peaks = peak_analyzer.find_peaks(temperature, heat_flow + noise)

    assert len(peaks) == 1
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

    np.testing.assert_allclose(onset_temp, simple_peak_data["expected_onset"], atol=0.1)


def test_endset_calculation(peak_analyzer, simple_peak_data):
    """Test endset temperature calculation."""
    endset_temp = peak_analyzer._calculate_endset(
        simple_peak_data["temperature"],
        simple_peak_data["heat_flow"],
        simple_peak_data["peak_idx"],
    )

    np.testing.assert_allclose(
        endset_temp, simple_peak_data["expected_endset"], atol=0.1
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
    assert rmse < 0.01

    # Recovered amplitudes and widths
    by_center = sorted(peak_params, key=lambda p: p["center"])
    np.testing.assert_allclose(
        [p["amplitude"] for p in by_center],
        overlapping_peaks_data["expected_amplitudes"],
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        [p["width"] for p in by_center],
        overlapping_peaks_data["expected_widths"],
        rtol=1e-2,
    )


def test_peak_deconvolution_shoulder(peak_analyzer):
    """A shoulder without its own maximum is still resolved."""
    temperature = np.linspace(300, 500, 1000)
    heat_flow = generate_multiple_peaks(temperature, [390, 410], [1.0, 0.6], [12, 12])

    peak_params, fitted_curve = peak_analyzer.deconvolute_peaks(
        temperature, heat_flow, 2
    )

    assert len(peak_params) == 2
    centers = sorted(p["center"] for p in peak_params)
    np.testing.assert_allclose(centers, [390, 410], atol=1.0)
    assert np.sqrt(np.mean((fitted_curve - heat_flow) ** 2)) < 0.01


# D18: deconvolution of peaks pointing either way
def test_peak_deconvolution_of_downward_peaks(peak_analyzer):
    """Endothermic (downward) peaks deconvolute like upward ones.

    The fit constrains amplitudes to be non-negative, so a curve whose
    peaks point down used to come back with ~1e-11 amplitudes, an
    arbitrary centre and a flat fitted curve — a silent failure that reads
    as a converged result. The peaks are now oriented before fitting and
    the amplitudes reported with the sign of the input.
    """
    temperature = np.linspace(300, 500, 1000)
    heat_flow = -generate_multiple_peaks(temperature, [380, 430], [1.0, 0.7], [12, 15])

    peak_params, fitted_curve = peak_analyzer.deconvolute_peaks(
        temperature, heat_flow, 2
    )

    assert len(peak_params) == 2
    by_centre = sorted(peak_params, key=lambda p: p["center"])
    np.testing.assert_allclose([p["center"] for p in by_centre], [380, 430], atol=1.0)
    np.testing.assert_allclose(
        [p["amplitude"] for p in by_centre], [-1.0, -0.7], atol=0.02
    )
    # Areas carry the sign too, so they can be summed with upward peaks
    assert all(p["area"] < 0 for p in peak_params)
    assert np.sqrt(np.mean((fitted_curve - heat_flow) ** 2)) < 0.01


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_peak_deconvolution_with_an_offset_baseline(peak_analyzer, sign):
    """A constant offset does not leak into the amplitudes.

    The model is a sum of Gaussians with no constant term, so the flat
    level of the curve has to be the zero they are measured from.
    """
    temperature = np.linspace(300, 500, 1000)
    heat_flow = 5.0 + sign * generate_multiple_peaks(temperature, [400], [1.0], [15])

    peak_params, fitted_curve = peak_analyzer.deconvolute_peaks(
        temperature, heat_flow, 1
    )

    assert len(peak_params) == 1
    assert peak_params[0]["amplitude"] == pytest.approx(sign * 1.0, abs=0.02)
    assert peak_params[0]["center"] == pytest.approx(400.0, abs=1.0)
    # The fitted curve comes back on the original level, not around zero
    assert np.sqrt(np.mean((fitted_curve - heat_flow) ** 2)) < 0.01


def test_peak_deconvolution_rejects_impossible_requests(peak_analyzer):
    """Fewer points than parameters, or a non-positive peak count."""
    temperature = np.linspace(300, 500, 1000)
    heat_flow = generate_multiple_peaks(temperature, [400], [1.0], [15])

    with pytest.raises(ValueError, match="at least 1"):
        peak_analyzer.deconvolute_peaks(temperature, heat_flow, 0)

    with pytest.raises(ValueError, match="at least 15 points"):
        peak_analyzer.deconvolute_peaks(temperature[:10], heat_flow[:10], 5)


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
    assert abs(peak.onset_temperature - simple_peak_data["expected_onset"]) < 0.5
    assert abs(peak.endset_temperature - simple_peak_data["expected_endset"]) < 0.5
    assert abs(peak.peak_height - simple_peak_data["expected_height"]) < 0.1
    # Gaussian area: amplitude * width * sqrt(pi) = 35.45 mW*K
    np.testing.assert_allclose(peak.peak_area, 20.0 * np.sqrt(np.pi), rtol=1e-3)
    # Enthalpy needs heating rate and mass
    assert np.isnan(peak.enthalpy)


def test_enthalpy_units(peak_analyzer, simple_peak_data):
    """Enthalpy in J/g: integral(q dT) / heating rate / mass."""
    peaks = peak_analyzer.find_peaks(
        simple_peak_data["temperature"],
        simple_peak_data["heat_flow"],
        heating_rate=10.0,  # K/min
        sample_mass=5.0,  # mg
    )

    # 35.45 mW*K / (10/60 K/s) = 212.7 mJ; / 5 mg = 42.5 J/g
    expected = 20.0 * np.sqrt(np.pi) / (10.0 / 60) / 5.0
    np.testing.assert_allclose(peaks[0].enthalpy, expected, rtol=1e-3)


def test_onset_with_sloped_baseline(peak_analyzer, simple_peak_data):
    """Onset/endset are measured against the given baseline."""
    temperature = simple_peak_data["temperature"]
    baseline = 0.002 * (temperature - 300) + 0.1
    heat_flow = simple_peak_data["heat_flow"] + baseline
    peak_idx = int(np.argmax(simple_peak_data["heat_flow"]))

    onset = peak_analyzer._calculate_onset(temperature, heat_flow, peak_idx, baseline)
    endset = peak_analyzer._calculate_endset(temperature, heat_flow, peak_idx, baseline)

    np.testing.assert_allclose(onset, simple_peak_data["expected_onset"], atol=0.1)
    np.testing.assert_allclose(endset, simple_peak_data["expected_endset"], atol=0.1)


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
