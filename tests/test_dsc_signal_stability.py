"""Tests for DSC signal stability detection."""

import numpy as np
import pytest

from pkynetics.technique_analysis.dsc import SignalStabilityDetector, StabilityMethod


def stepped_signal(noise=0.01, correlated=False, seed=0):
    """Five plateaus of 1200 points joined by 300-point ramps."""
    parts = []
    for level in range(5):
        parts.append(np.full(1200, float(level)))
        if level < 4:
            parts.append(np.linspace(level, level + 1, 300))
    clean = np.concatenate(parts)
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, len(clean))
    if correlated:
        # Instrument-like low-pass filtered noise, unit variance
        kernel = np.exp(-np.arange(10) / 2.0)
        white = np.convolve(white, kernel / np.sqrt(np.sum(kernel**2)), "same")
    return clean, clean + noise * white


PLATEAUS = [(0, 1200), (1500, 2700), (3000, 4200), (4500, 5700), (6000, 7200)]


def assert_regions_close(found, expected, tolerance):
    assert len(found) == len(expected)
    for (a, b), (c, d) in zip(found, expected):
        assert abs(a - c) <= tolerance and abs(b - d) <= tolerance


@pytest.mark.parametrize("correlated", [False, True])
def test_plateaus_found(correlated):
    _, signal = stepped_signal(correlated=correlated)
    regions = SignalStabilityDetector().find_stable_regions(signal)
    assert_regions_close(regions, PLATEAUS, tolerance=30)


def test_clean_signal_exact():
    clean, _ = stepped_signal()
    regions = SignalStabilityDetector().find_stable_regions(clean)
    assert_regions_close(regions, PLATEAUS, tolerance=1)


def test_linear_fit_accepts_ramps():
    """LINEAR_FIT accepts linear drifts that STATISTICAL rejects."""
    x = np.arange(3000, dtype=float)
    signal = 0.01 * x + np.random.default_rng(1).normal(0, 0.05, len(x))

    assert (
        SignalStabilityDetector(method="statistical").find_stable_regions(signal) == []
    )
    assert SignalStabilityDetector(
        method=StabilityMethod.LINEAR_FIT
    ).find_stable_regions(signal) == [(0, 3000)]


def test_noise_level():
    rng = np.random.default_rng(2)
    trend = np.linspace(0, 100, 20000)
    white = rng.normal(0, 0.1, 20000)
    np.testing.assert_allclose(
        SignalStabilityDetector.noise_level(trend + white), 0.1, rtol=0.05
    )
    # Correlated noise: neighbouring differences would underestimate it
    _, correlated = stepped_signal(noise=0.1, correlated=True)
    np.testing.assert_allclose(
        SignalStabilityDetector.noise_level(correlated), 0.1, rtol=0.1
    )


def test_min_points_and_short_signals():
    _, signal = stepped_signal()
    detector = SignalStabilityDetector(min_points=1500)
    assert detector.find_stable_regions(signal) == []
    assert SignalStabilityDetector().find_stable_regions(signal[:40]) == []


def test_x_values_and_validation():
    _, signal = stepped_signal()
    detector = SignalStabilityDetector()
    time = np.arange(len(signal)) * 0.5
    assert detector.find_stable_regions(signal, time) == detector.find_stable_regions(
        signal
    )
    with pytest.raises(ValueError, match="same length"):
        detector.find_stable_regions(signal, time[:-1])
    with pytest.raises(ValueError):
        SignalStabilityDetector(method="wavelet")
    with pytest.raises(ValueError, match="Window"):
        SignalStabilityDetector(window_size=2)


def test_evaluate_stability():
    clean, signal = stepped_signal()
    detector = SignalStabilityDetector()

    plateau = detector.evaluate_stability(signal, (100, 1100))
    assert abs(plateau["mean"]) < 0.01
    assert plateau["relative_std"] < 1.5
    assert plateau["length"] == 1000

    ramp = detector.evaluate_stability(signal, (1250, 1450))
    assert ramp["relative_std"] > 5
    assert ramp["relative_residual"] < 1.5  # linear within the noise
    np.testing.assert_allclose(ramp["slope"], 1 / 299, rtol=0.05)

    flat = detector.evaluate_stability(np.zeros(100), (0, 100))
    assert np.isfinite(flat["relative_std"])

    with pytest.raises(ValueError, match="Region"):
        detector.evaluate_stability(signal, (10, 11))
