"""Tests for the shared smoothing module."""

import numpy as np
import pytest
from scipy.signal import savgol_filter

from pkynetics.data_preprocessing import available_smoothing_methods, smooth_data
from pkynetics.technique_analysis.dsc import SignalProcessor


@pytest.fixture
def noisy_sine():
    """A sine with noise, and the clean sine underneath."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 500)
    clean = np.sin(x) + 5.0
    return clean + rng.normal(0, 0.1, x.size), clean


def test_available_methods():
    assert available_smoothing_methods() == ["lowess", "moving_average", "savgol"]


def test_unknown_method_lists_the_valid_ones():
    with pytest.raises(ValueError, match="moving_average"):
        smooth_data(np.random.rand(100), method="loess")


def test_savgol_default_is_unchanged(noisy_sine):
    """Without method=, smooth_data returns what it returned before #47."""
    data, _ = noisy_sine
    old_window = min(len(data) - 2, int(len(data) * 0.05) // 2 * 2 + 1)
    expected = savgol_filter(data, old_window, 3)
    np.testing.assert_array_equal(smooth_data(data), expected)


def test_fixed_window_matches_freeman_carroll_filter(noisy_sine):
    """Freeman-Carroll's old private filter, window 21, gives the same output."""
    data, _ = noisy_sine
    np.testing.assert_array_equal(
        smooth_data(data, window_length=21), savgol_filter(data, 21, 3)
    )


@pytest.mark.parametrize("method", ["savgol", "moving_average", "lowess"])
def test_every_method_reduces_noise(noisy_sine, method):
    data, clean = noisy_sine
    smoothed = smooth_data(data, window_length=21, method=method)
    assert smoothed.shape == data.shape
    assert np.std(smoothed - clean) < 0.5 * np.std(data - clean)


def test_moving_average_leaves_a_line_with_offset_unchanged():
    """Zero padding pulled the ends of an offset signal toward zero."""
    line = 1000.0 + 2.0 * np.arange(200, dtype=float)
    np.testing.assert_allclose(
        smooth_data(line, window_length=21, method="moving_average"), line
    )


def test_moving_average_passes_the_end_points_through():
    data = np.random.default_rng(1).normal(size=100)
    smoothed = smooth_data(data, window_length=11, method="moving_average")
    assert smoothed[0] == pytest.approx(data[0])
    assert smoothed[-1] == pytest.approx(data[-1])
    # One point in, the window is three wide.
    assert smoothed[1] == pytest.approx(np.mean(data[:3]))


def test_lowess_uses_the_abscissa():
    """Where the sampling changes rate, fitting against x beats the index.

    The gain is local and modest: lowess picks its neighbours by count, so on a
    monotone x the neighbourhood is nearly the same either way, and only the
    weights and the fitted line change. Here, a ramp that triples its rate, it
    is about a quarter less error next to the change.
    """
    rng = np.random.default_rng(2)
    x = np.concatenate([np.linspace(0, 5, 300), np.linspace(5.05, 15, 100)])
    clean = np.sin(x)
    data = clean + rng.normal(0, 0.02, x.size)
    near_change = slice(260, 340)

    against_x = smooth_data(data, window_length=21, method="lowess", x=x)
    against_index = smooth_data(data, window_length=21, method="lowess")

    error_x = np.std((against_x - clean)[near_change])
    error_index = np.std((against_index - clean)[near_change])
    assert error_x < 0.85 * error_index


@pytest.mark.parametrize("method", ["savgol", "moving_average"])
def test_x_is_rejected_where_it_would_be_ignored(method):
    data = np.random.rand(100)
    with pytest.raises(ValueError, match="only used by 'lowess'"):
        smooth_data(data, method=method, x=np.arange(100.0))


def test_x_must_match_the_data():
    with pytest.raises(ValueError, match="as long as data"):
        smooth_data(np.random.rand(100), method="lowess", x=np.arange(50.0))


@pytest.mark.parametrize("method", ["savgol", "moving_average", "lowess"])
def test_window_must_be_shorter_than_the_data(method):
    with pytest.raises(ValueError, match="less than data length"):
        smooth_data(np.random.rand(20), window_length=21, method=method)


def test_signal_processor_delegates():
    """SignalProcessor gets the fixed moving average and the abscissa for lowess."""
    processor = SignalProcessor()
    line = 50.0 + 0.1 * np.arange(300, dtype=float)
    np.testing.assert_allclose(
        processor.smooth_signal(line, method="moving_average"), line
    )

    x = np.linspace(300, 500, 300)
    smoothed = processor.smooth_signal(line, method="lowess", x=x)
    np.testing.assert_allclose(smoothed, line, atol=1e-8)
