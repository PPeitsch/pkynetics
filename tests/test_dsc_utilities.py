"""Tests for DSC analysis utilities module."""

import numpy as np
import pytest
from numpy.typing import NDArray

from pkynetics.technique_analysis.dsc.utilities import (
    DataValidator,
    DSCUnits,
    SignalProcessor,
    UnitConverter,
)


# Test data generation utilities
def generate_noisy_signal(
    n_points: int = 1000, noise_level: float = 0.1
) -> NDArray[np.float64]:
    """Generate noisy test signal."""
    x = np.linspace(0, 10, n_points)
    clean_signal = np.sin(x) + 0.5 * np.sin(2 * x)
    noise = np.random.normal(0, noise_level, n_points)
    return x, clean_signal + noise


def generate_temperature_ramp(
    start_temp: float = 300, end_temp: float = 500, n_points: int = 1000
) -> NDArray[np.float64]:
    """Generate temperature ramp data."""
    return np.linspace(start_temp, end_temp, n_points)


@pytest.fixture
def signal_processor():
    """Create SignalProcessor instance."""
    return SignalProcessor()


@pytest.fixture
def noisy_data():
    """Generate noisy test data."""
    x, signal = generate_noisy_signal()
    return {"x": x, "signal": signal, "clean_signal": np.sin(x) + 0.5 * np.sin(2 * x)}


@pytest.fixture
def temperature_data():
    """Generate temperature data."""
    return generate_temperature_ramp()


# Signal Processing Tests
def test_smooth_signal_savgol(signal_processor, noisy_data):
    """Test Savitzky-Golay smoothing."""
    smoothed = signal_processor.smooth_signal(noisy_data["signal"], method="savgol")

    # Check shape preservation
    assert len(smoothed) == len(noisy_data["signal"])

    # Check noise reduction
    original_noise = np.std(noisy_data["signal"] - noisy_data["clean_signal"])
    smoothed_noise = np.std(smoothed - noisy_data["clean_signal"])
    assert smoothed_noise < original_noise


def test_smooth_signal_moving_average(signal_processor, noisy_data):
    """Test moving average smoothing."""
    smoothed = signal_processor.smooth_signal(
        noisy_data["signal"], method="moving_average"
    )

    assert len(smoothed) == len(noisy_data["signal"])
    assert np.std(np.diff(smoothed)) < np.std(np.diff(noisy_data["signal"]))


def test_smooth_signal_lowess(signal_processor, noisy_data):
    """Test LOWESS smoothing."""
    smoothed = signal_processor.smooth_signal(noisy_data["signal"], method="lowess")

    assert len(smoothed) == len(noisy_data["signal"])
    assert np.std(np.diff(smoothed)) < np.std(np.diff(noisy_data["signal"]))


def test_remove_outliers(signal_processor, noisy_data):
    """Test outlier removal."""
    # Add artificial outliers
    signal_with_outliers = noisy_data["signal"].copy()
    outlier_indices = np.random.choice(len(signal_with_outliers), 10)
    signal_with_outliers[outlier_indices] += 10.0

    cleaned = signal_processor.remove_outliers(signal_with_outliers)

    assert len(cleaned) == len(signal_with_outliers)
    assert np.max(np.abs(cleaned)) < np.max(np.abs(signal_with_outliers))


def test_filter_signal(signal_processor, noisy_data):
    """Test frequency domain filtering."""
    filtered = signal_processor.filter_signal(
        noisy_data["signal"],
        sampling_rate=100.0,  # Hz
        cutoff_freq=10.0,  # Hz
        filter_type="lowpass",
    )

    assert len(filtered) == len(noisy_data["signal"])
    assert np.std(filtered) < np.std(noisy_data["signal"])


def test_calculate_derivatives(signal_processor, noisy_data):
    """Test derivative calculations."""
    derivatives = signal_processor.calculate_derivatives(
        noisy_data["x"], noisy_data["signal"], smooth=True
    )

    assert "first_derivative" in derivatives
    assert "second_derivative" in derivatives
    assert len(derivatives["first_derivative"]) == len(noisy_data["signal"])


def test_calculate_noise_level(signal_processor, noisy_data):
    """Test noise level estimation."""
    noise_level = signal_processor.calculate_noise_level(noisy_data["signal"])

    assert isinstance(noise_level, float)
    assert noise_level > 0
    assert noise_level < np.std(noisy_data["signal"])


# Unit Conversion Tests
def test_temperature_conversion():
    """Test temperature unit conversions."""
    # Celsius to Kelvin
    assert np.isclose(
        UnitConverter.convert_temperature(0.0, DSCUnits.CELSIUS, DSCUnits.KELVIN),
        273.15,
    )

    # Kelvin to Celsius
    assert np.isclose(
        UnitConverter.convert_temperature(273.15, DSCUnits.KELVIN, DSCUnits.CELSIUS),
        0.0,
    )

    # Array conversion
    temps = np.array([0.0, 100.0, 200.0])
    converted = UnitConverter.convert_temperature(
        temps, DSCUnits.CELSIUS, DSCUnits.KELVIN
    )
    assert np.allclose(converted, temps + 273.15)


def test_heat_flow_conversion():
    """Test heat flow unit conversions."""
    # mW to W
    assert np.isclose(
        UnitConverter.convert_heat_flow(1000.0, DSCUnits.MILLIWATTS, DSCUnits.WATTS),
        1.0,
    )

    # W to µW
    assert np.isclose(
        UnitConverter.convert_heat_flow(1.0, DSCUnits.WATTS, DSCUnits.MICROWATTS), 1e6
    )


def test_heating_rate_conversion():
    """Test heating rate unit conversions."""
    # K/min to K/s
    assert np.isclose(
        UnitConverter.convert_heating_rate(
            60.0, DSCUnits.KELVIN_PER_MINUTE, DSCUnits.KELVIN_PER_SECOND
        ),
        1.0,
    )


# Data Validation Tests
def test_validate_temperature_data():
    """Test temperature data validation."""
    temp = generate_temperature_ramp()

    # Valid data
    assert DataValidator.validate_temperature_data(temp)

    # Invalid data types
    with pytest.raises(ValueError):
        DataValidator.validate_temperature_data(temp.tolist())

    # Non-monotonic data is accepted by default (real DSC data has noise,
    # isotherms and cooling segments) but rejected in strict mode
    non_monotonic = temp.copy()
    non_monotonic[500] = non_monotonic[498]
    assert DataValidator.validate_temperature_data(non_monotonic)
    with pytest.raises(ValueError):
        DataValidator.validate_temperature_data(non_monotonic, strict_monotonic=True)

    # Unrealistic jumps between consecutive points
    jumpy = temp.copy()
    jumpy[500:] += 100
    with pytest.raises(ValueError):
        DataValidator.validate_temperature_data(jumpy)

    # Out of range data
    with pytest.raises(ValueError):
        DataValidator.validate_temperature_data(
            temp, min_temp=temp.mean(), max_temp=temp.max()
        )


def test_validate_heat_flow_data():
    """Test heat flow data validation."""
    heat_flow = np.random.randn(1000)
    temp = generate_temperature_ramp()

    # Valid data
    assert DataValidator.validate_heat_flow_data(heat_flow)
    assert DataValidator.validate_heat_flow_data(heat_flow, temp)

    # Mismatched lengths
    with pytest.raises(ValueError):
        DataValidator.validate_heat_flow_data(heat_flow, temp[:-10])

    # Invalid values
    invalid_data = heat_flow.copy()
    invalid_data[0] = np.inf
    with pytest.raises(ValueError):
        DataValidator.validate_heat_flow_data(invalid_data)


def test_check_sampling_rate():
    """Test sampling rate checking."""
    # Uniform sampling
    time = np.linspace(0, 999, 1000)
    interval = DataValidator.check_sampling_rate(time)
    assert isinstance(interval, float)
    assert np.isclose(interval, 1.0)

    time_non_uniform = np.array([0, 1, 2, 4, 5])
    with pytest.raises(ValueError, match="Non-uniform time sampling detected"):
        DataValidator.check_sampling_rate(time_non_uniform)


# Integration Tests
def test_signal_processing_workflow(signal_processor, noisy_data):
    """Test complete signal processing workflow."""
    # Apply multiple processing steps
    smoothed = signal_processor.smooth_signal(noisy_data["signal"])
    cleaned = signal_processor.remove_outliers(smoothed)
    filtered = signal_processor.filter_signal(
        cleaned, sampling_rate=100.0, cutoff_freq=10.0
    )

    # Calculate final quality metrics
    final_noise = signal_processor.calculate_noise_level(filtered)
    original_noise = signal_processor.calculate_noise_level(noisy_data["signal"])

    assert final_noise < original_noise


def test_temperature_conversion_chain():
    """Test chain of temperature conversions."""
    original_temp = 25.0  # °C

    # Convert through multiple units and back
    kelvin = UnitConverter.convert_temperature(
        original_temp, DSCUnits.CELSIUS, DSCUnits.KELVIN
    )
    fahrenheit = UnitConverter.convert_temperature(
        kelvin, DSCUnits.KELVIN, DSCUnits.FAHRENHEIT
    )
    final_celsius = UnitConverter.convert_temperature(
        fahrenheit, DSCUnits.FAHRENHEIT, DSCUnits.CELSIUS
    )

    assert np.isclose(original_temp, final_celsius)


def test_error_handling():
    """Test error handling across utilities."""
    # Invalid smoothing method
    with pytest.raises(ValueError):
        SignalProcessor().smooth_signal(np.random.rand(100), method="invalid")

    # Invalid unit conversion
    with pytest.raises(ValueError):
        UnitConverter.convert_temperature(
            0.0, DSCUnits.WATTS, DSCUnits.KELVIN  # Wrong unit type
        )

    # Invalid data validation
    with pytest.raises(ValueError):
        DataValidator.validate_temperature_data(
            np.array([]),  # Empty array
        )


def test_remove_outliers_flat_signal(signal_processor):
    """Flat regions (zero local variance) must not produce NaNs."""
    data = np.concatenate([np.zeros(50), np.random.default_rng(0).normal(0, 1, 50)])
    cleaned = signal_processor.remove_outliers(data)
    assert np.all(np.isfinite(cleaned))
    np.testing.assert_array_equal(cleaned[:40], 0.0)


def test_bandpass_filter(signal_processor):
    """Bandpass keeps the in-band component and removes the others."""
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    in_band = np.sin(2 * np.pi * 5 * t)
    data = in_band + np.sin(2 * np.pi * 0.2 * t) + np.sin(2 * np.pi * 30 * t)

    filtered = signal_processor.filter_signal(data, fs, (2.0, 10.0), "bandpass")

    core = slice(200, -200)  # Ignore filter edge effects
    assert np.max(np.abs(filtered[core] - in_band[core])) < 0.1

    with pytest.raises(ValueError):
        signal_processor.filter_signal(data, fs, 5.0, "bandpass")


def test_detect_temperature_program():
    """Program detection with rates in K/min and time in seconds."""
    t = np.arange(0, 1800, 1.0)

    # 10 K/min continuous heating
    result = DataValidator.detect_temperature_program(300 + 10 / 60 * t, t)
    assert result["type"] == "continuous"
    assert result["n_heating"] == 1
    assert np.isclose(result["avg_heating_rate"], 10.0)
    assert np.isnan(result["avg_cooling_rate"])

    # Equilibration hold, then heating: still continuous
    temp = np.where(t < 300, 300.0, 300 + 10 / 60 * (t - 300))
    result = DataValidator.detect_temperature_program(temp, t)
    assert result["type"] == "continuous"
    assert result["segments"][0]["type"] == "isothermal"
    assert result["segments"][-1]["start_idx"] > 0
    assert result["segments"][-1]["end_idx"] == len(t)

    # Heat - hold - heat: stepped
    temp = np.piecewise(
        t,
        [t < 600, (t >= 600) & (t < 1200), t >= 1200],
        [
            lambda x: 300 + 10 / 60 * x,
            400.0,
            lambda x: 400 + 10 / 60 * (x - 1200),
        ],
    )
    result = DataValidator.detect_temperature_program(temp, t)
    assert result["type"] == "stepped"
    assert result["n_heating"] == 2

    # Heat then cool: cyclic
    temp = np.where(t < 900, 300 + 10 / 60 * t, 450 - 10 / 60 * (t - 900))
    result = DataValidator.detect_temperature_program(temp, t)
    assert result["type"] == "cyclic"
    assert np.isclose(result["avg_cooling_rate"], -10.0, atol=0.1)


def test_detect_temperature_program_noisy():
    """Sensor noise and a brief temperature stall must not split a ramp."""
    rng = np.random.default_rng(1)
    t = np.arange(0, 3600, 0.5)
    temp = 300 + 1 / 60 * t  # 1 K/min
    stall = (t >= 1800) & (t < 1830)  # 30 s stall, e.g. during melting
    temp[stall] = temp[np.argmax(stall)]
    temp[t >= 1830] -= 0.5
    temp += rng.normal(0, 0.005, len(t))

    result = DataValidator.detect_temperature_program(temp, t)
    assert result["type"] == "continuous"
    assert len(result["segments"]) == 1
    assert np.isclose(result["avg_heating_rate"], 1.0, atol=0.05)

    # Filtering can be disabled
    raw = DataValidator.detect_temperature_program(temp, t, min_segment_duration=0)
    assert len(raw["segments"]) > 1
