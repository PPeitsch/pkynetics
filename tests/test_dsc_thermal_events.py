"""Tests for DSC thermal events detection functionality."""

import numpy as np
import pytest
from numpy.typing import NDArray

from pkynetics.technique_analysis.dsc.thermal_events import (
    CrystallizationEvent,
    GlassTransition,
    MeltingEvent,
    PhaseTransition,
    ThermalEventDetector,
)


# Test data generation utilities
def generate_glass_transition(
    temperature: NDArray[np.float64],
    tg: float = 373.15,  # 100°C
    delta_cp: float = 0.5,
    width: float = 10.0,
) -> NDArray[np.float64]:
    """Generate synthetic glass transition data."""
    # Create sigmoid shape characteristic of glass transition
    x = (temperature - tg) / width
    transition = delta_cp / (1 + np.exp(-x))
    return transition


def generate_crystallization_peak(
    temperature: NDArray[np.float64],
    peak_temp: float = 423.15,  # 150°C
    amplitude: float = -1.0,  # Exothermic
    width: float = 15.0,
) -> NDArray[np.float64]:
    """Generate synthetic crystallization peak."""
    return amplitude * np.exp(-(((temperature - peak_temp) / width) ** 2))


def generate_melting_peak(
    temperature: NDArray[np.float64],
    peak_temp: float = 473.15,  # 200°C
    amplitude: float = 1.0,  # Endothermic
    width: float = 20.0,
) -> NDArray[np.float64]:
    """Generate synthetic melting peak."""
    return amplitude * np.exp(-(((temperature - peak_temp) / width) ** 2))


def generate_phase_transition(
    temperature: NDArray[np.float64],
    transition_temp: float = 443.15,  # 170°C
    amplitude: float = 0.8,
    width: float = 12.0,
    transition_type: str = "first_order",
) -> NDArray[np.float64]:
    """Generate synthetic phase transition data."""
    if transition_type == "first_order":
        return amplitude * np.exp(-(((temperature - transition_temp) / width) ** 2))
    else:  # second_order
        x = (temperature - transition_temp) / width
        return amplitude * (1 / (1 + np.exp(-x)) - 0.5)


@pytest.fixture
def event_detector():
    """Create ThermalEventDetector instance."""
    return ThermalEventDetector()


@pytest.fixture
def temperature_data():
    """Generate temperature array."""
    return np.linspace(298.15, 573.15, 1000)  # 25°C to 300°C


@pytest.fixture
def glass_transition_data(temperature_data):
    """Generate data with glass transition."""
    heat_flow = generate_glass_transition(temperature_data)
    baseline = np.zeros_like(temperature_data)
    return {
        "temperature": temperature_data,
        "heat_flow": heat_flow,
        "baseline": baseline,
        "expected_tg": 373.15,
        "expected_width": 10.0,
    }


@pytest.fixture
def crystallization_data(temperature_data):
    """Generate data with crystallization peak."""
    heat_flow = generate_crystallization_peak(temperature_data)
    baseline = np.zeros_like(temperature_data)
    return {
        "temperature": temperature_data,
        "heat_flow": heat_flow,
        "baseline": baseline,
        "expected_peak": 423.15,
        "expected_width": 15.0,
    }


@pytest.fixture
def melting_data(temperature_data):
    """Generate data with melting peak."""
    heat_flow = generate_melting_peak(temperature_data)
    baseline = np.zeros_like(temperature_data)
    return {
        "temperature": temperature_data,
        "heat_flow": heat_flow,
        "baseline": baseline,
        "expected_peak": 473.15,
        "expected_width": 20.0,
    }


@pytest.fixture
def complex_data(temperature_data):
    """Generate data with multiple thermal events."""
    gt = generate_glass_transition(temperature_data, tg=353.15)
    cryst = generate_crystallization_peak(temperature_data, peak_temp=423.15)
    melt = generate_melting_peak(temperature_data, peak_temp=493.15)

    heat_flow = gt + cryst + melt
    baseline = np.zeros_like(temperature_data)

    return {
        "temperature": temperature_data,
        "heat_flow": heat_flow,
        "baseline": baseline,
        "expected_events": {
            "glass_transition": 353.15,
            "crystallization": 423.15,
            "melting": 493.15,
        },
    }


# Glass Transition Tests
def test_glass_transition_detection(event_detector, glass_transition_data):
    """Test detection of glass transition."""
    result = event_detector.detect_glass_transition(
        glass_transition_data["temperature"],
        glass_transition_data["heat_flow"],
        glass_transition_data["baseline"],
    )

    assert isinstance(result, GlassTransition)
    assert abs(result.midpoint_temperature - glass_transition_data["expected_tg"]) < 5.0
    assert abs(result.width - glass_transition_data["expected_width"]) < 2.0
    assert result.delta_cp > 0
    assert all(metric > 0 for metric in result.quality_metrics.values())


def test_glass_transition_without_baseline(event_detector, glass_transition_data):
    """Test glass transition detection without baseline."""
    result = event_detector.detect_glass_transition(
        glass_transition_data["temperature"], glass_transition_data["heat_flow"]
    )

    assert isinstance(result, GlassTransition)
    assert np.isnan(result.delta_cp)


def test_no_glass_transition(event_detector, temperature_data):
    """Test behavior when no glass transition is present."""
    heat_flow = np.zeros_like(temperature_data)  # Flat line
    result = event_detector.detect_glass_transition(temperature_data, heat_flow)

    assert result is None


# Crystallization Tests
def test_crystallization_detection(event_detector, crystallization_data):
    """Test detection of crystallization event."""
    events = event_detector.detect_crystallization(
        crystallization_data["temperature"],
        crystallization_data["heat_flow"],
        crystallization_data["baseline"],
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, CrystallizationEvent)
    assert abs(event.peak_temperature - crystallization_data["expected_peak"]) < 5.0
    assert abs(event.width - crystallization_data["expected_width"]) < 2.0
    assert event.enthalpy < 0  # Exothermic
    assert event.crystallization_rate is not None
    assert all(metric > 0 for metric in event.quality_metrics.values())


def test_multiple_crystallization_events(event_detector, temperature_data):
    """Test detection of multiple crystallization events."""
    # Generate two crystallization peaks
    heat_flow = generate_crystallization_peak(
        temperature_data, peak_temp=373.15
    ) + generate_crystallization_peak(temperature_data, peak_temp=473.15)

    events = event_detector.detect_crystallization(temperature_data, heat_flow)

    assert len(events) == 2
    assert abs(events[0].peak_temperature - 373.15) < 5.0
    assert abs(events[1].peak_temperature - 473.15) < 5.0


# Melting Tests
def test_melting_detection(event_detector, melting_data):
    """Test detection of melting event."""
    events = event_detector.detect_melting(
        melting_data["temperature"], melting_data["heat_flow"], melting_data["baseline"]
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, MeltingEvent)
    assert abs(event.peak_temperature - melting_data["expected_peak"]) < 5.0
    assert abs(event.width - melting_data["expected_width"]) < 2.0
    assert event.enthalpy > 0  # Endothermic
    assert all(metric > 0 for metric in event.quality_metrics.values())


# Phase Transition Tests
def test_first_order_transition(event_detector, temperature_data):
    """Test detection of first-order phase transition."""
    heat_flow = generate_phase_transition(
        temperature_data, transition_type="first_order"
    )

    transitions = event_detector.detect_phase_transitions(temperature_data, heat_flow)

    assert len(transitions) > 0
    transition = transitions[0]
    assert isinstance(transition, PhaseTransition)
    assert transition.transition_type == "first_order"
    assert all(metric > 0 for metric in transition.quality_metrics.values())


def test_second_order_transition(event_detector, temperature_data):
    """Test detection of second-order phase transition."""
    heat_flow = generate_phase_transition(
        temperature_data, transition_type="second_order"
    )

    transitions = event_detector.detect_phase_transitions(temperature_data, heat_flow)

    assert len(transitions) > 0
    assert any(t.transition_type == "second_order" for t in transitions)


# Complex Data Tests
def test_multiple_events_detection(event_detector, complex_data):
    """Test detection of multiple thermal events in complex data."""
    # Detect all types of events
    gt = event_detector.detect_glass_transition(
        complex_data["temperature"], complex_data["heat_flow"]
    )
    cryst = event_detector.detect_crystallization(
        complex_data["temperature"], complex_data["heat_flow"]
    )
    melt = event_detector.detect_melting(
        complex_data["temperature"], complex_data["heat_flow"]
    )

    # Check glass transition
    assert gt is not None
    assert (
        abs(
            gt.midpoint_temperature
            - complex_data["expected_events"]["glass_transition"]
        )
        < 5.0
    )

    # Check crystallization
    assert len(cryst) > 0
    assert (
        abs(
            cryst[0].peak_temperature
            - complex_data["expected_events"]["crystallization"]
        )
        < 5.0
    )

    # Check melting
    assert len(melt) > 0
    assert (
        abs(melt[0].peak_temperature - complex_data["expected_events"]["melting"]) < 5.0
    )


# Error Handling Tests
def test_invalid_input_shape(event_detector):
    """Test handling of mismatched array shapes."""
    temp = np.array([1, 2, 3])
    heat_flow = np.array([1, 2])

    with pytest.raises(ValueError):
        event_detector.detect_glass_transition(temp, heat_flow)


def test_empty_input(event_detector):
    """Test handling of empty input arrays."""
    with pytest.raises(ValueError):
        event_detector.detect_crystallization(np.array([]), np.array([]))


def test_noisy_data_handling(event_detector, glass_transition_data):
    """Test handling of noisy data."""
    noise = np.random.normal(0, 0.1, size=len(glass_transition_data["heat_flow"]))
    noisy_data = glass_transition_data["heat_flow"] + noise

    result = event_detector.detect_glass_transition(
        glass_transition_data["temperature"], noisy_data
    )

    # Should still detect the transition despite noise
    assert result is not None
    assert result.quality_metrics["snr"] < 1.0  # Lower SNR due to noise


# Quality Metrics Tests
def test_quality_metrics_calculation(event_detector, crystallization_data):
    """Test calculation of quality metrics."""
    events = event_detector.detect_crystallization(
        crystallization_data["temperature"], crystallization_data["heat_flow"]
    )

    assert len(events) > 0
    metrics = events[0].quality_metrics

    assert "peak_to_noise" in metrics
    assert "sharpness" in metrics
    assert "baseline_stability" in metrics
    assert all(isinstance(v, float) for v in metrics.values())
