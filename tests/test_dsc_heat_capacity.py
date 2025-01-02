"""Tests for DSC specific heat capacity analysis module."""

import numpy as np
import pytest
from numpy.typing import NDArray

from pkynetics.technique_analysis.dsc.heat_capacity import (
    CalibrationData,
    CpCalculator,
    CpMethod,
    CpResult,
)


# Test data generation utilities
def generate_standard_dsc_data(
    temp_range: tuple = (300, 500),
    n_points: int = 1000,
    sample_mass: float = 10.0,
    ref_mass: float = 10.0,
    heating_rate: float = 10.0,
) -> dict:
    """Generate synthetic DSC data for standard Cp method."""
    temperature = np.linspace(*temp_range, n_points)

    # Generate synthetic heat flows
    baseline_hf = 0.1 * (temperature - temp_range[0])  # Linear baseline
    ref_cp = 0.8 + 0.0002 * (temperature - 300)  # Temperature-dependent Cp

    # Sample heat flow with temperature dependence
    sample_hf = baseline_hf + sample_mass * heating_rate * (
        0.5 + 0.001 * (temperature - 300)
    )

    # Reference heat flow
    ref_hf = baseline_hf + ref_mass * heating_rate * ref_cp

    return {
        "temperature": temperature,
        "sample_heat_flow": sample_hf,
        "reference_heat_flow": ref_hf,
        "baseline_heat_flow": baseline_hf,
        "heating_rate": heating_rate,
        "sample_mass": sample_mass,
        "reference_mass": ref_mass,
        "reference_cp": ref_cp,
        "expected_cp": 0.5 + 0.001 * (temperature - 300),  # True Cp
    }


def generate_modulated_dsc_data(
    temp_range: tuple = (300, 500),
    n_points: int = 1000,
    modulation_period: float = 60.0,
    modulation_amplitude: float = 0.5,
) -> dict:
    """Generate synthetic modulated DSC data."""
    temperature = np.linspace(*temp_range, n_points)
    time = np.linspace(0, modulation_period * 10, n_points)

    # Base Cp signal
    base_cp = 0.5 + 0.001 * (temperature - 300)

    # Generate reversing heat flow
    omega = 2 * np.pi / modulation_period
    heating_rate = omega * modulation_amplitude
    reversing_hf = base_cp * heating_rate

    return {
        "temperature": temperature,
        "time": time,
        "reversing_heat_flow": reversing_hf,
        "modulation_period": modulation_period,
        "modulation_amplitude": modulation_amplitude,
        "expected_cp": base_cp,
    }


def generate_step_dsc_data(
    temp_range: tuple = (300, 500),
    n_points: int = 1000,
    step_size: float = 10.0,
    n_steps: int = 20,
) -> dict:
    """Generate synthetic DSC data for step method."""
    # Generate stepped temperature profile with clear plateaus
    temperature = np.zeros(n_points)
    heat_flow = np.zeros(n_points)
    points_per_step = n_points // n_steps

    base_temps = np.linspace(*temp_range, n_steps)
    base_cp = 0.5 + 0.001 * (base_temps - 300)

    for i in range(n_steps):
        start_idx = i * points_per_step
        end_idx = (i + 1) * points_per_step
        temperature[start_idx:end_idx] = base_temps[i]
        heat_flow[start_idx:end_idx] = base_cp[i] * step_size

    return {
        "temperature": temperature,
        "heat_flow": heat_flow,
        "step_size": step_size,
        "expected_cp": np.repeat(base_cp, points_per_step),
        "sample_mass": 1.0,
    }


@pytest.fixture
def cp_calculator():
    """Create CpCalculator instance."""
    return CpCalculator()


@pytest.fixture
def standard_data():
    """Generate standard DSC test data."""
    return generate_standard_dsc_data()


@pytest.fixture
def modulated_data():
    """Generate modulated DSC test data."""
    return generate_modulated_dsc_data()


@pytest.fixture
def step_data():
    """Generate step method test data."""
    return generate_step_dsc_data()


@pytest.fixture
def sapphire_calibration_data():
    """Generate sapphire calibration data."""
    data = generate_standard_dsc_data()
    # Modify heat flow to match sapphire characteristics
    data["sample_heat_flow"] *= 1.1  # Introduce systematic error
    return data


# Basic Functionality Tests
def test_standard_cp_calculation(cp_calculator, standard_data):
    """Test standard three-run Cp calculation."""
    result = cp_calculator.calculate_cp(standard_data, method=CpMethod.STANDARD)

    assert isinstance(result, CpResult)
    assert result.method == CpMethod.STANDARD
    np.testing.assert_allclose(
        result.specific_heat, standard_data["expected_cp"], rtol=0.05  # 5% tolerance
    )
    assert all(metric > 0 for metric in result.quality_metrics.values())


def test_modulated_cp_calculation(cp_calculator, modulated_data):
    """Test modulated DSC Cp calculation."""
    result = cp_calculator.calculate_cp(modulated_data, method=CpMethod.MODULATED)

    assert isinstance(result, CpResult)
    assert result.method == CpMethod.MODULATED
    np.testing.assert_allclose(
        result.specific_heat, modulated_data["expected_cp"], rtol=0.05
    )
    assert "phase_angle" in result.metadata


def test_step_cp_calculation(cp_calculator, step_data):
    """Test step method Cp calculation."""
    result = cp_calculator.calculate_cp(step_data, method=CpMethod.STEP)

    assert isinstance(result, CpResult)
    assert result.method == CpMethod.STEP
    np.testing.assert_allclose(
        result.specific_heat, step_data["expected_cp"], rtol=0.05
    )
    assert "step_points" in result.metadata


def test_continuous_cp_calculation(cp_calculator, standard_data):
    """Test continuous Cp calculation."""
    result = cp_calculator.calculate_cp(
        {
            "temperature": standard_data["temperature"],
            "heat_flow": standard_data["sample_heat_flow"],
            "heating_rate": standard_data["heating_rate"],
            "sample_mass": standard_data["sample_mass"],
        },
        method=CpMethod.CONTINUOUS,
    )

    assert isinstance(result, CpResult)
    assert result.method == CpMethod.CONTINUOUS
    assert len(result.specific_heat) == len(standard_data["temperature"])


# Calibration Tests
def test_calibration(cp_calculator, sapphire_calibration_data):
    """Test DSC calibration with sapphire."""
    cal_result = cp_calculator.calibrate(
        sapphire_calibration_data, reference_material="sapphire"
    )

    assert isinstance(cal_result, CalibrationData)
    assert cal_result.reference_material == "sapphire"
    assert len(cal_result.calibration_factors) == len(
        sapphire_calibration_data["temperature"]
    )
    assert np.all(cal_result.uncertainty > 0)


def test_calibrated_measurement(
    cp_calculator, standard_data, sapphire_calibration_data
):
    """Test Cp measurement with calibration applied."""
    # First calibrate
    cp_calculator.calibrate(sapphire_calibration_data, reference_material="sapphire")

    # Then measure with calibration
    result = cp_calculator.calculate_cp(standard_data, method=CpMethod.DIRECT)

    assert isinstance(result, CpResult)
    assert "calibration_material" in result.metadata
    assert result.uncertainty is not None


# Uncertainty Analysis Tests
def test_standard_uncertainty(cp_calculator, standard_data):
    """Test uncertainty calculation for standard method."""
    result = cp_calculator.calculate_cp(standard_data, method=CpMethod.STANDARD)

    assert result.uncertainty is not None
    assert np.all(result.uncertainty > 0)
    assert np.all(
        result.uncertainty < result.specific_heat
    )  # Uncertainty should be reasonable


def test_modulated_uncertainty(cp_calculator, modulated_data):
    """Test uncertainty calculation for modulated method."""
    result = cp_calculator.calculate_cp(modulated_data, method=CpMethod.MODULATED)

    assert result.uncertainty is not None
    assert np.all(result.uncertainty > 0)
    assert "snr" in result.quality_metrics


def test_calibration_uncertainty(cp_calculator, sapphire_calibration_data):
    """Test calibration uncertainty propagation."""
    cal_result = cp_calculator.calibrate(
        sapphire_calibration_data, reference_material="sapphire"
    )

    assert cal_result.uncertainty is not None
    assert np.all(cal_result.uncertainty > 0)
    assert np.all(cal_result.uncertainty < cal_result.calibration_factors)


# Error Handling Tests
def test_invalid_method(cp_calculator, standard_data):
    """Test handling of invalid Cp calculation method."""
    with pytest.raises(ValueError):
        cp_calculator.calculate_cp(standard_data, method="invalid_method")


def test_missing_data(cp_calculator):
    """Test handling of missing required data."""
    with pytest.raises(ValueError):
        cp_calculator.calculate_cp({}, method=CpMethod.STANDARD)


def test_invalid_calibration_material(cp_calculator, standard_data):
    """Test handling of invalid reference material."""
    with pytest.raises(ValueError):
        cp_calculator.calibrate(standard_data, reference_material="invalid_material")


def test_uncalibrated_direct_method(cp_calculator, standard_data):
    """Test direct method without calibration."""
    with pytest.raises(ValueError):
        cp_calculator.calculate_cp(standard_data, method=CpMethod.DIRECT)


# Integration Tests
# def test_multiple_methods_comparison(cp_calculator, standard_data):
#    """Compare results from different Cp calculation methods."""
#    methods = [CpMethod.STANDARD, CpMethod.CONTINUOUS]
#    results = []
#
#    for method in methods:
#        result = cp_calculator.calculate_cp(standard_data, method=method)
#        results.append(result)
#
#    # Results should be similar within tolerance
#    for r1, r2 in zip(results[:-1], results[1:]):
#        np.testing.assert_allclose(
#            r1.specific_heat,
#            r2.specific_heat,
#            rtol=0.1,  # 10% tolerance between methods
#        )


def test_calibration_workflow(cp_calculator, sapphire_calibration_data, standard_data):
    """Test complete calibration and measurement workflow."""
    # Calibrate
    cal_result = cp_calculator.calibrate(
        sapphire_calibration_data, reference_material="sapphire"
    )

    # Measure without calibration
    uncal_result = cp_calculator.calculate_cp(standard_data, method=CpMethod.STANDARD)

    # Measure with calibration
    cal_result = cp_calculator.calculate_cp(standard_data, method=CpMethod.DIRECT)

    # Calibrated result should have lower uncertainty
    assert np.mean(cal_result.uncertainty) < np.mean(uncal_result.uncertainty)


def test_quality_metrics(cp_calculator, standard_data):
    """Test quality metrics calculation."""
    result = cp_calculator.calculate_cp(standard_data, method=CpMethod.STANDARD)

    assert "snr" in result.quality_metrics
    assert "smoothness" in result.quality_metrics
    assert all(metric > 0 for metric in result.quality_metrics.values())
