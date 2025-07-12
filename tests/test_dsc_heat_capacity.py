# tests/test_dsc_heat_capacity.py

import numpy as np
import pytest

from pkynetics.technique_analysis.dsc.heat_capacity import CpCalculator
from pkynetics.technique_analysis.dsc.types import CpMethod, OperationMode


# Reusable test data
@pytest.fixture
def sample_data():
    """Provides common sample data for tests."""
    return {
        "temperature": np.linspace(300, 400, 101),
        "heat_flow": np.linspace(1, 2, 101),
        "sample_mass": 10.0,
        "heating_rate": 10.0,
    }


@pytest.fixture
def reference_data():
    """Provides common reference data for tests."""
    return {
        "temperature": np.linspace(300, 400, 101),
        "heat_flow": np.linspace(0.5, 1, 101),
        "mass": 10.0,
        "cp": np.full(101, 0.8),
    }


def test_three_step_cp_calculation(sample_data, reference_data):
    """Test three-step Cp calculation."""
    calculator = CpCalculator()
    result = calculator.calculate_cp(
        **sample_data,
        method=CpMethod.THREE_STEP,
        reference_data=reference_data,
        use_calibration=False,
    )
    assert result is not None
    assert result.method == CpMethod.THREE_STEP
    assert len(result.specific_heat) == len(sample_data["temperature"])


def test_modulated_cp_calculation(sample_data):
    """Test modulated Cp calculation."""
    calculator = CpCalculator()
    result = calculator.calculate_cp(
        **sample_data,
        method=CpMethod.MODULATED,
        modulation_period=60.0,
        modulation_amplitude=0.5,
        use_calibration=False,
    )
    assert result is not None
    assert result.method == CpMethod.MODULATED


def test_stepped_cp_calculation(sample_data, reference_data):
    """Test stepped Cp calculation."""
    calculator = CpCalculator()
    # Simulate stepped data by creating stable regions
    sample_data["heat_flow"][10:20] = 1.2
    sample_data["heat_flow"][50:60] = 1.6

    result = calculator.calculate_cp(
        **sample_data,
        method=CpMethod.THREE_STEP,
        operation_mode=OperationMode.STEPPED,
        reference_data=reference_data,
        use_calibration=False,
    )
    assert result is not None
    assert result.operation_mode == OperationMode.STEPPED
    assert result.stable_regions is not None


def test_continuous_cp_calculation(sample_data, reference_data):
    """Test continuous Cp calculation."""
    calculator = CpCalculator()
    result = calculator.calculate_cp(
        **sample_data,
        method=CpMethod.THREE_STEP,
        operation_mode=OperationMode.CONTINUOUS,
        reference_data=reference_data,
        use_calibration=False,
    )
    assert result is not None
    assert result.operation_mode == OperationMode.CONTINUOUS
    assert result.stable_regions is None


def test_calibration(sample_data):
    """Test DSC calibration."""
    calculator = CpCalculator()
    calibration_data = calculator.calibrate(
        **sample_data, reference_material="sapphire"
    )
    assert calibration_data is not None
    assert calibration_data.reference_material == "sapphire"
    assert calculator.calibration_data is not None


def test_calibrated_measurement(sample_data, reference_data):
    """Test calibrated Cp measurement."""
    calculator = CpCalculator()
    calculator.calibrate(**sample_data, reference_material="sapphire")

    result = calculator.calculate_cp(
        **sample_data,
        method=CpMethod.THREE_STEP,
        reference_data=reference_data,
        use_calibration=True,
    )
    assert result is not None
    assert result.metadata.get("calibration_applied") is True


def test_three_step_uncertainty(sample_data, reference_data):
    """Test uncertainty calculation for three-step method."""
    calculator = CpCalculator()
    result = calculator.calculate_cp(
        **sample_data,
        method=CpMethod.THREE_STEP,
        reference_data=reference_data,
        use_calibration=False,
    )
    assert np.all(result.uncertainty >= 0)


def test_modulated_uncertainty(sample_data):
    """Test uncertainty calculation for modulated method."""
    calculator = CpCalculator()
    result = calculator.calculate_cp(
        **sample_data,
        method=CpMethod.MODULATED,
        modulation_period=60.0,
        modulation_amplitude=0.5,
        use_calibration=False,
    )
    assert np.all(result.uncertainty >= 0)


def test_calibration_uncertainty(sample_data):
    """Test uncertainty calculation for calibration."""
    calculator = CpCalculator()
    calibration_data = calculator.calibrate(
        **sample_data, reference_material="sapphire"
    )
    assert np.all(calibration_data.uncertainty >= 0)


def test_invalid_method(sample_data):
    """Test handling of invalid Cp method."""
    calculator = CpCalculator()
    with pytest.raises(ValueError):
        calculator.calculate_cp(**sample_data, method="invalid_method")


def test_missing_data(sample_data):
    """Test handling of missing reference data for three-step method."""
    calculator = CpCalculator()
    with pytest.raises(ValueError):
        calculator.calculate_cp(**sample_data, method=CpMethod.THREE_STEP)


def test_invalid_calibration_material(sample_data):
    """Test handling of invalid calibration material."""
    calculator = CpCalculator()
    with pytest.raises(ValueError):
        calculator.calibrate(**sample_data, reference_material="unobtanium")


def test_single_step_method(sample_data):
    """Test single-step (direct) method without calibration."""
    calculator = CpCalculator()
    result = calculator.calculate_cp(
        **sample_data, method=CpMethod.SINGLE_STEP, use_calibration=False
    )
    assert result.method == CpMethod.SINGLE_STEP
    assert np.mean(result.specific_heat) > 0


def test_calibration_workflow(sample_data, reference_data):
    """Test full calibration and measurement workflow."""
    calculator = CpCalculator()
    # Calibrate first
    calculator.calibrate(**sample_data, reference_material="sapphire")
    assert calculator.calibration_data is not None

    # Then perform a measurement
    result = calculator.calculate_cp(
        **sample_data,
        method=CpMethod.THREE_STEP,
        reference_data=reference_data,
        use_calibration=True,
    )
    assert result.metadata.get("calibration_applied") is True
    assert result.specific_heat is not None


def test_quality_metrics(sample_data, reference_data):
    """Test calculation of quality metrics."""
    calculator = CpCalculator()
    result = calculator.calculate_cp(
        **sample_data,
        method=CpMethod.THREE_STEP,
        reference_data=reference_data,
        use_calibration=False,
    )
    assert "quality_score" in result.quality_metrics
    assert 0 <= result.quality_metrics["quality_score"] <= 1
