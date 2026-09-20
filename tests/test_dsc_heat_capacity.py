"""Tests for DSC specific heat capacity analysis module.

Synthetic data come from a simple instrument model: the measured heat flow
is the blank signal plus the sensitivity k(T) times the heat absorbed by the
sample, m * Cp(T) * dT/dt (mW = mg * J/(g*K) * K/s), optionally with a
first-order thermal lag and a sample-specific isothermal offset.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from pkynetics.technique_analysis.dsc.heat_capacity import (
    CalibrationData,
    CpCalculator,
    CpMethod,
    CpResult,
    reference_cp,
)
from pkynetics.technique_analysis.dsc.types import OperationMode

SAMPLE_MASS = 20.0  # mg
SAPPHIRE_MASS = 30.0  # mg


def sample_cp(temperature: NDArray[np.float64]) -> NDArray[np.float64]:
    """True Cp of the synthetic sample, J/(g*K)."""
    return 0.9 + 4e-4 * (temperature - 300)


def sensitivity(temperature: NDArray[np.float64]) -> NDArray[np.float64]:
    """Temperature-dependent instrument sensitivity (unknown to the user)."""
    return 1.15 - 3e-4 * (temperature - 300)


def lag(signal: NDArray[np.float64], time: NDArray[np.float64], tau: float):
    """First-order thermal lag with time constant tau (s)."""
    out = np.empty_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        alpha = 1 - np.exp(-(time[i] - time[i - 1]) / tau)
        out[i] = out[i - 1] + alpha * (signal[i] - out[i - 1])
    return out


def measure(temperature, time, mass, cp, offset=0.0, tau=None, exo_up=False):
    """Measured heat flow (mW) of a run: blank + k(T) * m * Cp * dT/dt."""
    rate = np.gradient(temperature, time)
    absorbed = mass * cp * rate
    if tau is not None:
        absorbed = lag(absorbed, time, tau)
    blank = blank_signal(temperature, time)
    q = blank + sensitivity(temperature) * absorbed + offset
    return -q if exo_up else q


def blank_signal(temperature, time):
    """Empty-pan signal: offset, drift with temperature and a rate term."""
    return -2.0 + 0.004 * (temperature - 300) + 0.3 * np.gradient(temperature, time)


@pytest.fixture
def continuous_runs():
    """Blank, sapphire and sample runs of a 10 K/min ramp, 310-600 K."""
    time = np.arange(0, 1740.0, 1.0)
    temperature = 310 + 10 / 60 * time
    return {
        "time": time,
        "temperature": temperature,
        "blank": measure(temperature, time, 0.0, 0.0),
        "sapphire": measure(
            temperature, time, SAPPHIRE_MASS, reference_cp("sapphire", temperature)
        ),
        "sample": measure(temperature, time, SAMPLE_MASS, sample_cp(temperature)),
    }


def stepped_program(iso_minutes=30.0, rate=5.0, steps=(350, 400, 450, 500)):
    """Isotherms joined by heating ramps (K, K/min)."""
    time, temperature = [0.0], [steps[0]]
    for t_from, t_to in zip(steps, steps[1:]):
        for _ in range(int(iso_minutes * 60)):
            time.append(time[-1] + 1)
            temperature.append(t_from)
        n = int((t_to - t_from) / rate * 60)
        for k in range(1, n + 1):
            time.append(time[-1] + 1)
            temperature.append(t_from + (t_to - t_from) * k / n)
    for _ in range(int(iso_minutes * 60)):
        time.append(time[-1] + 1)
        temperature.append(steps[-1])
    return np.array(time), np.array(temperature, dtype=np.float64)


@pytest.fixture
def stepped_runs():
    """Stepped runs with thermal lag and a sample-specific offset."""
    time, temperature = stepped_program()
    offset = 0.5 + 0.01 * (temperature - 350)  # e.g. different radiation
    return {
        "time": time,
        "temperature": temperature,
        "blank": measure(temperature, time, 0.0, 0.0, tau=60),
        "sapphire": measure(
            temperature,
            time,
            SAPPHIRE_MASS,
            reference_cp("sapphire", temperature),
            tau=60,
        ),
        "sample": measure(
            temperature, time, SAMPLE_MASS, sample_cp(temperature), offset, tau=60
        ),
    }


def mean_cp(t_from: float, t_to: float) -> float:
    grid = np.linspace(t_from, t_to, 1001)
    return float(np.mean(sample_cp(grid)))


@pytest.fixture
def cp_calculator():
    return CpCalculator()


# Reference data
def test_reference_cp_sapphire():
    """NIST-JANAF Shomate values for alpha-Al2O3 and zinc."""
    # Hand-evaluated Shomate equation / 101.9613 g/mol:
    # 298.15 K: 78.80 J/(mol*K); 500 K: 106.12 J/(mol*K)
    np.testing.assert_allclose(reference_cp("sapphire", 298.15), 0.7729, atol=5e-4)
    np.testing.assert_allclose(reference_cp("sapphire", 500.0), 1.0408, atol=5e-4)
    np.testing.assert_allclose(reference_cp("zinc", 298.15), 0.388, atol=5e-3)
    np.testing.assert_allclose(reference_cp("Sapphire", [400.0, 600.0]).shape, (2,))

    with pytest.raises(ValueError, match="valid range"):
        reference_cp("sapphire", 250.0)
    with pytest.raises(ValueError, match="valid range"):
        reference_cp("zinc", 700.0)  # above the melting point
    with pytest.raises(ValueError, match="Unknown"):
        reference_cp("unobtainium", 300.0)


# Continuous ramps
def test_single_step_units(cp_calculator):
    """mW / mg / (K/min / 60) gives J/(g*K)."""
    time = np.arange(0, 600.0)
    temperature = 350 + 10 / 60 * time
    heat_flow = SAMPLE_MASS * 1.0 * (10 / 60) * np.ones_like(time)  # Cp = 1 J/(gK)

    result = cp_calculator.calculate_cp(
        temperature, heat_flow, SAMPLE_MASS, 10.0, method="single_step"
    )

    np.testing.assert_allclose(result.specific_heat, 1.0)


@pytest.mark.parametrize("exo_up", [False, True])
def test_three_step_continuous(continuous_runs, exo_up):
    """The ratio method recovers Cp despite the unknown sensitivity."""
    runs = continuous_runs
    sign = -1 if exo_up else 1
    calculator = CpCalculator(exo_up=exo_up)

    result = calculator.calculate_cp(
        runs["temperature"],
        sign * runs["sample"],
        SAMPLE_MASS,
        10.0,
        method=CpMethod.THREE_STEP,
        reference_data={"heat_flow": sign * runs["sapphire"], "mass": SAPPHIRE_MASS},
        blank_heat_flow=sign * runs["blank"],
    )

    assert isinstance(result, CpResult)
    core = slice(5, -5)  # gradient edge effects
    np.testing.assert_allclose(
        result.specific_heat[core], sample_cp(runs["temperature"])[core], rtol=1e-3
    )
    np.testing.assert_allclose(
        result.uncertainty / result.specific_heat,
        result.quality_metrics["avg_uncertainty"],
    )
    assert result.metadata["blank_corrected"]


def test_three_step_with_cp_array(cp_calculator, continuous_runs):
    """Reference Cp may be given explicitly instead of a material name."""
    runs = continuous_runs
    result = cp_calculator.calculate_cp(
        runs["temperature"],
        runs["sample"],
        SAMPLE_MASS,
        10.0,
        reference_data={
            "heat_flow": runs["sapphire"],
            "mass": SAPPHIRE_MASS,
            "cp": reference_cp("sapphire", runs["temperature"]),
        },
        blank_heat_flow=runs["blank"],
    )
    np.testing.assert_allclose(
        result.specific_heat[5:-5], sample_cp(runs["temperature"])[5:-5], rtol=1e-3
    )


def test_single_step_needs_calibration(cp_calculator, continuous_runs):
    """Uncalibrated single-step Cp carries the instrument sensitivity."""
    runs = continuous_runs
    result = cp_calculator.calculate_cp(
        runs["temperature"],
        runs["sample"],
        SAMPLE_MASS,
        10.0,
        method=CpMethod.SINGLE_STEP,
        blank_heat_flow=runs["blank"],
    )

    expected = sample_cp(runs["temperature"]) * sensitivity(runs["temperature"])
    np.testing.assert_allclose(result.specific_heat[5:-5], expected[5:-5], rtol=1e-3)


def test_calibration_workflow(cp_calculator, continuous_runs):
    """Calibrating with sapphire corrects single-step results."""
    runs = continuous_runs
    calibration = cp_calculator.calibrate(
        runs["temperature"],
        runs["sapphire"],
        SAPPHIRE_MASS,
        10.0,
        reference_material="sapphire",
        blank_heat_flow=runs["blank"],
    )

    assert isinstance(calibration, CalibrationData)
    assert calibration.reference_material == "sapphire"
    np.testing.assert_allclose(
        calibration.calibration_factors[5:-5],
        1 / sensitivity(runs["temperature"])[5:-5],
        rtol=1e-3,
    )

    result = cp_calculator.calculate_cp(
        runs["temperature"][5:-5],
        runs["sample"][5:-5],
        SAMPLE_MASS,
        10.0,
        method=CpMethod.SINGLE_STEP,
        blank_heat_flow=runs["blank"][5:-5],
    )

    assert result.metadata["calibration_applied"]
    np.testing.assert_allclose(
        result.specific_heat[5:-5],
        sample_cp(runs["temperature"][5:-5])[5:-5],
        rtol=2e-3,
    )
    # Calibration uncertainty is propagated
    assert np.all(result.uncertainty > 0)

    # Ratio methods are not affected by the calibration
    three_step = cp_calculator.calculate_cp(
        runs["temperature"],
        runs["sample"],
        SAMPLE_MASS,
        10.0,
        reference_data={"heat_flow": runs["sapphire"], "mass": SAPPHIRE_MASS},
        blank_heat_flow=runs["blank"],
    )
    assert "calibration_applied" not in three_step.metadata

    # Outside the calibrated range
    with pytest.raises(ValueError, match="calibration validity"):
        cp_calculator.calculate_cp(
            runs["temperature"] + 50,
            runs["sample"],
            SAMPLE_MASS,
            10.0,
            method=CpMethod.SINGLE_STEP,
        )


# Stepped programs
def test_stepped_three_step(cp_calculator, stepped_runs):
    """Step method: lag and isothermal offsets do not bias the result.

    Regression for #73: STEPPED used to process every point individually
    instead of averaging over each isothermal plateau, so instability at
    the start of an isotherm scattered the result. One Cp per step is the
    property that guards it — hence the assertion on the length.
    """
    runs = stepped_runs
    result = cp_calculator.calculate_cp(
        runs["temperature"],
        runs["sample"],
        SAMPLE_MASS,
        5.0,
        method=CpMethod.THREE_STEP,
        operation_mode=OperationMode.STEPPED,
        reference_data={
            "heat_flow": runs["sapphire"],
            "mass": SAPPHIRE_MASS,
            "temperature": runs["temperature"],
        },
        time=runs["time"],
        blank_heat_flow=runs["blank"],
    )

    assert result.operation_mode == OperationMode.STEPPED
    assert len(result.specific_heat) == 3
    np.testing.assert_allclose(result.temperature, [375, 425, 475])
    expected = [mean_cp(350, 400), mean_cp(400, 450), mean_cp(450, 500)]
    # Sensitivity varies within a step, so the ratio is not exact
    np.testing.assert_allclose(result.specific_heat, expected, rtol=5e-3)
    assert len(result.stable_regions) == 3


def test_stepped_single_step(cp_calculator):
    """Step method without reference, unit sensitivity: exact mean Cp."""
    time, temperature = stepped_program()
    rate = np.gradient(temperature, time)
    heat_flow = lag(SAMPLE_MASS * sample_cp(temperature) * rate, time, 60) + 1.0

    result = cp_calculator.calculate_cp(
        temperature,
        heat_flow,
        SAMPLE_MASS,
        5.0,
        method="single_step",
        operation_mode="stepped",
        time=time,
    )

    expected = [mean_cp(350, 400), mean_cp(400, 450), mean_cp(450, 500)]
    np.testing.assert_allclose(result.specific_heat, expected, rtol=1e-3)


def test_isotherms_carry_no_cp_signal(cp_calculator, stepped_runs):
    """Offsets during isotherms must not be taken as heat capacity."""
    runs = stepped_runs
    shifted = runs["sample"] + 5.0  # constant offset only
    result = cp_calculator.calculate_cp(
        runs["temperature"],
        shifted,
        SAMPLE_MASS,
        5.0,
        method="single_step",
        operation_mode="stepped",
        time=runs["time"],
        blank_heat_flow=runs["blank"],
    )
    reference = cp_calculator.calculate_cp(
        runs["temperature"],
        runs["sample"],
        SAMPLE_MASS,
        5.0,
        method="single_step",
        operation_mode="stepped",
        time=runs["time"],
        blank_heat_flow=runs["blank"],
    )
    np.testing.assert_allclose(result.specific_heat, reference.specific_heat)


# Modulated DSC
def test_modulated_cp(cp_calculator):
    """Reversing Cp from modulation amplitudes."""
    period, amplitude = 60.0, 0.5
    time = np.arange(0, 3600.0, 0.5)
    omega = 2 * np.pi / period
    temperature = 350 + 2 / 60 * time + amplitude * np.sin(omega * time)
    rate = np.gradient(temperature, time)
    heat_flow = -1.0 + SAMPLE_MASS * sample_cp(temperature) * rate

    result = cp_calculator.calculate_cp(
        temperature,
        heat_flow,
        SAMPLE_MASS,
        2.0,
        method=CpMethod.MODULATED,
        time=time,
        modulation_period=period,
    )

    assert result.method == CpMethod.MODULATED
    np.testing.assert_allclose(
        result.specific_heat, sample_cp(result.temperature), rtol=5e-3
    )

    with pytest.raises(ValueError, match="Time array required"):
        cp_calculator.calculate_cp(
            temperature, heat_flow, SAMPLE_MASS, 2.0, method=CpMethod.MODULATED
        )


# Error handling
def test_invalid_inputs(cp_calculator, continuous_runs):
    runs = continuous_runs
    args = (runs["temperature"], runs["sample"])

    with pytest.raises(ValueError):
        cp_calculator.calculate_cp(*args, SAMPLE_MASS, 10.0, method="invalid")
    with pytest.raises(ValueError, match="Reference data required"):
        cp_calculator.calculate_cp(*args, SAMPLE_MASS, 10.0)
    with pytest.raises(ValueError, match="mass must be positive"):
        cp_calculator.calculate_cp(*args, 0.0, 10.0, method="single_step")
    with pytest.raises(ValueError, match="Heating rate"):
        cp_calculator.calculate_cp(*args, SAMPLE_MASS, 0.0, method="single_step")
    with pytest.raises(ValueError, match="same length"):
        cp_calculator.calculate_cp(
            runs["temperature"], runs["sample"][:-1], SAMPLE_MASS, 10.0
        )
    with pytest.raises(ValueError, match="same length"):
        cp_calculator.calculate_cp(
            *args,
            SAMPLE_MASS,
            10.0,
            reference_data={"heat_flow": runs["sapphire"][:-1], "mass": 30.0},
        )
    with pytest.raises(ValueError, match="Missing"):
        cp_calculator.calculate_cp(
            *args, SAMPLE_MASS, 10.0, reference_data={"heat_flow": runs["sapphire"]}
        )
    with pytest.raises(ValueError, match="Time array required"):
        cp_calculator.calculate_cp(
            *args, SAMPLE_MASS, 10.0, method="single_step", operation_mode="stepped"
        )
    with pytest.raises(ValueError, match="Unknown reference material"):
        cp_calculator.calibrate(
            *args, SAMPLE_MASS, 10.0, reference_material="unobtainium"
        )


def test_quality_metrics(cp_calculator, continuous_runs, capsys):
    runs = continuous_runs
    result = cp_calculator.calculate_cp(
        runs["temperature"],
        runs["sample"],
        SAMPLE_MASS,
        10.0,
        reference_data={"heat_flow": runs["sapphire"], "mass": SAPPHIRE_MASS},
        blank_heat_flow=runs["blank"],
    )

    metrics = result.quality_metrics
    for key in ("snr", "avg_uncertainty", "max_uncertainty", "smoothness"):
        assert key in metrics
    np.testing.assert_allclose(metrics["slope"], 4e-4, rtol=0.05)
    assert metrics["r_squared"] > 0.99
    # No debug output
    assert capsys.readouterr().out == ""
