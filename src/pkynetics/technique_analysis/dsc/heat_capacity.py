"""Heat capacity calculation and analysis module.

Units: temperature in K, time in s, heat flow in mW, mass in mg, heating
rate in K/min; specific heat capacity in J/(g*K). Since mW/mg = W/g,
Cp = q / m / (beta / 60).

Sign convention: by default endothermic heat flow is positive
(``exo_up=False``). Heat flows are corrected with the blank (empty pan)
run when one is given; otherwise they are assumed to be blank-corrected.

Methods:
- SINGLE_STEP: Cp from the sample run alone (optionally blank-corrected);
  accurate only with a heat flow calibration (see :meth:`CpCalculator.calibrate`).
- THREE_STEP: ratio method of ASTM E1269 with blank, reference (sapphire)
  and sample runs; instrument sensitivity cancels out.
- MODULATED: reversing Cp from the first-harmonic amplitudes of heat flow
  and heating rate in temperature-modulated DSC.

Operation modes:
- CONTINUOUS: Cp at every point of a linear ramp.
- STEPPED: heating steps between isotherms (e.g. ISO 11357-4 step method);
  the heat absorbed in each step, relative to the isothermal levels, gives
  the mean Cp over the step. Requires the time array.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid

from .types import CalibrationData, CpMethod, CpResult, OperationMode
from .utilities import DataValidator

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Shomate equation coefficients (A, B, C, D, E; Cp in J/(mol*K), t = T/1000),
# NIST-JANAF Thermochemical Tables (Chase, 1998), via the NIST Chemistry
# WebBook. Relative uncertainties are indicative.
REFERENCE_MATERIALS: Dict[str, Dict[str, Any]] = {
    "sapphire": {
        "shomate": (102.4290, 38.74980, -15.91090, 2.628181, -3.007551),
        "molar_mass": 101.9613,  # g/mol, alpha-Al2O3
        "valid_range": (298.0, 2327.0),
        "relative_uncertainty": 0.005,
        "source": "NIST-JANAF (Chase 1998), alpha-Al2O3",
    },
    "zinc": {
        "shomate": (25.60123, -4.405292, 20.42206, -7.399697, -0.045801),
        "molar_mass": 65.38,
        "valid_range": (298.0, 692.73),  # solid, up to the melting point
        "relative_uncertainty": 0.01,
        "source": "NIST-JANAF (Chase 1998), Zn solid",
    },
}

# Indicative relative standard uncertainties of the inputs
U_HEAT_FLOW = 0.01
U_MASS = 0.001
U_HEATING_RATE = 0.01


def reference_cp(material: str, temperature: Union[float, FloatArray]) -> FloatArray:
    """
    Specific heat capacity of a reference material in J/(g*K).

    Args:
        material: Reference material name ('sapphire' or 'zinc')
        temperature: Temperature(s) in K

    Returns:
        Specific heat capacity at the given temperature(s)

    Raises:
        ValueError: If the material is unknown or the temperature is out of
            the valid range of the reference data
    """
    data = REFERENCE_MATERIALS.get(material.lower())
    if data is None:
        raise ValueError(f"Unknown reference material: {material}")

    temperature = np.asarray(temperature, dtype=np.float64)
    low, high = data["valid_range"]
    if np.any(temperature < low) or np.any(temperature > high):
        raise ValueError(
            f"Temperature outside the valid range of {material} data "
            f"({low}-{high} K)"
        )

    a, b, c, d, e = data["shomate"]
    t = temperature / 1000
    cp_molar = a + b * t + c * t**2 + d * t**3 + e / t**2
    result: FloatArray = cp_molar / data["molar_mass"]
    return result


class CpCalculator:
    """Class for specific heat capacity calculations and analysis."""

    def __init__(
        self,
        data_validator: Optional[DataValidator] = None,
        exo_up: bool = False,
    ):
        """
        Initialize calculator.

        Args:
            data_validator: Optional data validator
            exo_up: True if exothermic heat flow is positive in the data
        """
        self.data_validator = data_validator or DataValidator()
        self.exo_up = exo_up
        self.calibration_data: Optional[CalibrationData] = None

    def calculate_cp(
        self,
        temperature: FloatArray,
        heat_flow: FloatArray,
        sample_mass: float,
        heating_rate: float,
        method: Union[CpMethod, str] = CpMethod.THREE_STEP,
        operation_mode: Union[OperationMode, str] = OperationMode.CONTINUOUS,
        reference_data: Optional[Dict[str, Any]] = None,
        use_calibration: bool = True,
        time: Optional[FloatArray] = None,
        blank_heat_flow: Optional[FloatArray] = None,
        **kwargs: Any,
    ) -> CpResult:
        """
        Calculate specific heat capacity.

        Args:
            temperature: Sample temperature array (K)
            heat_flow: Sample heat flow array (mW)
            sample_mass: Sample mass (mg)
            heating_rate: Nominal heating rate (K/min); in stepped mode the
                actual steps are measured from the data
            method: Calculation method
            operation_mode: Continuous ramp or stepped program
            reference_data: For THREE_STEP: dict with 'heat_flow' (mW) and
                'mass' (mg) of the reference run, recorded with the same
                temperature program and sampling as the sample run, and
                either 'material' (default 'sapphire') or 'cp' (array in
                J/(g*K) at the sample temperatures). Optional 'temperature'
                (the reference run's temperatures, used for its Cp in
                stepped mode).
            use_calibration: Whether to apply a stored calibration
            time: Time array (s); required for STEPPED and MODULATED
            blank_heat_flow: Heat flow of the blank run (mW), same program
                and sampling; if None, heat flows are taken as corrected
            **kwargs: MODULATED: modulation_period (s, default 60),
                periods_per_window (default 2)

        Returns:
            CpResult object with calculated heat capacity

        Raises:
            ValueError: If inputs are invalid or method is unsupported
        """
        temperature = np.asarray(temperature, dtype=np.float64)
        heat_flow = np.asarray(heat_flow, dtype=np.float64)
        self.data_validator.validate_temperature_data(temperature)
        self._check_length(heat_flow, temperature, "Heat flow")
        if not np.all(np.isfinite(heat_flow)):
            raise ValueError("Heat flow contains invalid values")

        if sample_mass <= 0:
            raise ValueError("Sample mass must be positive")
        if heating_rate <= 0:
            raise ValueError("Heating rate must be positive")

        method = CpMethod(method)
        operation_mode = OperationMode(operation_mode)

        if time is not None:
            time = np.asarray(time, dtype=np.float64)
            self._check_length(time, temperature, "Time")
        if blank_heat_flow is not None:
            blank_heat_flow = np.asarray(blank_heat_flow, dtype=np.float64)
            self._check_length(blank_heat_flow, temperature, "Blank heat flow")

        if method == CpMethod.MODULATED:
            if time is None:
                raise ValueError("Time array required for modulated method")
            result = self._calculate_modulated_cp(
                temperature, heat_flow, sample_mass, heating_rate, time, **kwargs
            )
        elif method in (CpMethod.SINGLE_STEP, CpMethod.THREE_STEP):
            if method == CpMethod.THREE_STEP:
                if reference_data is None:
                    raise ValueError("Reference data required for three-step method")
                self.validate_reference_data(reference_data, len(temperature))
            if operation_mode == OperationMode.STEPPED:
                if time is None:
                    raise ValueError("Time array required for stepped mode")
                result = self._calculate_stepped_cp(
                    temperature,
                    heat_flow,
                    sample_mass,
                    heating_rate,
                    time,
                    blank_heat_flow,
                    method,
                    reference_data,
                )
            else:
                result = self._calculate_continuous_cp(
                    temperature,
                    heat_flow,
                    sample_mass,
                    heating_rate,
                    blank_heat_flow,
                    method,
                    reference_data,
                )
        else:
            raise ValueError(f"Unsupported Cp calculation method: {method}")

        if use_calibration and self.calibration_data is not None:
            result = self._apply_calibration(result)

        return result

    def calibrate(
        self,
        temperature: FloatArray,
        heat_flow: FloatArray,
        sample_mass: float,
        heating_rate: float,
        reference_material: str = "sapphire",
        operation_mode: Union[OperationMode, str] = OperationMode.CONTINUOUS,
        time: Optional[FloatArray] = None,
        blank_heat_flow: Optional[FloatArray] = None,
    ) -> CalibrationData:
        """
        Calibrate single-step Cp with a run of a reference material.

        The calibration factors are reference Cp / measured single-step Cp;
        subsequent single-step results are multiplied by them.

        Args:
            temperature: Temperature array of the reference run (K)
            heat_flow: Heat flow of the reference run (mW)
            sample_mass: Mass of the reference material (mg)
            heating_rate: Heating rate (K/min)
            reference_material: Name of the reference material
            operation_mode: Measurement mode
            time: Time array (s), required for stepped mode
            blank_heat_flow: Heat flow of the blank run (mW)

        Returns:
            CalibrationData object (also stored on the calculator)

        Raises:
            ValueError: If the reference material is unknown or data is invalid
        """
        if reference_material.lower() not in REFERENCE_MATERIALS:
            raise ValueError(f"Unknown reference material: {reference_material}")
        operation_mode = OperationMode(operation_mode)

        measured = self.calculate_cp(
            temperature,
            heat_flow,
            sample_mass,
            heating_rate,
            method=CpMethod.SINGLE_STEP,
            operation_mode=operation_mode,
            use_calibration=False,
            time=time,
            blank_heat_flow=blank_heat_flow,
        )

        ref_cp = reference_cp(reference_material, measured.temperature)
        factors = ref_cp / measured.specific_heat

        u_ref = REFERENCE_MATERIALS[reference_material.lower()]["relative_uncertainty"]
        u_meas = measured.uncertainty / np.abs(measured.specific_heat)
        uncertainty = np.sqrt(u_meas**2 + u_ref**2) * np.abs(factors)

        self.calibration_data = CalibrationData(
            reference_material=reference_material,
            temperature=measured.temperature,
            measured_cp=measured.specific_heat,
            reference_cp=ref_cp,
            calibration_factors=factors,
            uncertainty=uncertainty,
            valid_range=(
                float(np.min(measured.temperature)),
                float(np.max(measured.temperature)),
            ),
            metadata={
                "sample_mass": sample_mass,
                "heating_rate": heating_rate,
                "operation_mode": operation_mode.value,
                "reference_source": REFERENCE_MATERIALS[reference_material.lower()][
                    "source"
                ],
            },
        )
        return self.calibration_data

    # ------------------------------------------------------------------
    # Continuous ramps
    # ------------------------------------------------------------------

    def _calculate_continuous_cp(
        self,
        temperature: FloatArray,
        heat_flow: FloatArray,
        sample_mass: float,
        heating_rate: float,
        blank_heat_flow: Optional[FloatArray],
        method: CpMethod,
        reference_data: Optional[Dict[str, Any]],
    ) -> CpResult:
        """Cp at every point of a linear heating ramp."""
        sample_signal = self._endo_positive(heat_flow, blank_heat_flow)
        metadata: Dict[str, Any] = {
            "sample_mass": sample_mass,
            "heating_rate": heating_rate,
            "operation_mode": OperationMode.CONTINUOUS.value,
            "blank_corrected": blank_heat_flow is not None,
        }

        if method == CpMethod.SINGLE_STEP:
            cp = sample_signal / sample_mass / (heating_rate / 60)
            u_rel = np.sqrt(U_HEAT_FLOW**2 + U_MASS**2 + U_HEATING_RATE**2)
        else:
            assert reference_data is not None
            ref_signal = self._endo_positive(
                np.asarray(reference_data["heat_flow"], dtype=np.float64),
                blank_heat_flow,
            )
            ref_cp = self._reference_cp_at(reference_data, temperature)
            ref_mass = float(reference_data["mass"])
            if np.any(ref_signal == 0):
                raise ValueError("Reference heat flow equals the blank")
            cp = sample_signal / ref_signal * ref_mass / sample_mass * ref_cp
            u_rel = self._three_step_relative_uncertainty(reference_data)
            metadata["reference_mass"] = ref_mass

        return self._result(
            temperature,
            cp,
            np.abs(cp) * u_rel,
            method,
            OperationMode.CONTINUOUS,
            metadata,
        )

    # ------------------------------------------------------------------
    # Stepped programs
    # ------------------------------------------------------------------

    def _calculate_stepped_cp(
        self,
        temperature: FloatArray,
        heat_flow: FloatArray,
        sample_mass: float,
        heating_rate: float,
        time: FloatArray,
        blank_heat_flow: Optional[FloatArray],
        method: CpMethod,
        reference_data: Optional[Dict[str, Any]],
    ) -> CpResult:
        """Mean Cp over each heating step between two isotherms."""
        steps = self.find_heating_steps(temperature, time)
        if not steps:
            raise ValueError("No heating steps between isotherms found")

        sample_signal = self._endo_positive(heat_flow, blank_heat_flow)
        if method == CpMethod.THREE_STEP:
            assert reference_data is not None
            ref_signal = self._endo_positive(
                np.asarray(reference_data["heat_flow"], dtype=np.float64),
                blank_heat_flow,
            )
            ref_temperature = np.asarray(
                reference_data.get("temperature", temperature), dtype=np.float64
            )
            ref_mass = float(reference_data["mass"])
            u_rel = self._three_step_relative_uncertainty(reference_data)
        else:
            u_rel = float(np.sqrt(U_HEAT_FLOW**2 + U_MASS**2 + U_HEATING_RATE**2))

        temps, cps, regions = [], [], []
        for step in steps:
            q_sample, t_start, t_end = self._step_heat(
                sample_signal, temperature, time, step
            )
            if method == CpMethod.SINGLE_STEP:
                cp = q_sample / sample_mass / (t_end - t_start)
            else:
                q_ref, r_start, r_end = self._step_heat(
                    ref_signal, ref_temperature, time, step
                )
                mean_ref_cp = self._mean_reference_cp(
                    reference_data, ref_temperature, r_start, r_end, step
                )
                # Q_ref = m_ref * mean Cp_ref * dT_ref
                cp = (
                    q_sample
                    / q_ref
                    * ref_mass
                    / sample_mass
                    * mean_ref_cp
                    * (r_end - r_start)
                    / (t_end - t_start)
                )
            temps.append((t_start + t_end) / 2)
            cps.append(cp)
            regions.append((step["start_idx"], step["end_idx"]))

        cp_array = np.array(cps)
        metadata: Dict[str, Any] = {
            "sample_mass": sample_mass,
            "heating_rate": heating_rate,
            "operation_mode": OperationMode.STEPPED.value,
            "blank_corrected": blank_heat_flow is not None,
            "step_temperatures": [
                (float(temperature[s["start_idx"]]), float(temperature[s["end_idx"]]))
                for s in steps
            ],
        }
        if method == CpMethod.THREE_STEP:
            metadata["reference_mass"] = ref_mass

        return self._result(
            np.array(temps),
            cp_array,
            np.abs(cp_array) * u_rel,
            method,
            OperationMode.STEPPED,
            metadata,
            stable_regions=regions,
        )

    def find_heating_steps(
        self, temperature: FloatArray, time: FloatArray
    ) -> List[Dict[str, int]]:
        """
        Find heating steps of a stepped program.

        Args:
            temperature: Temperature array (K)
            time: Time array (s)

        Returns:
            List of dicts with the indices of the preceding isotherm
            ('iso_start'), the start of the ramp ('start_idx'), the start
            ('end_idx') and end ('iso_end') of the following isotherm
        """
        segments = self.data_validator.detect_temperature_program(temperature, time)[
            "segments"
        ]
        steps = []
        for before, ramp, after in zip(segments, segments[1:], segments[2:]):
            if (
                before["type"] == "isothermal"
                and ramp["type"] == "heating"
                and after["type"] == "isothermal"
            ):
                steps.append(
                    {
                        "iso_start": int(before["start_idx"]),
                        "start_idx": int(ramp["start_idx"]),
                        "end_idx": int(after["start_idx"]),
                        "iso_end": int(after["end_idx"]),
                    }
                )
        return steps

    @staticmethod
    def _step_heat(
        signal: FloatArray,
        temperature: FloatArray,
        time: FloatArray,
        step: Dict[str, int],
    ) -> Tuple[float, float, float]:
        """
        Heat absorbed in a heating step (mJ) and the isotherm temperatures.

        The signal is referenced to the isothermal levels: the level of the
        preceding isotherm before the ramp and of the following isotherm
        after it, joined linearly over the ramp. The levels are the means of
        the last third of each isotherm, where the response has settled. The
        integral runs until the start of that last third.
        """
        pre = slice(
            step["start_idx"] - (step["start_idx"] - step["iso_start"]) // 3,
            step["start_idx"],
        )
        post_start = step["iso_end"] - (step["iso_end"] - step["end_idx"]) // 3
        post = slice(post_start, step["iso_end"])

        level_pre = float(np.mean(signal[pre]))
        level_post = float(np.mean(signal[post]))

        window = slice(step["start_idx"], post_start)
        t = time[window]
        t_ramp_end = time[step["end_idx"]]
        base = np.where(
            t < t_ramp_end,
            np.interp(t, [t[0], t_ramp_end], [level_pre, level_post]),
            level_post,
        )
        heat = float(trapezoid(signal[window] - base, t))
        return (
            heat,
            float(np.mean(temperature[pre])),
            float(np.mean(temperature[post])),
        )

    def _mean_reference_cp(
        self,
        reference_data: Optional[Dict[str, Any]],
        ref_temperature: FloatArray,
        t_start: float,
        t_end: float,
        step: Dict[str, int],
    ) -> float:
        """Mean reference Cp between two temperatures."""
        assert reference_data is not None
        if "cp" in reference_data:
            cp = np.asarray(reference_data["cp"], dtype=np.float64)
            window = slice(step["start_idx"], step["end_idx"] + 1)
            order = np.argsort(ref_temperature[window])
            grid = np.linspace(t_start, t_end, 200)
            values = np.interp(grid, ref_temperature[window][order], cp[window][order])
            return float(np.mean(values))
        material = reference_data.get("material", "sapphire")
        grid = np.linspace(t_start, t_end, 200)
        return float(np.mean(reference_cp(material, grid)))

    # ------------------------------------------------------------------
    # Temperature-modulated DSC
    # ------------------------------------------------------------------

    def _calculate_modulated_cp(
        self,
        temperature: FloatArray,
        heat_flow: FloatArray,
        sample_mass: float,
        heating_rate: float,
        time: FloatArray,
        modulation_period: float = 60.0,
        periods_per_window: int = 2,
        **kwargs: Any,
    ) -> CpResult:
        """
        Reversing Cp = A_q / (A_beta * m), from the first-harmonic amplitudes
        of heat flow (mW) and heating rate (K/s), in sliding windows of whole
        modulation periods (linear trends removed).
        """
        if modulation_period <= 0:
            raise ValueError("Modulation period must be positive")

        dt = float(np.median(np.diff(time)))
        window = int(round(periods_per_window * modulation_period / dt))
        if window < 8 or window > len(time):
            raise ValueError(
                "Data must cover whole modulation periods with enough points"
            )

        rate = np.gradient(temperature, time)
        omega = 2 * np.pi / modulation_period

        def amplitude(x: FloatArray, t: FloatArray) -> float:
            trend = np.polyval(np.polyfit(t, x, 1), t)
            residual = x - trend
            span = t[-1] - t[0]
            a = 2 / span * trapezoid(residual * np.sin(omega * t), t)
            b = 2 / span * trapezoid(residual * np.cos(omega * t), t)
            return float(np.hypot(a, b))

        step = max(window // (2 * periods_per_window), 1)
        temps, cps = [], []
        for start in range(0, len(time) - window + 1, step):
            sl = slice(start, start + window)
            a_rate = amplitude(rate[sl], time[sl])
            if a_rate <= 0:
                continue
            a_q = amplitude(heat_flow[sl], time[sl])
            temps.append(float(np.mean(temperature[sl])))
            cps.append(a_q / a_rate / sample_mass)

        if not cps:
            raise ValueError("No valid modulation windows found")

        cp_array = np.array(cps)
        u_rel = float(np.sqrt(2 * U_HEAT_FLOW**2 + U_MASS**2))
        return self._result(
            np.array(temps),
            cp_array,
            cp_array * u_rel,
            CpMethod.MODULATED,
            OperationMode.CONTINUOUS,
            {
                "sample_mass": sample_mass,
                "heating_rate": heating_rate,
                "modulation_period": modulation_period,
                "periods_per_window": periods_per_window,
                "operation_mode": OperationMode.CONTINUOUS.value,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_length(array: FloatArray, temperature: FloatArray, name: str) -> None:
        if len(array) != len(temperature):
            raise ValueError(f"{name} and temperature arrays must have same length")

    def _endo_positive(
        self, heat_flow: FloatArray, blank_heat_flow: Optional[FloatArray]
    ) -> FloatArray:
        """Blank-corrected heat flow with endothermic heat flow positive."""
        corrected = (
            heat_flow if blank_heat_flow is None else heat_flow - blank_heat_flow
        )
        return -corrected if self.exo_up else corrected

    @staticmethod
    def _reference_cp_at(
        reference_data: Dict[str, Any], temperature: FloatArray
    ) -> FloatArray:
        """Reference Cp at the measurement temperatures."""
        if "cp" in reference_data:
            return np.asarray(reference_data["cp"], dtype=np.float64)
        ref_temperature = np.asarray(
            reference_data.get("temperature", temperature), dtype=np.float64
        )
        return reference_cp(reference_data.get("material", "sapphire"), ref_temperature)

    @staticmethod
    def _three_step_relative_uncertainty(reference_data: Dict[str, Any]) -> float:
        """Relative uncertainty of the three-step ratio method."""
        material = reference_data.get("material", "sapphire").lower()
        u_ref = reference_data.get(
            "relative_uncertainty",
            REFERENCE_MATERIALS.get(material, {}).get("relative_uncertainty", 0.01),
        )
        # Two heat flow differences, two masses, reference Cp
        return float(np.sqrt(2 * U_HEAT_FLOW**2 + 2 * U_MASS**2 + u_ref**2))

    def _result(
        self,
        temperature: FloatArray,
        cp: FloatArray,
        uncertainty: FloatArray,
        method: CpMethod,
        operation_mode: OperationMode,
        metadata: Dict[str, Any],
        stable_regions: Optional[List[Tuple[int, int]]] = None,
    ) -> CpResult:
        return CpResult(
            temperature=np.asarray(temperature, dtype=np.float64),
            specific_heat=np.asarray(cp, dtype=np.float64),
            uncertainty=np.asarray(uncertainty, dtype=np.float64),
            method=method,
            quality_metrics=self._calculate_quality_metrics(
                temperature, cp, uncertainty
            ),
            metadata=metadata,
            operation_mode=operation_mode,
            stable_regions=stable_regions,
        )

    def _calculate_quality_metrics(
        self,
        temperature: FloatArray,
        cp: FloatArray,
        uncertainty: FloatArray,
    ) -> Dict[str, float]:
        """Quality metrics of a Cp curve."""
        temp = np.ravel(temperature)
        cp_vals = np.ravel(cp)
        unc = np.ravel(uncertainty)
        metrics: Dict[str, float] = {}

        relative = unc / np.maximum(np.abs(cp_vals), np.finfo(float).tiny)
        metrics["avg_uncertainty"] = float(np.mean(relative))
        metrics["max_uncertainty"] = float(np.max(relative))

        if len(cp_vals) > 2:
            noise = float(np.std(np.diff(cp_vals)))
            spread = float(np.ptp(cp_vals))
            metrics["snr"] = spread / noise if noise > 0 else float("inf")
            metrics["smoothness"] = float(1 / (1 + np.std(np.gradient(cp_vals))))
        else:
            metrics["snr"] = float("nan")
            metrics["smoothness"] = float("nan")

        if len(cp_vals) > 1 and np.ptp(temp) > 0:
            fit = stats.linregress(temp, cp_vals)
            metrics.update(
                {
                    "slope": float(fit.slope),
                    "intercept": float(fit.intercept),
                    "r_squared": float(fit.rvalue**2),
                    "std_error": float(fit.stderr),
                }
            )
        return metrics

    def _apply_calibration(self, result: CpResult) -> CpResult:
        """Multiply by calibration factors interpolated at the result
        temperatures (only single-step results: the ratio methods do not
        depend on the instrument sensitivity)."""
        if self.calibration_data is None or result.method != CpMethod.SINGLE_STEP:
            return result

        low, high = self.calibration_data.valid_range
        if np.min(result.temperature) < low or np.max(result.temperature) > high:
            raise ValueError("Temperature range outside calibration validity")

        order = np.argsort(self.calibration_data.temperature)
        cal_temp = self.calibration_data.temperature[order]
        factors = np.interp(
            result.temperature,
            cal_temp,
            self.calibration_data.calibration_factors[order],
        )
        factor_unc = np.interp(
            result.temperature, cal_temp, self.calibration_data.uncertainty[order]
        )

        calibrated_cp = result.specific_heat * factors
        calibrated_uncertainty = np.sqrt(
            (result.uncertainty * factors) ** 2
            + (result.specific_heat * factor_unc) ** 2
        )

        return CpResult(
            temperature=result.temperature,
            specific_heat=calibrated_cp,
            uncertainty=calibrated_uncertainty,
            method=result.method,
            quality_metrics=self._calculate_quality_metrics(
                result.temperature, calibrated_cp, calibrated_uncertainty
            ),
            metadata={
                **result.metadata,
                "calibration_applied": True,
                "reference_material": self.calibration_data.reference_material,
            },
            operation_mode=result.operation_mode,
            stable_regions=result.stable_regions,
        )

    def validate_reference_data(
        self,
        reference_data: Dict[str, Any],
        expected_length: Optional[int] = None,
    ) -> bool:
        """
        Validate reference measurement data for the three-step method.

        Args:
            reference_data: Dictionary with 'heat_flow', 'mass' and either
                'material' (default 'sapphire') or 'cp'
            expected_length: Required length of the arrays (sample run)

        Returns:
            True if valid, raises ValueError otherwise
        """
        missing = [f for f in ("heat_flow", "mass") if f not in reference_data]
        if missing:
            raise ValueError(f"Missing required reference data fields: {missing}")

        mass = reference_data["mass"]
        if not isinstance(mass, (int, float)) or mass <= 0:
            raise ValueError("Reference mass must be a positive number")

        material = reference_data.get("material", "sapphire")
        if "cp" not in reference_data and material.lower() not in REFERENCE_MATERIALS:
            raise ValueError(f"Unknown reference material: {material}")

        for field in ("heat_flow", "cp", "temperature"):
            if field in reference_data:
                array = np.asarray(reference_data[field])
                if expected_length is not None and len(array) != expected_length:
                    raise ValueError(
                        f"Reference {field} must have the same length as the "
                        "sample data"
                    )
        return True
