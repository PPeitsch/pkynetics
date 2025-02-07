"""Heat capacity calculation and analysis module."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .signal_stability import SignalStabilityDetector
from .types import CalibrationData, CpMethod, CpResult, OperationMode, StabilityMethod
from .utilities import DataValidator, SignalProcessor


class CpCalculator:
    """Class for specific heat capacity calculations and analysis."""

    def __init__(
            self,
            signal_processor: Optional[SignalProcessor] = None,
            stability_detector: Optional[SignalStabilityDetector] = None,
            data_validator: Optional[DataValidator] = None,
    ):
        """Initialize calculator with optional components."""
        self.signal_processor = signal_processor or SignalProcessor()
        self.stability_detector = stability_detector or SignalStabilityDetector()
        self.data_validator = data_validator or DataValidator()
        self.calibration_data: Optional[CalibrationData] = None
        self._reference_data = self._load_reference_data()

    def calculate_cp(
            self,
            temperature: NDArray[np.float64],
            heat_flow: NDArray[np.float64],
            sample_mass: float,
            heating_rate: float,
            method: Union[CpMethod, str] = CpMethod.THREE_STEP,
            operation_mode: Union[OperationMode, str] = OperationMode.CONTINUOUS,
            stability_method: Union[StabilityMethod, str] = StabilityMethod.STATISTICAL,
            reference_data: Optional[Dict] = None,
            use_calibration: bool = True,
            **kwargs,
    ) -> CpResult:
        """
        Calculate specific heat capacity.

        Args:
            temperature: Temperature array (K)
            heat_flow: Heat flow array (mW)
            sample_mass: Sample mass (mg)
            heating_rate: Heating rate (K/min)
            method: Calculation method
            operation_mode: Measurement mode
            stability_method: Method for detecting stable regions
            reference_data: Optional reference measurement data
            use_calibration: Whether to apply calibration
            **kwargs: Additional method-specific parameters

        Returns:
            CpResult object with calculated heat capacity

        Raises:
            ValueError: If inputs are invalid or method is unsupported
        """
        # Validate inputs
        self.data_validator.validate_temperature_data(temperature)
        self.data_validator.validate_heat_flow_data(heat_flow, temperature)

        if sample_mass <= 0:
            raise ValueError("Sample mass must be positive")
        if heating_rate == 0:
            raise ValueError("Heating rate cannot be zero")

        # Convert string enums to proper types
        if isinstance(method, str):
            method = CpMethod(method)
        if isinstance(operation_mode, str):
            operation_mode = OperationMode(operation_mode)
        if isinstance(stability_method, str):
            stability_method = StabilityMethod(stability_method)

        # Select calculation method
        if method == CpMethod.THREE_STEP:
            if reference_data is None:
                raise ValueError("Reference data required for three-step method")
            result = self._calculate_three_step_cp(
                temperature,
                heat_flow,
                sample_mass,
                heating_rate,
                reference_data,
                operation_mode,
                stability_method,
                **kwargs,
            )
        elif method == CpMethod.SINGLE_STEP:
            result = self._calculate_single_step_cp(
                temperature,
                heat_flow,
                sample_mass,
                heating_rate,
                operation_mode,
                stability_method,
                **kwargs,
            )
        elif method == CpMethod.MODULATED:
            result = self._calculate_modulated_cp(
                temperature,
                heat_flow,
                sample_mass,
                heating_rate,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported Cp calculation method: {method}")

        # Apply calibration if requested
        if use_calibration and self.calibration_data is not None:
            result = self._apply_calibration(result)

        return result

    def calibrate(
            self,
            temperature: NDArray[np.float64],
            heat_flow: NDArray[np.float64],
            sample_mass: float,
            heating_rate: float,
            reference_material: str,
            operation_mode: Union[OperationMode, str] = OperationMode.CONTINUOUS,
            stability_method: Union[StabilityMethod, str] = StabilityMethod.STATISTICAL,
    ) -> CalibrationData:
        """
        Perform DSC calibration using reference material.

        Args:
            temperature: Temperature array (K)
            heat_flow: Heat flow array (mW)
            sample_mass: Sample mass (mg)
            heating_rate: Heating rate (K/min)
            reference_material: Name of reference material
            operation_mode: Measurement mode
            stability_method: Method for detecting stable regions

        Returns:
            CalibrationData object

        Raises:
            ValueError: If reference material is unknown or data is invalid
        """
        # Get reference Cp data
        ref_data = self._get_reference_cp(reference_material)
        if ref_data is None:
            raise ValueError(f"Unknown reference material: {reference_material}")

        # Calculate measured Cp without calibration
        measured_result = self.calculate_cp(
            temperature,
            heat_flow,
            sample_mass,
            heating_rate,
            method=CpMethod.SINGLE_STEP,
            operation_mode=operation_mode,
            stability_method=stability_method,
            use_calibration=False,
        )

        # Interpolate reference Cp to measurement temperatures
        ref_cp_interp = np.interp(
            measured_result.temperature,
            ref_data["temperature"],
            ref_data["cp"],
        )

        # Calculate calibration factors
        factors = ref_cp_interp / measured_result.specific_heat

        # Calculate uncertainty
        uncertainty = self._calculate_calibration_uncertainty(
            measured_result.specific_heat,
            ref_cp_interp,
            measured_result.uncertainty,
            ref_data.get("uncertainty", np.zeros_like(ref_data["cp"]))
        )

        # Create and store calibration data
        self.calibration_data = CalibrationData(
            reference_material=reference_material,
            temperature=measured_result.temperature,
            measured_cp=measured_result.specific_heat,
            reference_cp=ref_cp_interp,
            calibration_factors=factors,
            uncertainty=uncertainty,
            valid_range=(float(np.min(temperature)), float(np.max(temperature))),
            metadata={
                "sample_mass": sample_mass,
                "heating_rate": heating_rate,
                "operation_mode": operation_mode.value,
            }
        )

        return self.calibration_data

    def _calculate_three_step_cp(
            self,
            temperature: NDArray[np.float64],
            heat_flow: NDArray[np.float64],
            sample_mass: float,
            heating_rate: float,
            reference_data: Dict,
            operation_mode: OperationMode,
            stability_method: StabilityMethod,
            **kwargs,
    ) -> CpResult:
        """
        Calculate Cp using three-step method.

        Args:
            temperature: Temperature array (K)
            heat_flow: Heat flow array (mW)
            sample_mass: Sample mass (mg)
            heating_rate: Heating rate (K/min)
            reference_data: Dictionary containing reference measurement data
            operation_mode: Measurement operation mode
            stability_method: Method for detecting stable regions
            **kwargs: Additional parameters

        Returns:
            CpResult containing calculated heat capacity and analysis metrics

        Raises:
            ValueError: If required reference data is missing or arrays have inconsistent shapes
        """
        # Ensure arrays are 1D
        temperature = np.asarray(temperature).ravel()
        heat_flow = np.asarray(heat_flow).ravel()
        ref_temp = np.asarray(reference_data["temperature"]).ravel()
        ref_heat_flow = np.asarray(reference_data["heat_flow"]).ravel()
        ref_cp = np.asarray(reference_data["cp"]).ravel()

        # Validate array lengths
        if not all(len(arr) == len(temperature) for arr in [heat_flow, ref_temp, ref_heat_flow, ref_cp]):
            raise ValueError("All data arrays must have the same length")

        # Validate reference data
        required_fields = ["temperature", "heat_flow", "mass", "cp"]
        if not all(field in reference_data for field in required_fields):
            raise ValueError(f"Missing required reference data: {required_fields}")

        # Find stable regions if in stepped mode
        stable_regions = None
        if operation_mode == OperationMode.STEPPED:
            stable_regions = self.stability_detector.find_stable_regions(
                heat_flow, temperature, method=stability_method
            )

        if stable_regions:
            # Calculate Cp for stable regions
            cp_values = []
            temps = []
            uncertainties = []

            for start_idx, end_idx in stable_regions:
                region_slice = slice(start_idx, end_idx)

                # Calculate regional averages
                temp = np.mean(temperature[region_slice])
                sample_signal = np.mean(heat_flow[region_slice])
                ref_signal = np.mean(ref_heat_flow[region_slice])
                ref_cp_value = np.mean(ref_cp[region_slice])

                # Calculate Cp
                cp = self._calculate_regional_cp(
                    sample_signal,
                    ref_signal,
                    sample_mass,
                    reference_data["mass"],
                    ref_cp_value,
                )

                # Calculate uncertainty
                uncertainty = self._calculate_three_step_uncertainty(
                    sample_signal,
                    ref_signal,
                    ref_cp_value,
                    sample_mass,
                    reference_data["mass"],
                )

                cp_values.append(cp)
                temps.append(temp)
                uncertainties.append(uncertainty)

            # Convert to arrays
            cp_array = np.array(cp_values)
            temp_array = np.array(temps)
            uncertainty_array = np.array(uncertainties)

        else:
            # Continuous mode calculations
            cp_array = self._calculate_continuous_cp(
                heat_flow,
                ref_heat_flow,
                sample_mass,
                reference_data["mass"],
                ref_cp,
            )
            temp_array = temperature
            uncertainty_array = self._calculate_three_step_uncertainty(
                heat_flow,
                ref_heat_flow,
                ref_cp,
                sample_mass,
                reference_data["mass"],
            )

        # Ensure all output arrays are 1D
        temp_array = np.asarray(temp_array).ravel()
        cp_array = np.asarray(cp_array).ravel()
        uncertainty_array = np.asarray(uncertainty_array).ravel()

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            temp_array, cp_array, uncertainty_array
        )

        return CpResult(
            temperature=temp_array,
            specific_heat=cp_array,
            uncertainty=uncertainty_array,
            method=CpMethod.THREE_STEP,
            quality_metrics=quality_metrics,
            metadata={
                "sample_mass": sample_mass,
                "reference_mass": reference_data["mass"],
                "heating_rate": heating_rate,
                "operation_mode": operation_mode.value,
            },
            operation_mode=operation_mode,
            stable_regions=stable_regions,
        )

    def _calculate_single_step_cp(
            self,
            temperature: NDArray[np.float64],
            heat_flow: NDArray[np.float64],
            sample_mass: float,
            heating_rate: float,
            operation_mode: OperationMode,
            stability_method: StabilityMethod,
            **kwargs,
    ) -> CpResult:
        """Calculate Cp using single-step method."""
        # Find stable regions if in stepped mode
        stable_regions = None
        if operation_mode == OperationMode.STEPPED:
            stable_regions = self.stability_detector.find_stable_regions(
                heat_flow, temperature, method=stability_method
            )

        if stable_regions:
            # Calculate Cp for stable regions
            cp_values = []
            temps = []
            uncertainties = []

            for start_idx, end_idx in stable_regions:
                region_slice = slice(start_idx, end_idx)

                # Calculate regional averages
                temp = np.mean(temperature[region_slice])
                heat_flow_avg = np.mean(heat_flow[region_slice])

                # Calculate Cp
                cp = heat_flow_avg / (sample_mass * heating_rate)

                # Calculate uncertainty
                uncertainty = self._calculate_single_step_uncertainty(
                    heat_flow_avg, heating_rate, sample_mass
                )

                cp_values.append(cp)
                temps.append(temp)
                uncertainties.append(uncertainty)

            # Convert to arrays
            cp_array = np.array(cp_values)
            temp_array = np.array(temps)
            uncertainty_array = np.array(uncertainties)

        else:
            # Continuous mode calculations
            cp_array = heat_flow / (sample_mass * heating_rate)
            temp_array = temperature
            uncertainty_array = self._calculate_single_step_uncertainty(
                heat_flow, heating_rate, sample_mass
            )

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            temp_array, cp_array, uncertainty_array
        )

        return CpResult(
            temperature=temp_array,
            specific_heat=cp_array,
            uncertainty=uncertainty_array,
            method=CpMethod.SINGLE_STEP,
            quality_metrics=quality_metrics,
            metadata={
                "sample_mass": sample_mass,
                "heating_rate": heating_rate,
                "operation_mode": operation_mode.value,
            },
            operation_mode=operation_mode,
            stable_regions=stable_regions,
        )

    def _calculate_modulated_cp(
            self,
            temperature: NDArray[np.float64],
            heat_flow: NDArray[np.float64],
            sample_mass: float,
            heating_rate: float,
            modulation_period: float = 60.0,
            modulation_amplitude: float = 0.5,
            **kwargs,
    ) -> CpResult:
        """Calculate Cp using modulated DSC method."""
        # Calculate modulation parameters
        omega = 2 * np.pi / modulation_period
        heating_rate_mod = omega * modulation_amplitude

        # Separate reversing and non-reversing components
        reversing_cp = heat_flow / (sample_mass * heating_rate_mod)

        # Calculate phase angle and uncertainty
        phase_angle = np.arctan2(
            np.gradient(heat_flow, temperature),
            heat_flow
        )

        uncertainty = self._calculate_modulated_uncertainty(
            heat_flow, modulation_amplitude, modulation_period, sample_mass
        )

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            temperature, reversing_cp, uncertainty
        )
        quality_metrics["phase_angle"] = float(np.mean(phase_angle))
        quality_metrics["modulation_quality"] = float(
            1 - np.std(phase_angle) / np.pi
        )

        return CpResult(
            temperature=temperature,
            specific_heat=reversing_cp,
            uncertainty=uncertainty,
            method=CpMethod.MODULATED,
            quality_metrics=quality_metrics,
            metadata={
                "sample_mass": sample_mass,
                "heating_rate": heating_rate,
                "modulation_period": modulation_period,
                "modulation_amplitude": modulation_amplitude,
                "operation_mode": OperationMode.CONTINUOUS.value,
            },
            operation_mode=OperationMode.CONTINUOUS,
        )

    def _calculate_three_step_uncertainty(
            self,
            sample_signal: NDArray[np.float64],
            ref_signal: NDArray[np.float64],
            ref_cp: Union[float, NDArray[np.float64]],
            sample_mass: float,
            ref_mass: float,
    ) -> NDArray[np.float64]:
        """Calculate uncertainty for three-step method."""
        # Component uncertainties
        u_signal = 0.01  # 1% signal uncertainty
        u_mass = 0.001  # 0.1% mass uncertainty
        u_ref = 0.02  # 2% reference Cp uncertainty

        # Combined uncertainty using propagation of errors
        u_combined = np.sqrt(
            (u_signal * sample_signal / ref_signal) ** 2 +
            u_signal ** 2 +
            (u_mass * sample_mass) ** 2 +
            (u_mass * ref_mass) ** 2 +
            (u_ref * ref_cp) ** 2
        )

        return u_combined

    def _calculate_single_step_uncertainty(
            self,
            heat_flow: Union[float, NDArray[np.float64]],
            heating_rate: float,
            sample_mass: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate uncertainty for single-step method."""
        # Component uncertainties
        u_flow = 0.01  # 1% heat flow uncertainty
        u_rate = 0.01  # 1% heating rate uncertainty
        u_mass = 0.001  # 0.1% mass uncertainty

        # Combined uncertainty
        u_combined = np.sqrt(
            (u_flow * heat_flow) ** 2 +
            (u_rate * heating_rate) ** 2 +
            (u_mass * sample_mass) ** 2
        )

        return u_combined

    def _calculate_modulated_uncertainty(
            self,
            heat_flow: NDArray[np.float64],
            amplitude: float,
            period: float,
            sample_mass: float,
    ) -> NDArray[np.float64]:
        """Calculate uncertainty for modulated DSC method."""
        # Component uncertainties
        u_flow = 0.02  # 2% heat flow uncertainty
        u_amp = 0.01  # 1% amplitude uncertainty
        u_period = 0.01  # 1% period uncertainty
        u_mass = 0.001  # 0.1% mass uncertainty

        # Calculate modulation parameters uncertainty
        omega = 2 * np.pi / period
        u_omega = omega * u_period

        # Combined uncertainty
        u_combined = np.sqrt(
            (u_flow * heat_flow) ** 2 +
            (u_amp * amplitude) ** 2 +
            (u_omega * period) ** 2 +
            (u_mass * sample_mass) ** 2
        )

        return u_combined

    def _calculate_quality_metrics(
            self,
            temperature: NDArray[np.float64],
            cp: NDArray[np.float64],
            uncertainty: NDArray[np.float64],
    ) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        metrics = {}

        # Debug original shapes
        print("\nDebug: Original array shapes:")
        print(f"Temperature: shape={temperature.shape}, type={type(temperature)}")
        print(f"Cp: shape={cp.shape}, type={type(cp)}")
        print(f"Uncertainty: shape={uncertainty.shape}, type={type(uncertainty)}")

        # Ensure arrays are 1D
        temp = temperature.ravel()
        cp_vals = cp.ravel()
        unc = uncertainty.ravel()

        # Debug after ravel
        print("\nDebug: Array lengths after ravel:")
        print(f"Temperature: len={len(temp)}, shape={temp.shape}")
        print(f"Cp: len={len(cp_vals)}, shape={cp_vals.shape}")
        print(f"Uncertainty: len={len(unc)}, shape={unc.shape}")

        # Print some values
        print("\nDebug: First few values of each array:")
        print(f"Temperature: {temp[:5]}")
        print(f"Cp: {cp_vals[:5]}")
        print(f"Uncertainty: {unc[:5]}")

        if not (len(temp) == len(cp_vals) == len(unc)):
            raise ValueError(
                f"All input arrays must have the same length. Got: temperature={len(temp)}, cp={len(cp_vals)}, uncertainty={len(unc)}")

        # Signal-to-noise ratio
        noise = np.std(np.diff(cp_vals))
        signal = np.ptp(cp_vals)
        metrics["snr"] = float(signal / noise if noise > 0 else np.inf)

        # Relative uncertainty
        metrics["avg_uncertainty"] = float(np.mean(unc / cp_vals))
        metrics["max_uncertainty"] = float(np.max(unc / cp_vals))

        # Smoothness metric
        try:
            dcp_dt = np.gradient(cp_vals)
            smoothness = 1 / (1 + np.std(dcp_dt))
            metrics["smoothness"] = float(smoothness)
        except Exception as e:
            print(f"Warning: Smoothness calculation failed: {str(e)}")
            metrics["smoothness"] = 0.0

        # Linear fit metrics
        try:
            # Make sure arrays are properly shaped for linregress
            if len(temp) > 1 and len(cp_vals) > 1:
                slope, intercept, r_value, _, stderr = stats.linregress(temp, cp_vals)
                metrics.update({
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_value ** 2),
                    "std_error": float(stderr),
                })
            else:
                raise ValueError("Insufficient data points for linear regression")
        except Exception as e:
            print(f"Warning: Linear regression failed: {str(e)}")
            metrics.update({
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "std_error": 0.0,
            })

        # Overall quality score (0 to 1)
        valid_metrics = [
            metrics["snr"] / (metrics["snr"] + 1),
            1 - metrics["avg_uncertainty"],
            metrics["smoothness"],
            metrics["r_squared"]
        ]
        metrics["quality_score"] = float(np.mean(valid_metrics))

        return metrics

    def _load_reference_data(self) -> Dict:
        """Load reference material Cp data."""
        # Example reference data (should be loaded from database)
        # Sapphire (Al2O3) reference data
        temp_range = np.linspace(200, 800, 601)

        sapphire = {
            "temperature": temp_range,
            "cp": 1.0289 + 2.3506e-4 * temp_range + 1.6818e-7 * temp_range ** 2,
            "uncertainty": 0.02,  # 2% uncertainty
            "valid_range": (200, 800),
            "molecular_weight": 101.96,  # g/mol
            "purity": 0.9999,
        }

        # Zinc reference data
        zinc = {
            "temperature": np.linspace(290, 450, 161),
            "cp": 0.3889 + 2.7247e-4 * temp_range,
            "uncertainty": 0.015, "valid_range": (290, 450),
            "molecular_weight": 65.38,  # g/mol
            "purity": 0.9999,
            "melting_point": 419.53,  # K
        }

        return {
            "sapphire": sapphire,
            "zinc": zinc,
        }

    def _calculate_calibration_uncertainty(
            self,
            measured_cp: NDArray[np.float64],
            reference_cp: NDArray[np.float64],
            measurement_uncertainty: NDArray[np.float64],
            reference_uncertainty: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate comprehensive uncertainty in calibration factors."""
        # Relative uncertainties
        u_meas_rel = measurement_uncertainty / measured_cp
        u_ref_rel = reference_uncertainty / reference_cp

        # Systematic uncertainty components
        u_temp = 0.005  # 0.5% temperature uncertainty
        u_cal = 0.01  # 1% calibration stability

        # Combined relative uncertainty
        u_combined = np.sqrt(
            u_meas_rel ** 2 +
            u_ref_rel ** 2 +
            u_temp ** 2 +
            u_cal ** 2
        )

        return u_combined * measured_cp

    def _apply_calibration(
            self,
            result: CpResult,
    ) -> CpResult:
        """Apply calibration correction to Cp results."""
        if self.calibration_data is None:
            return result

        # Check temperature range validity
        if (np.min(result.temperature) < self.calibration_data.valid_range[0] or
                np.max(result.temperature) > self.calibration_data.valid_range[1]):
            raise ValueError("Temperature range outside calibration validity")

        # Interpolate calibration factors
        factors = np.interp(
            result.temperature,
            self.calibration_data.temperature,
            self.calibration_data.calibration_factors,
        )

        # Apply calibration
        calibrated_cp = result.specific_heat * factors

        # Update uncertainty
        cal_uncertainty = np.interp(
            result.temperature,
            self.calibration_data.temperature,
            self.calibration_data.uncertainty,
        )

        calibrated_uncertainty = np.sqrt(
            result.uncertainty ** 2 +
            (calibrated_cp * cal_uncertainty) ** 2
        )

        # Create new result with calibrated values
        return CpResult(
            temperature=result.temperature,
            specific_heat=calibrated_cp,
            uncertainty=calibrated_uncertainty,
            method=result.method,
            quality_metrics=result.quality_metrics,
            metadata={
                **result.metadata,
                "calibration_applied": True,
                "reference_material": self.calibration_data.reference_material,
            },
            operation_mode=result.operation_mode,
            stable_regions=result.stable_regions,
        )

    def _calculate_regional_cp(
            self,
            sample_signal: float,
            ref_signal: float,
            sample_mass: float,
            ref_mass: float,
            ref_cp: float,
    ) -> float:
        """Calculate Cp for a single region in stepped mode."""
        return (sample_signal / ref_signal) * (ref_mass / sample_mass) * ref_cp

    def _calculate_continuous_cp(
            self,
            sample_signal: NDArray[np.float64],
            ref_signal: NDArray[np.float64],
            sample_mass: float,
            ref_mass: float,
            ref_cp: Union[float, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Calculate Cp for continuous measurement."""
        return (sample_signal / ref_signal) * (ref_mass / sample_mass) * ref_cp

    def _get_reference_cp(self, material: str) -> Optional[Dict]:
        """Get reference Cp data for calibration material."""
        return self._reference_data.get(material.lower())

    def validate_reference_data(
            self,
            reference_data: Dict,
            required_fields: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate reference measurement data.

        Args:
            reference_data: Dictionary containing reference data
            required_fields: List of required field names

        Returns:
            True if valid, raises ValueError otherwise
        """
        if required_fields is None:
            required_fields = ["temperature", "heat_flow", "mass", "cp"]

        # Check required fields
        missing = [f for f in required_fields if f not in reference_data]
        if missing:
            raise ValueError(f"Missing required reference data fields: {missing}")

        # Validate arrays
        arrays = [reference_data[f] for f in ["temperature", "heat_flow", "cp"]]
        if not all(isinstance(arr, np.ndarray) for arr in arrays):
            raise TypeError("Temperature, heat flow, and cp must be numpy arrays")
        if not all(len(arr) == len(arrays[0]) for arr in arrays):
            raise ValueError("All arrays must have the same length")

        # Validate numeric values
        if not isinstance(reference_data["mass"], (int, float)) or reference_data["mass"] <= 0:
            raise ValueError("Reference mass must be a positive number")

        return True
