"""Specific heat capacity analysis module for DSC measurements."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .types import CalibrationData, CpMethod, CpResult, OperationMode, StabilityMethod


class CpCalculator:
    """Class for specific heat capacity calculations."""

    def __init__(self):
        """Initialize Cp calculator."""
        self.calibration_data: Optional[CalibrationData] = None
        self._reference_data = self._load_reference_data()

    def _get_heat_flow(
        self, data: Dict[str, NDArray[np.float64]], required: bool = True
    ) -> Optional[NDArray[np.float64]]:
        """Get heat flow data from dictionary."""
        for key in ["heat_flow", "sample_heat_flow", "total_heat_flow"]:
            hf = data.get(key)
            if hf is not None:
                return hf
        if required:
            raise KeyError("No heat flow data found in input data")
        return None

    def calculate_cp(
        self,
        sample_data: Dict[str, NDArray[np.float64]],
        method: Union[CpMethod, str] = CpMethod.THREE_STEP,
        operation_mode: Union[OperationMode, str] = OperationMode.CONTINUOUS,
        stability_method: Union[StabilityMethod, str] = StabilityMethod.BASIC,
        use_calibration: bool = True,
        **kwargs,
    ) -> CpResult:
        """
        Calculate specific heat capacity.

        Args:
            sample_data: Dictionary containing measurement data
            method: Heat capacity calculation method
            operation_mode: Measurement operation mode (continuous or stepped)
            stability_method: Method for detecting stable regions in stepped mode
            use_calibration: Whether to apply calibration correction
            **kwargs: Additional parameters

        Returns:
            CpResult object containing calculated heat capacity

        Raises:
            ValueError: If method is invalid or required data is missing
        """
        # Validate method types
        if isinstance(method, str):
            try:
                method = CpMethod(method)
            except ValueError:
                raise ValueError(f"Invalid calculation method: {method}")

        if isinstance(operation_mode, str):
            try:
                operation_mode = OperationMode(operation_mode)
            except ValueError:
                raise ValueError(f"Invalid operation mode: {operation_mode}")

        if isinstance(stability_method, str):
            try:
                stability_method = StabilityMethod(stability_method)
            except ValueError:
                raise ValueError(f"Invalid stability method: {stability_method}")

        # Validate required fields
        required = ["temperature"]
        if not all(field in sample_data for field in required):
            raise ValueError(
                f"Missing required fields: {[f for f in required if f not in sample_data]}"
            )

        # Dictionary of calculation methods
        calculation_methods = {
            CpMethod.THREE_STEP: self._calculate_three_step_cp,
            CpMethod.SINGLE_STEP: self._calculate_single_step_cp,
            CpMethod.MODULATED: self._calculate_modulated_cp,
        }

        if method not in calculation_methods:
            raise ValueError(f"Unknown Cp calculation method: {method}")

        # Add parameters to kwargs
        kwargs["operation_mode"] = operation_mode
        kwargs["stability_method"] = stability_method
        kwargs["use_calibration"] = use_calibration

        result = calculation_methods[method](sample_data, **kwargs)

        return result

    def calibrate(
        self,
        reference_data: Dict[str, NDArray[np.float64]],
        reference_material: str,
        operation_mode: Union[OperationMode, str] = OperationMode.CONTINUOUS,
        stability_method: Union[StabilityMethod, str] = StabilityMethod.BASIC,
    ) -> CalibrationData:
        """
        Perform DSC calibration using reference material.

        Args:
            reference_data: Measurement data for reference material
            reference_material: Name of reference material
            operation_mode: Measurement operation mode
            stability_method: Method for detecting stable regions

        Returns:
            CalibrationData object
        """
        # Get reference Cp data
        ref_cp = self._get_reference_cp(reference_material)

        # Calculate measured Cp without applying calibration
        measured_result = self.calculate_cp(
            reference_data,
            method=CpMethod.SINGLE_STEP,
            operation_mode=operation_mode,
            stability_method=stability_method,
            use_calibration=False,
        )

        # Interpolate reference Cp to measurement temperatures
        ref_cp_interp = np.interp(
            measured_result.temperature, ref_cp["temperature"], ref_cp["cp"]
        )

        # Calculate calibration factors
        factors = ref_cp_interp / measured_result.specific_heat

        # Calculate uncertainty in calibration
        uncertainty = self._calculate_calibration_uncertainty(
            measured_result.specific_heat, ref_cp_interp, measured_result.uncertainty
        )

        # Create calibration data
        self.calibration_data = CalibrationData(
            reference_material=reference_material,
            temperature=measured_result.temperature,
            measured_cp=measured_result.specific_heat,
            reference_cp=ref_cp_interp,
            calibration_factors=factors,
            uncertainty=uncertainty,
            valid_range=(
                float(np.min(measured_result.temperature)),
                float(np.max(measured_result.temperature)),
            ),
        )

        return self.calibration_data

    def _load_reference_data(self) -> Dict[str, Dict[str, NDArray[np.float64]]]:
        """Load reference material Cp data."""
        # Example reference data (should be loaded from database)
        sapphire_temp = np.linspace(200, 800, 601)
        sapphire_cp = 1.0289 + 2.3506e-4 * sapphire_temp + 1.6818e-7 * sapphire_temp**2

        return {
            "sapphire": {
                "temperature": sapphire_temp,
                "cp": sapphire_cp,
                "uncertainty": 0.02 * sapphire_cp,  # 2% uncertainty
            }
        }

    def _calculate_three_step_cp(
        self,
        data: Dict[str, NDArray[np.float64]],
        operation_mode: OperationMode = OperationMode.CONTINUOUS,
        stability_method: StabilityMethod = StabilityMethod.BASIC,
        **kwargs,
    ) -> CpResult:
        """
        Calculate Cp using three-step method.

        Args:
            data: Dictionary containing sample, baseline, and reference runs
            operation_mode: Measurement operation mode
            stability_method: Method for detecting stable regions
            **kwargs: Additional parameters

        Returns:
            CpResult object
        """
        # Extract required data
        temp = data["temperature"]
        sample_hf = data["sample_heat_flow"]
        ref_hf = data.get("reference_heat_flow")
        baseline_hf = data.get("baseline_heat_flow")
        time = data.get("time")

        if ref_hf is None or baseline_hf is None:
            raise ValueError(
                "Reference and baseline runs required for three-step method"
            )

        if time is None:
            raise ValueError("Time data required for analysis")

        # Calculate stable regions if in stepped mode
        stable_regions = None
        if operation_mode == OperationMode.STEPPED:
            stable_regions = self._detect_stable_regions(
                temp, sample_hf, time, stability_method, **kwargs
            )

        # Correct heat flows
        sample_corr = sample_hf - baseline_hf
        ref_corr = ref_hf - baseline_hf

        # Calculate Cp for appropriate regions
        if operation_mode == OperationMode.STEPPED and stable_regions:
            cp_values = []
            temps = []
            uncertainties = []

            for start_idx, end_idx in stable_regions:
                # Calculate averages for stable region
                sample_mean = np.mean(sample_corr[start_idx:end_idx])
                ref_mean = np.mean(ref_corr[start_idx:end_idx])
                temp_mean = np.mean(temp[start_idx:end_idx])

                # Calculate Cp for this region
                cp = (sample_mean * data["reference_mass"] * data["reference_cp"]) / (
                    ref_mean * data["sample_mass"]
                )

                cp_values.append(cp)
                temps.append(temp_mean)

                # Calculate uncertainty for this region
                uncertainty = self._calculate_three_step_uncertainty(
                    sample_mean,
                    ref_mean,
                    data["reference_cp"],
                    data["sample_mass"],
                    data["reference_mass"],
                )
                uncertainties.append(uncertainty)

            cp_array = np.array(cp_values)
            temp_array = np.array(temps)
            uncertainty_array = np.array(uncertainties)

        else:
            # Continuous mode - calculate for all points
            cp_array = (sample_corr * data["reference_mass"] * data["reference_cp"]) / (
                ref_corr * data["sample_mass"]
            )
            temp_array = temp
            uncertainty_array = self._calculate_three_step_uncertainty(
                sample_corr,
                ref_corr,
                data["reference_cp"],
                data["sample_mass"],
                data["reference_mass"],
            )

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            temp_array,
            cp_array,
            uncertainty_array,
            linear_fit=True,  # Include linear fit parameters
        )

        return CpResult(
            temperature=temp_array,
            specific_heat=cp_array,
            method=CpMethod.THREE_STEP,
            uncertainty=uncertainty_array,
            quality_metrics=quality_metrics,
            metadata={
                "sample_mass": data["sample_mass"],
                "reference_mass": data["reference_mass"],
                "operation_mode": operation_mode,
            },
            operation_mode=operation_mode,
            stable_regions=stable_regions,
        )

    def _calculate_single_step_cp(
        self,
        data: Dict[str, NDArray[np.float64]],
        operation_mode: OperationMode = OperationMode.CONTINUOUS,
        stability_method: StabilityMethod = StabilityMethod.BASIC,
        use_calibration: bool = True,
        **kwargs,
    ) -> CpResult:
        """
        Calculate Cp using single-step method.

        Args:
            data: Dictionary containing measurement data
            operation_mode: Measurement operation mode
            stability_method: Method for detecting stable regions
            use_calibration: Whether to apply calibration
            **kwargs: Additional parameters

        Returns:
            CpResult object
        """
        # Extract required data
        temp = data["temperature"]
        heat_flow = self._get_heat_flow(data)
        time = data["time"]
        sample_mass = data["sample_mass"]

        if "heating_rate" in data:
            heating_rate = data["heating_rate"]
        else:
            heating_rate = np.gradient(temp, time) * 60  # Convert to K/min

        # Find stable regions if in stepped mode
        stable_regions = None
        if operation_mode == OperationMode.STEPPED:
            stable_regions = self._detect_stable_regions(
                temp, heat_flow, time, stability_method, **kwargs
            )

        # Calculate Cp for appropriate regions
        if operation_mode == OperationMode.STEPPED and stable_regions:
            cp_values = []
            temps = []
            uncertainties = []

            for start_idx, end_idx in stable_regions:
                # Calculate averages for stable region
                hf_mean = np.mean(heat_flow[start_idx:end_idx])
                rate_mean = np.mean(heating_rate[start_idx:end_idx])
                temp_mean = np.mean(temp[start_idx:end_idx])

                # Calculate Cp for this region
                cp = hf_mean / (sample_mass * rate_mean)

                cp_values.append(cp)
                temps.append(temp_mean)

                # Calculate uncertainty for this region
                uncertainty = self._calculate_single_step_uncertainty(
                    hf_mean, rate_mean, sample_mass
                )
                uncertainties.append(uncertainty)

            cp_array = np.array(cp_values)
            temp_array = np.array(temps)
            uncertainty_array = np.array(uncertainties)

        else:
            # Continuous mode - calculate for all points
            cp_array = heat_flow / (sample_mass * heating_rate)
            temp_array = temp
            uncertainty_array = self._calculate_single_step_uncertainty(
                heat_flow, heating_rate, sample_mass
            )

        # Apply calibration if requested and available
        if use_calibration and self.calibration_data is not None:
            cp_array, uncertainty_array = self._apply_calibration(
                temp_array, cp_array, uncertainty_array
            )

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            temp_array, cp_array, uncertainty_array, linear_fit=True
        )

        return CpResult(
            temperature=temp_array,
            specific_heat=cp_array,
            method=CpMethod.SINGLE_STEP,
            uncertainty=uncertainty_array,
            quality_metrics=quality_metrics,
            metadata={
                "sample_mass": sample_mass,
                "heating_rate": (
                    heating_rate.tolist()
                    if isinstance(heating_rate, np.ndarray)
                    else heating_rate
                ),
                "operation_mode": operation_mode,
                "calibration_applied": use_calibration,
            },
            operation_mode=operation_mode,
            stable_regions=stable_regions,
        )

    def _calculate_modulated_cp(
        self,
        data: Dict[str, NDArray[np.float64]],
        **kwargs,
    ) -> CpResult:
        """Calculate Cp using modulated DSC method."""
        temp = data["temperature"]
        reversing_hf = data["reversing_heat_flow"]
        time = data["time"]
        modulation_period = data.get("modulation_period", 60.0)
        modulation_amplitude = data.get("modulation_amplitude", 0.5)

        omega = 2 * np.pi / modulation_period
        heating_rate = omega * modulation_amplitude
        cp = reversing_hf / heating_rate

        # Calculate phase angle
        phase_angle = np.arctan2(np.gradient(reversing_hf, time), reversing_hf)

        uncertainty = self._calculate_modulated_uncertainty(
            reversing_hf, modulation_amplitude, modulation_period
        )

        quality_metrics = self._calculate_quality_metrics(
            temp, cp, uncertainty, linear_fit=True
        )

        return CpResult(
            temperature=temp,
            specific_heat=cp,
            method=CpMethod.MODULATED,
            uncertainty=uncertainty,
            quality_metrics=quality_metrics,
            metadata={
                "modulation_period": modulation_period,
                "modulation_amplitude": modulation_amplitude,
                "phase_angle": float(np.mean(phase_angle)),
            },
            operation_mode=OperationMode.CONTINUOUS,
        )

    def _detect_stable_regions(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        time: NDArray[np.float64],
        method: StabilityMethod = StabilityMethod.BASIC,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Detect stable isothermal regions in DSC data.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            time: Time array
            method: Method to use for stability detection
            **kwargs: Additional parameters

        Returns:
            List of (start_idx, end_idx) tuples for stable regions
        """
        if method == StabilityMethod.BASIC:
            return self._detect_basic_stability(temperature, heat_flow, time, **kwargs)
        else:
            # Por ahora solo implementamos el método básico
            return self._detect_basic_stability(temperature, heat_flow, time, **kwargs)

    def _detect_basic_stability(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        time: NDArray[np.float64],
        dt_threshold: float = 0.1,
        dhf_threshold: float = 0.01,
        min_points: int = 50,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """
        Basic stability detection using temperature and heat flow rates.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            time: Time array
            dt_threshold: Maximum allowed temperature rate (K/min)
            dhf_threshold: Maximum allowed heat flow rate variation
            min_points: Minimum number of points for stable region

        Returns:
            List of (start_idx, end_idx) tuples for stable regions
        """
        # Calculate rates
        dt_dt = np.gradient(temperature, time) * 60  # Convert to K/min
        dhf_dt = np.gradient(heat_flow, time)

        # Find regions where both temperature and heat flow are stable
        temp_stable = np.abs(dt_dt) < dt_threshold
        hf_stable = np.abs(dhf_dt) < dhf_threshold
        stable_mask = temp_stable & hf_stable

        # Find continuous stable regions
        stable_regions = []
        start_idx = None

        for i in range(len(stable_mask)):
            if stable_mask[i] and start_idx is None:
                start_idx = i
            elif not stable_mask[i] and start_idx is not None:
                if i - start_idx >= min_points:
                    stable_regions.append((start_idx, i))
                start_idx = None

        # Handle last region
        if start_idx is not None and len(stable_mask) - start_idx >= min_points:
            stable_regions.append((start_idx, len(stable_mask)))

        return stable_regions

    def _calculate_quality_metrics(
        self,
        temperature: NDArray[np.float64],
        cp: NDArray[np.float64],
        uncertainty: NDArray[np.float64],
        linear_fit: bool = True,
        *signals: NDArray[np.float64],
    ) -> Dict[str, float]:
        """Calculate quality metrics for Cp measurement."""
        metrics = {}

        # Calculate signal-to-noise ratio
        noise_level = np.std(np.diff(cp))
        signal_level = np.max(cp) - np.min(cp)
        metrics["snr"] = float(signal_level / noise_level if noise_level > 0 else 0)

        # Calculate average relative uncertainty
        metrics["avg_uncertainty"] = float(np.mean(uncertainty / cp))

        # Calculate smoothness metric
        metrics["smoothness"] = float(1 / (1 + np.std(np.gradient(cp, temperature))))

        # Linear fit metrics if requested
        if linear_fit:
            slope, intercept, r_value, _, stderr = stats.linregress(temperature, cp)
            metrics.update(
                {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_value**2),
                    "std_error": float(stderr),
                    "equation": f"Cp(T) = {intercept:.4f} + ({slope:.6f} * T) [J/g·K]",
                }
            )

        return metrics

    def _calculate_three_step_uncertainty(
        self,
        sample_hf: NDArray[np.float64],
        ref_hf: NDArray[np.float64],
        ref_cp: Union[float, NDArray[np.float64]],
        sample_mass: float,
        ref_mass: float,
    ) -> NDArray[np.float64]:
        """Calculate uncertainty for three-step method."""
        u_hf = 0.005  # 0.5% heat flow uncertainty
        u_mass = 0.0005  # 0.05% mass uncertainty
        u_ref = 0.005  # 0.5% reference Cp uncertainty

        u_combined = np.sqrt(
            (u_hf * sample_hf) ** 2
            + (u_hf * ref_hf) ** 2
            + (u_mass * sample_mass) ** 2
            + (u_mass * ref_mass) ** 2
            + (u_ref * ref_cp) ** 2
        )

        return u_combined

    def _calculate_single_step_uncertainty(
        self,
        heat_flow: NDArray[np.float64],
        heating_rate: Union[float, NDArray[np.float64]],
        sample_mass: float,
    ) -> NDArray[np.float64]:
        """Calculate uncertainty for single-step method."""
        u_hf = 0.01  # 1% heat flow uncertainty
        u_rate = 0.01  # 1% heating rate uncertainty
        u_mass = 0.001  # 0.1% mass uncertainty

        u_combined = np.sqrt(
            (u_hf * heat_flow) ** 2
            + (u_rate * heating_rate) ** 2
            + (u_mass * sample_mass) ** 2
        )

        return u_combined

    def _calculate_modulated_uncertainty(
        self,
        heat_flow: NDArray[np.float64],
        amplitude: float,
        period: float,
    ) -> NDArray[np.float64]:
        """Calculate uncertainty for modulated DSC method."""
        # Component uncertainties
        u_hf = 0.02  # 2% heat flow uncertainty
        u_amp = 0.01  # 1% amplitude uncertainty
        u_period = 0.01  # 1% period uncertainty

        # Combined uncertainty
        u_combined = np.sqrt(
            (u_hf * heat_flow) ** 2
            + (u_amp * amplitude) ** 2
            + (u_period * period) ** 2
        )

        return u_combined

    def _apply_calibration(
        self,
        temperature: NDArray[np.float64],
        cp: NDArray[np.float64],
        uncertainty: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Apply calibration to Cp values.

        Args:
            temperature: Temperature array
            cp: Uncalibrated Cp array
            uncertainty: Uncertainty array

        Returns:
            Tuple of (calibrated_cp, calibrated_uncertainty)
        """
        if self.calibration_data is None:
            return cp, uncertainty

        # Check temperature range
        if (
            np.min(temperature) < self.calibration_data.valid_range[0]
            or np.max(temperature) > self.calibration_data.valid_range[1]
        ):
            raise ValueError("Temperature range outside calibration validity")

        # Interpolate calibration factors
        factors = np.interp(
            temperature,
            self.calibration_data.temperature,
            self.calibration_data.calibration_factors,
        )

        # Apply calibration
        calibrated_cp = cp * factors

        # Update uncertainty
        cal_uncertainty = np.interp(
            temperature,
            self.calibration_data.temperature,
            self.calibration_data.uncertainty,
        )

        calibrated_uncertainty = np.sqrt(
            uncertainty**2 + (calibrated_cp * cal_uncertainty) ** 2
        )

        return calibrated_cp, calibrated_uncertainty

    def _get_reference_cp(self, material: str) -> Dict[str, NDArray[np.float64]]:
        """
        Get reference Cp data for calibration material.

        Args:
            material: Name of reference material

        Returns:
            Dictionary with temperature and Cp arrays

        Raises:
            ValueError: If material is not found in reference data
        """
        if material not in self._reference_data:
            raise ValueError(f"Unknown reference material: {material}")

        return self._reference_data[material]

    def _calculate_calibration_uncertainty(
        self,
        measured_cp: NDArray[np.float64],
        reference_cp: NDArray[np.float64],
        measurement_uncertainty: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate uncertainty in calibration factors."""
        # Component uncertainties
        u_meas = measurement_uncertainty
        u_ref = 0.02 * reference_cp  # 2% uncertainty in reference values

        # Combined uncertainty
        u_combined = np.sqrt(u_meas**2 + u_ref**2)

        return u_combined
