"""Utility functions for DSC data analysis."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import signal


def validate_window_size(
    data_length: int, window_length: int, min_size: int = 3
) -> int:
    """
    Validate and adjust window size for signal processing operations.

    Args:
        data_length: Length of the data array
        window_length: Requested window length
        min_size: Minimum acceptable window size (must be odd)

    Returns:
        Valid window length (odd number, between min_size and data_length)
    """
    # Ensure min_size is odd
    min_size = (min_size // 2) * 2 + 1

    # Adjust window size if necessary
    if window_length >= data_length:
        window_length = min(data_length - 1, 21)

    # Ensure window is odd
    if window_length % 2 == 0:
        window_length -= 1

    # Ensure window is at least min_size
    return max(min_size, window_length)


def safe_savgol_filter(
    data: NDArray[np.float64], window_length: int, polyorder: int
) -> NDArray[np.float64]:
    """
    Apply Savitzky-Golay filter with validated window size.

    Args:
        data: Input data array
        window_length: Desired window length
        polyorder: Polynomial order for filtering

    Returns:
        Filtered data array
    """
    valid_window = validate_window_size(len(data), window_length)
    valid_polyorder = min(valid_window - 1, polyorder)
    return np.asarray(
        signal.savgol_filter(data, valid_window, valid_polyorder), dtype=np.float64
    )


def find_intersection_point(
    x: NDArray[np.float64],
    y1: NDArray[np.float64],
    y2: NDArray[np.float64],
    start_idx: int,
    direction: str = "forward",
) -> Tuple[float, int]:
    """
    Find the intersection point between two curves.

    Args:
        x: x-axis values
        y1: First curve values
        y2: Second curve values
        start_idx: Starting index for search
        direction: Search direction ("forward" or "backward")

    Returns:
        Tuple of (intersection x value, index)
    """
    if direction == "forward":
        search_range = range(start_idx, len(x) - 1)
    else:
        search_range = range(start_idx, 0, -1)

    for i in search_range:
        if direction == "forward":
            if (y1[i] <= y2[i] and y1[i + 1] >= y2[i + 1]) or (
                y1[i] >= y2[i] and y1[i + 1] <= y2[i + 1]
            ):
                break
        else:
            if (y1[i] <= y2[i] and y1[i - 1] >= y2[i - 1]) or (
                y1[i] >= y2[i] and y1[i - 1] <= y2[i - 1]
            ):
                break
    else:
        return x[start_idx], start_idx

    # Linear interpolation to find precise intersection
    if direction == "forward":
        idx1, idx2 = i, i + 1
    else:
        idx1, idx2 = i - 1, i

    x1, x2 = x[idx1], x[idx2]
    y1_1, y1_2 = y1[idx1], y1[idx2]
    y2_1, y2_2 = y2[idx1], y2[idx2]

    # Intersection point by linear interpolation
    dx = x2 - x1
    dy1 = y1_2 - y1_1
    dy2 = y2_2 - y2_1

    if abs(dy1 - dy2) < 1e-10:  # Parallel lines
        return x[i], i

    x_int = x1 + (y2_1 - y1_1) * dx / (dy1 - dy2)
    return float(x_int), i


class DSCUnits(Enum):
    """Enumeration of common DSC measurement units."""

    # Temperature units
    CELSIUS = "°C"
    KELVIN = "K"
    FAHRENHEIT = "°F"

    # Heat flow units
    MILLIWATTS = "mW"
    WATTS = "W"
    MICROWATTS = "µW"

    # Heat capacity units
    JOULES_PER_GRAM_KELVIN = "J/(g·K)"
    JOULES_PER_MOL_KELVIN = "J/(mol·K)"
    CAL_PER_GRAM_KELVIN = "cal/(g·K)"

    # Heating rate units
    KELVIN_PER_MINUTE = "K/min"
    CELSIUS_PER_MINUTE = "°C/min"
    KELVIN_PER_SECOND = "K/s"


class SignalProcessor:
    """Class for DSC signal processing operations."""

    def __init__(self, default_window: int = 21, default_polyorder: int = 3):
        """
        Initialize signal processor.

        Args:
            default_window: Default window size for smoothing operations
            default_polyorder: Default polynomial order for smoothing
        """
        self.default_window = default_window
        self.default_polyorder = default_polyorder

    def smooth_signal(
        self,
        data: NDArray[np.float64],
        window_length: Optional[int] = None,
        polyorder: Optional[int] = None,
        method: str = "savgol",
    ) -> NDArray[np.float64]:
        """
        Apply smoothing to signal data.

        Args:
            data: Input signal array
            window_length: Window size for smoothing
            polyorder: Polynomial order for Savitzky-Golay filter
            method: Smoothing method ('savgol', 'moving_average', 'lowess')

        Returns:
            Smoothed signal array
        """
        import statsmodels.api as sm

        window = window_length if window_length is not None else self.default_window
        polyorder = polyorder if polyorder is not None else self.default_polyorder

        if window % 2 == 0:
            window += 1  # Ensure odd window length

        if method == "savgol":
            return np.asarray(
                signal.savgol_filter(data, window, polyorder), dtype=np.float64
            )
        elif method == "moving_average":
            kernel = np.ones(window) / window
            return np.convolve(data, kernel, mode="same")
        elif method == "lowess":
            x = np.arange(len(data))
            frac = min(1.0, max(0.01, window / len(data)))
            lowess = sm.nonparametric.lowess(data, x, frac=frac, return_sorted=False)
            return np.asarray(lowess, dtype=np.float64)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def remove_outliers(
        self,
        data: NDArray[np.float64],
        window: Optional[int] = None,
        threshold: float = 3.0,
    ) -> NDArray[np.float64]:
        """
        Remove outliers from signal data.

        The local score is a modified z-score built on the median and the
        median absolute deviation, not on the mean and the standard
        deviation. A mean/std score is computed from statistics the outlier
        itself shifts, so it caps at (n-1)/sqrt(n) for a window of n points:
        in the truncated windows at the edges, or when two outliers share a
        window, it never reaches the usual threshold. The median and the MAD
        do not move with a minority of contaminated points, so the score
        stays large no matter where the outlier sits.

        Args:
            data: Input signal array
            window: Window size for local outlier detection
            threshold: Modified z-score threshold for outlier detection

        Returns:
            Signal array with outliers removed
        """
        window = window or self.default_window
        cleaned_data = data.copy()

        # Use rolling window to detect local outliers
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            local_data = data[start:end]

            local_median = float(np.median(local_data))
            deviations = np.abs(local_data - local_median)
            # 1.4826 * MAD estimates sigma for normally distributed data
            scale = 1.4826 * float(np.median(deviations))
            if scale == 0:
                # Over half the window holds the exact same value (a flat,
                # coarsely digitized stretch), so the MAD carries no spread.
                # Fall back to the mean absolute deviation, which a minority
                # of outliers still cannot dominate.
                scale = 1.2533 * float(np.mean(deviations))
            if scale == 0:
                # Every point identical: nothing can be an outlier
                continue

            if abs(float(data[i]) - local_median) / scale >= threshold:
                # Replace outlier with local median
                cleaned_data[i] = local_median

        return np.asarray(cleaned_data, dtype=np.float64)

    def filter_signal(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        cutoff_freq: Union[float, Tuple[float, float]],
        filter_type: str = "lowpass",
        order: int = 4,
    ) -> NDArray[np.float64]:
        """
        Apply frequency-domain filtering.

        Args:
            data: Input signal array
            sampling_rate: Data sampling rate in Hz
            cutoff_freq: Filter cutoff frequency in Hz, or (low, high) for bandpass
            filter_type: Filter type ('lowpass', 'highpass', 'bandpass')
            order: Filter order

        Returns:
            Filtered signal array
        """
        nyquist = sampling_rate / 2

        if filter_type == "bandpass":
            if not isinstance(cutoff_freq, tuple) or len(cutoff_freq) != 2:
                raise ValueError("Bandpass filter requires tuple of frequencies")
            low, high = cutoff_freq
            b, a = signal.butter(order, (low / nyquist, high / nyquist), btype="band")
            return np.asarray(signal.filtfilt(b, a, data), dtype=np.float64)

        if isinstance(cutoff_freq, tuple):
            raise ValueError("Tuple of frequencies is only valid for bandpass")
        normalized_cutoff = cutoff_freq / nyquist

        if filter_type == "lowpass":
            b, a = signal.butter(order, normalized_cutoff, btype="low")
        elif filter_type == "highpass":
            b, a = signal.butter(order, normalized_cutoff, btype="high")
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        return np.asarray(signal.filtfilt(b, a, data), dtype=np.float64)

    def calculate_derivatives(
        self,
        temperature: NDArray[np.float64],
        heat_flow: NDArray[np.float64],
        smooth: bool = True,
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Calculate derivatives of heat flow with respect to temperature.

        Args:
            temperature: Temperature array
            heat_flow: Heat flow array
            smooth: Whether to smooth derivatives

        Returns:
            Dictionary with first and second derivatives
        """
        if smooth:
            heat_flow = self.smooth_signal(heat_flow)

        d1 = np.gradient(heat_flow, temperature)
        d2 = np.gradient(d1, temperature)

        if smooth:
            d1 = self.smooth_signal(d1)
            d2 = self.smooth_signal(d2)

        return {"first_derivative": d1, "second_derivative": d2}

    def calculate_noise_level(
        self, data: NDArray[np.float64], window: Optional[int] = None
    ) -> float:
        """
        Estimate noise level in signal.

        Args:
            data: Input signal array
            window: Window size for local noise estimation

        Returns:
            Estimated noise level
        """
        window = window or self.default_window

        # Calculate local standard deviations
        local_std = []
        for i in range(0, len(data) - window, window):
            local_std.append(np.std(data[i : i + window]))

        # Use median of local standard deviations as noise estimate
        return float(np.median(local_std))


class UnitConverter:
    """Class for DSC unit conversions."""

    # Conversion factors
    _TEMPERATURE_FACTORS = {
        (DSCUnits.CELSIUS, DSCUnits.KELVIN): lambda x: x + 273.15,
        (DSCUnits.KELVIN, DSCUnits.CELSIUS): lambda x: x - 273.15,
        (DSCUnits.FAHRENHEIT, DSCUnits.CELSIUS): lambda x: (x - 32) * 5 / 9,
        (DSCUnits.CELSIUS, DSCUnits.FAHRENHEIT): lambda x: x * 9 / 5 + 32,
    }

    _HEAT_FLOW_FACTORS = {
        (DSCUnits.MILLIWATTS, DSCUnits.WATTS): lambda x: x / 1000,
        (DSCUnits.WATTS, DSCUnits.MILLIWATTS): lambda x: x * 1000,
        (DSCUnits.MICROWATTS, DSCUnits.MILLIWATTS): lambda x: x / 1000,
        (DSCUnits.MILLIWATTS, DSCUnits.MICROWATTS): lambda x: x * 1000,
    }

    @classmethod
    def convert_temperature(
        cls,
        value: Union[float, NDArray[np.float64]],
        from_unit: DSCUnits,
        to_unit: DSCUnits,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Convert temperature between units.

        Args:
            value: Temperature value(s) to convert
            from_unit: Original temperature unit
            to_unit: Target temperature unit

        Returns:
            Converted temperature value(s)
        """
        if from_unit == to_unit:
            return value

        conversion = cls._TEMPERATURE_FACTORS.get((from_unit, to_unit))
        if conversion is None:
            # Try to find multi-step conversion
            intermediate = DSCUnits.CELSIUS
            try:
                step1 = cls._TEMPERATURE_FACTORS[(from_unit, intermediate)]
                step2 = cls._TEMPERATURE_FACTORS[(intermediate, to_unit)]
                converted: Union[float, NDArray[np.float64]] = step2(step1(value))
                return converted
            except KeyError:
                raise ValueError(f"No conversion path from {from_unit} to {to_unit}")

        result: Union[float, NDArray[np.float64]] = conversion(value)
        return result

    @classmethod
    def convert_heat_flow(
        cls,
        value: Union[float, NDArray[np.float64]],
        from_unit: DSCUnits,
        to_unit: DSCUnits,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Convert heat flow between units.

        Args:
            value: Heat flow value(s) to convert
            from_unit: Original heat flow unit
            to_unit: Target heat flow unit

        Returns:
            Converted heat flow value(s)
        """
        if from_unit == to_unit:
            return value

        conversion = cls._HEAT_FLOW_FACTORS.get((from_unit, to_unit))
        if conversion is None:
            # Try to find multi-step conversion
            intermediate = DSCUnits.MILLIWATTS
            try:
                step1 = cls._HEAT_FLOW_FACTORS[(from_unit, intermediate)]
                step2 = cls._HEAT_FLOW_FACTORS[(intermediate, to_unit)]
                converted: Union[float, NDArray[np.float64]] = step2(step1(value))
                return converted
            except KeyError:
                raise ValueError(f"No conversion path from {from_unit} to {to_unit}")

        result: Union[float, NDArray[np.float64]] = conversion(value)
        return result

    @staticmethod
    def convert_heating_rate(
        value: float, from_unit: DSCUnits, to_unit: DSCUnits
    ) -> float:
        """
        Convert heating rate between units.

        Args:
            value: Heating rate value to convert
            from_unit: Original heating rate unit
            to_unit: Target heating rate unit

        Returns:
            Converted heating rate value
        """
        # Define conversion factors relative to K/min
        factors = {
            DSCUnits.KELVIN_PER_MINUTE: 1.0,
            DSCUnits.CELSIUS_PER_MINUTE: 1.0,  # Same numerical value
            DSCUnits.KELVIN_PER_SECOND: 60.0,
        }

        try:
            return value * factors[from_unit] / factors[to_unit]
        except KeyError:
            raise ValueError(
                f"Unsupported heating rate unit conversion: {from_unit} to {to_unit}"
            )


class DataValidator:
    """Class for validating DSC data."""

    @staticmethod
    def validate_temperature_data(
        temperature: NDArray[np.float64],
        min_temp: Optional[float] = None,
        max_temp: Optional[float] = None,
        strict_monotonic: bool = False,
    ) -> bool:
        """
        Validate temperature data.

        Args:
            temperature: Temperature array
            min_temp: Minimum allowed temperature
            max_temp: Maximum allowed temperature
            strict_monotonic: If True, requires strictly increasing temperature

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not isinstance(temperature, np.ndarray):
            raise ValueError("Temperature must be a numpy array")

        if not np.issubdtype(temperature.dtype, np.floating):
            raise ValueError("Temperature must be floating-point type")

        if len(temperature) < 2:
            raise ValueError("Temperature array must have at least 2 points")

        # Check for invalid values
        if not np.all(np.isfinite(temperature)):
            raise ValueError("Temperature contains invalid values (inf or nan)")

        # Check temperature range if specified
        if min_temp is not None and np.min(temperature) < min_temp:
            raise ValueError(f"Temperature below minimum allowed value: {min_temp}")

        if max_temp is not None and np.max(temperature) > max_temp:
            raise ValueError(f"Temperature above maximum allowed value: {max_temp}")

        if strict_monotonic:
            # Only check for strictly increasing if requested
            if not np.all(np.diff(temperature) > 0):
                raise ValueError("Temperature must be strictly increasing")
        else:
            # Check for reasonable temperature changes (no sudden jumps)
            temp_diff = np.abs(np.diff(temperature))
            if np.any(temp_diff > 50):  # 50K maximum step between points
                raise ValueError("Detected unrealistic temperature jumps in data")

        return True

    @staticmethod
    def validate_heat_flow_data(
        heat_flow: NDArray[np.float64],
        temperature: Optional[NDArray[np.float64]] = None,
    ) -> bool:
        """
        Validate heat flow data.

        Args:
            heat_flow: Heat flow array
            temperature: Optional temperature array for length comparison

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not isinstance(heat_flow, np.ndarray):
            raise ValueError("Heat flow must be a numpy array")

        if not np.issubdtype(heat_flow.dtype, np.floating):
            raise ValueError("Heat flow must be floating-point type")

        if temperature is not None and len(heat_flow) != len(temperature):
            raise ValueError("Heat flow and temperature arrays must have same length")

        if not np.all(np.isfinite(heat_flow)):
            raise ValueError("Heat flow contains invalid values")

        # Check for realistic heat flow values (typical DSC range ±500 mW)
        if np.any(np.abs(heat_flow) > 500):
            raise ValueError("Heat flow values outside typical DSC range")

        return True

    @staticmethod
    def detect_temperature_program(
        temperature: NDArray[np.float64],
        time: NDArray[np.float64],
        min_segment_duration: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Detect temperature program type and parameters.

        Args:
            temperature: Temperature array
            time: Time array in seconds
            min_segment_duration: Shortest segment to report, in seconds. The
                rate is median-filtered over this window and shorter runs are
                merged into the preceding segment, so noise and brief sample
                temperature stalls (e.g. during melting) do not split ramps.

        Returns:
            Dictionary containing program type ('continuous', 'stepped' or
            'cyclic'), segments (type, mean rate in K/min, start_idx, end_idx
            exclusive), average heating/cooling rates in K/min (nan if absent)
            and segment counts
        """
        # Heating rate in K/min (time is in seconds)
        temp_rate = np.gradient(temperature, time) * 60

        # Median-filter the rate over min_segment_duration
        dt = float(np.median(np.diff(time)))
        window = int(min_segment_duration / dt) if dt > 0 else 1
        window = min(window, len(temp_rate))
        if window % 2 == 0:
            window -= 1
        smoothed_rate = signal.medfilt(temp_rate, window) if window >= 3 else temp_rate

        # Rates below this magnitude are treated as isothermal (K/min)
        rate_threshold = 0.5

        labels = np.where(
            smoothed_rate > rate_threshold,
            "heating",
            np.where(smoothed_rate < -rate_threshold, "cooling", "isothermal"),
        )

        def _runs(lbl: NDArray[Any]) -> Tuple[List[int], List[int]]:
            boundaries = np.flatnonzero(lbl[1:] != lbl[:-1]) + 1
            return (
                [0, *boundaries.tolist()],
                [*boundaries.tolist(), len(lbl)],
            )

        # Merge the shortest run below min_segment_duration into its preceding
        # run (the following one for the first run) until none remain
        starts, ends = _runs(labels)
        while len(starts) > 1:
            durations = [time[e - 1] - time[st] for st, e in zip(starts, ends)]
            shortest = int(np.argmin(durations))
            if durations[shortest] >= min_segment_duration:
                break
            neighbour = shortest - 1 if shortest > 0 else 1
            labels[starts[shortest] : ends[shortest]] = labels[starts[neighbour]]
            starts, ends = _runs(labels)

        segments: List[Dict[str, Union[str, float, int]]] = [
            {
                "type": str(labels[a]),
                "rate": float(np.mean(temp_rate[a:b])),
                "start_idx": int(a),
                "end_idx": int(b),
            }
            for a, b in zip(starts, ends)
        ]

        types = [seg["type"] for seg in segments]
        n_isothermal = types.count("isothermal")
        n_heating = types.count("heating")
        n_cooling = types.count("cooling")

        # Isothermal holds between two ramps indicate a stepped program;
        # leading/trailing equilibration holds do not.
        n_interior_isothermal = types[1:-1].count("isothermal")

        if n_cooling > 0 and n_heating > 0:
            program_type = "cyclic"
        elif n_interior_isothermal > 0:
            program_type = "stepped"
        else:
            program_type = "continuous"

        def _mean_rate(kind: str) -> float:
            rates = [float(seg["rate"]) for seg in segments if seg["type"] == kind]
            return float(np.mean(rates)) if rates else float("nan")

        return {
            "type": program_type,
            "segments": segments,
            "avg_heating_rate": _mean_rate("heating"),
            "avg_cooling_rate": _mean_rate("cooling"),
            "n_isothermal": n_isothermal,
            "n_heating": n_heating,
            "n_cooling": n_cooling,
        }

    @staticmethod
    def check_sampling_rate(
        time: NDArray[np.float64],
        tolerance: float = 0.1,
    ) -> float:
        """
        Check if sampling is uniform.

        Args:
            time: Time array
            tolerance: Allowed relative deviation from the mean sampling interval

        Returns:
            Mean sampling interval if uniform, raises ValueError otherwise
        """
        if len(time) < 2:
            raise ValueError("Time array must have at least 2 points")

        dt = np.diff(time)
        mean_dt = np.mean(dt)

        if mean_dt <= 0:
            raise ValueError("Time must be increasing")

        if np.any(np.abs(dt - mean_dt) / mean_dt > tolerance):
            raise ValueError("Non-uniform time sampling detected")

        return float(mean_dt)
