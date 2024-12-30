"""Baseline correction methods for DSC data."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


class BaselineCorrector:
    """Class for DSC baseline correction."""

    def __init__(self, method: str = "linear"):
        """Initialize baseline corrector.

        Args:
            method: Baseline correction method ('linear', 'polynomial', 'sigmoidal')
        """
        self.method = method
        self._correction_methods = {
            "linear": self._fit_linear_baseline,
            "polynomial": self._fit_polynomial_baseline,
            "sigmoidal": self._fit_sigmoidal_baseline
        }

    def correct(self,
                temperature: NDArray[np.float64],
                heat_flow: NDArray[np.float64],
                **kwargs) -> Tuple[NDArray[np.float64], Dict]:
        """Apply baseline correction."""
        if self.method not in self._correction_methods:
            raise ValueError(f"Unknown baseline method: {self.method}")

        correction_func = self._correction_methods[self.method]
        return correction_func(temperature, heat_flow, **kwargs)

    def _fit_linear_baseline(self,
                             temperature: NDArray[np.float64],
                             heat_flow: NDArray[np.float64],
                             range_points: Optional[List[Tuple[float, float]]] = None
                             ) -> Tuple[NDArray[np.float64], Dict]:
        """Fit linear baseline."""
        if range_points is None:
            points = [(temperature[0], temperature[-1])]
        else:
            points = range_points

        temp_points = []
        heat_points = []
        for start_temp, end_temp in points:
            mask = (temperature >= start_temp) & (temperature <= end_temp)
            temp_points.extend(temperature[mask])
            heat_points.extend(heat_flow[mask])

        coeffs = np.polyfit(temp_points, heat_points, 1)
        baseline = np.polyval(coeffs, temperature)

        return baseline, {
            "type": "linear",
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1])
        }

    def _fit_polynomial_baseline(self,
                                 temperature: NDArray[np.float64],
                                 heat_flow: NDArray[np.float64],
                                 degree: int = 3,
                                 **kwargs) -> Tuple[NDArray[np.float64], Dict]:
        """Fit polynomial baseline."""
        # Implementation similar to linear but with higher degree
        coeffs = np.polyfit(temperature, heat_flow, degree)
        baseline = np.polyval(coeffs, temperature)

        return baseline, {
            "type": "polynomial",
            "degree": degree,
            "coefficients": coeffs.tolist()
        }

    def _fit_sigmoidal_baseline(self,
                                temperature: NDArray[np.float64],
                                heat_flow: NDArray[np.float64],
                                **kwargs) -> Tuple[NDArray[np.float64], Dict]:
        """Fit sigmoidal baseline."""

        def sigmoid(x, a, b, c, d):
            return a + (b - a) / (1 + np.exp(-c * (x - d)))

        try:
            popt, _ = curve_fit(sigmoid, temperature, heat_flow)
            baseline = sigmoid(temperature, *popt)

            return baseline, {
                "type": "sigmoidal",
                "parameters": popt.tolist()
            }
        except RuntimeError:
            # Fallback to linear if sigmoid fitting fails
            return self._fit_linear_baseline(temperature, heat_flow)
