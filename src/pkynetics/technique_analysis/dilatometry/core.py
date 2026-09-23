"""The analysis entry point."""

from typing import Any, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from pkynetics.technique_analysis.utilities import detect_segment_direction

from .detection import DEFAULT_DETECTION
from .methods.lever import lever_method
from .methods.tangent import tangent_method
from .transformation_points import find_transformation_limits
from .types import ReturnDict


def analyze_dilatometry_curve(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    method: str = "lever",
    margin_percent: Optional[float] = None,
    find_inflection_margin: float = 0.2,
    min_points_fit: int = 10,
    min_r2_optimal_margin: float = 0.99,
    deviation_fraction: float = 0.05,
    detection: str = DEFAULT_DETECTION,
    **detection_options: Any,
) -> ReturnDict:
    """
    Analyze the dilatometry curve to extract key transformation parameters.

    Args:
        temperature: Array of temperature values (°C).
        strain: Array of strain or relative length change values.
        method: Analysis method ('lever' or 'tangent'). Default is 'lever'.
        margin_percent: Margin percentage (0.0 to 1.0) for fitting linear segments
                        (used by both methods). If None for tangent, optimal margin is found.
                        Default for lever is often implicitly 0.2 or uses find_inflection_margin.
        find_inflection_margin: Margin percentage (0.1-0.4) used specifically by the
                                'lever' method's `find_inflection_points` function. Default is 0.2.
        min_points_fit: Minimum number of points required for reliable linear fitting
                        in tangent/lever methods. Default is 10.
        min_r2_optimal_margin: Minimum R² required when using `find_optimal_margin`
                               in the tangent method. Default is 0.99.
        deviation_fraction: Fraction of the peak excursion of ``dS/dT`` that still
            counts as transforming, for both methods. Default is 0.05.
        detection: Which rule locates the transformation limits
            ('derivative' by default). This is a different question from
            `method`: `detection` is *where* the transformation is, `method`
            is *how far it has gone*, and the two are chosen independently.
        **detection_options: Passed through to the chosen detector.

    Returns:
        Dictionary containing analysis results: start, end, mid temperatures,
        transformed fraction, extrapolations, quality metrics (for tangent), etc.

    Raises:
        ValueError: If method is not supported, data is insufficient, or analysis fails.
    """
    if len(temperature) != len(strain):
        raise ValueError("Temperature and strain arrays must have the same length.")
    if len(temperature) < max(
        20, min_points_fit * 2
    ):  # Need a reasonable number of points overall
        raise ValueError(f"Insufficient data points ({len(temperature)}) for analysis.")

    # Ensure input arrays are numpy arrays
    temperature = np.asarray(temperature, dtype=np.float64)
    strain = np.asarray(strain, dtype=np.float64)

    # Detect direction early on
    is_cooling = detect_segment_direction(temperature, strain)

    # --- Method Dispatch ---
    if method.lower() == "lever":
        # Use find_inflection_margin for finding points, and a separate margin (or default) for fraction calc
        lever_margin = (
            margin_percent if margin_percent is not None else 0.2
        )  # Default margin for fraction calc if not given
        return lever_method(
            temperature,
            strain,
            is_cooling=is_cooling,
            margin_percent_fraction=lever_margin,
            find_inflection_margin=find_inflection_margin,
            min_points_fit=min_points_fit,
            detection=detection,
            **detection_options,
        )
    elif method.lower() == "tangent":
        return tangent_method(
            temperature,
            strain,
            is_cooling=is_cooling,
            margin_percent=margin_percent,  # Can be None to trigger optimal search
            deviation_fraction=deviation_fraction,
            min_points_fit=min_points_fit,
            min_r2_optimal_margin=min_r2_optimal_margin,
            detection=detection,
            **detection_options,
        )
    else:
        raise ValueError(
            f"Unsupported method: '{method}'. Choose 'lever' or 'tangent'."
        )


class DilatometryAnalyzer:
    """Dilatometry analysis with the settings held in one place.

    Every function in this package takes the same handful of settings, under
    names that drifted apart as the module grew: the margin used to *locate* a
    transformation has been called ``margin``, ``find_inflection_margin`` and
    ``limits_margin``, and the margin used to *fit the baselines* has been
    ``margin_percent`` and ``margin_percent_fraction``. Those are two settings,
    not five, and here they are named once:

    ============================== ==================================================
    Setting                        What it controls
    ============================== ==================================================
    ``limits_margin``              Fraction of the data at each end taken as baseline
                                   when locating the transformation.
    ``baseline_margin``            Fraction used to fit the baselines the transformed
                                   fraction is measured against. ``None`` lets the
                                   tangent method search for the widest one that fits.
    ``deviation_fraction``         Fraction of the peak excursion of ``dS/dT`` that
                                   still counts as transforming.
    ``min_points_fit``             Points a linear fit needs to be trusted.
    ``min_r2``                     R² a searched-for baseline margin must reach.
    ``baseline_min_r2``            R² below which a baseline window is reported as
                                   reaching into a transformation.
    ``detection``                  Which rule locates the transformation. Separate
                                   from the ``method`` of :meth:`analyze`: this is
                                   *where* the transformation is, that is *how far
                                   it has gone*.
    ============================== ==================================================

    The free functions keep their own parameter names and go on working
    unchanged; this class is a second way in, not a replacement.

    Examples:
        >>> analyzer = DilatometryAnalyzer(limits_margin=0.15)
        >>> result = analyzer.analyze(temperature, strain)          # doctest: +SKIP
        >>> analyzer.limits                                          # doctest: +SKIP
        (142, 486)
    """

    def __init__(
        self,
        limits_margin: float = 0.2,
        baseline_margin: Optional[float] = None,
        deviation_fraction: float = 0.05,
        min_points_fit: int = 10,
        min_r2: float = 0.99,
        baseline_min_r2: float = 0.99,
        smooth_window_fraction: float = 0.05,
        polyorder: int = 2,
        min_points_smooth: int = 5,
        detection: str = DEFAULT_DETECTION,
    ) -> None:
        self.limits_margin = limits_margin
        self.baseline_margin = baseline_margin
        self.deviation_fraction = deviation_fraction
        self.min_points_fit = min_points_fit
        self.min_r2 = min_r2
        self.baseline_min_r2 = baseline_min_r2
        self.smooth_window_fraction = smooth_window_fraction
        self.polyorder = polyorder
        self.min_points_smooth = min_points_smooth
        self.detection = detection

        # Filled in by analyze(); None until then
        self.temperature: Optional[NDArray[np.float64]] = None
        self.strain: Optional[NDArray[np.float64]] = None
        self.is_cooling: Optional[bool] = None
        self.limits: Optional[Tuple[int, int]] = None
        self.result: Optional[ReturnDict] = None

    def find_limits(
        self,
        temperature: NDArray[np.float64],
        strain: NDArray[np.float64],
        is_cooling: Optional[bool] = None,
    ) -> Tuple[int, int]:
        """Indices where the transformation starts and ends.

        Locates them directly, with this analyzer's settings. It is what
        :func:`find_transformation_limits` does, with the configuration already
        filled in; :meth:`analyze` does not go through it, since each method
        locates its limits its own way.

        Args:
            temperature: Array of temperature values (°C).
            strain: Array of strain or relative length change values.
            is_cooling: Ramp direction; detected from the data when omitted.

        Returns:
            Tuple ``(start_idx, end_idx)``, in array order.
        """
        temperature = np.asarray(temperature, dtype=np.float64)
        strain = np.asarray(strain, dtype=np.float64)
        if is_cooling is None:
            is_cooling = detect_segment_direction(temperature, strain)
        return find_transformation_limits(
            temperature,
            strain,
            is_cooling=is_cooling,
            margin=self.limits_margin,
            deviation_fraction=self.deviation_fraction,
            smooth_window_fraction=self.smooth_window_fraction,
            polyorder=self.polyorder,
            min_points_smooth=self.min_points_smooth,
            baseline_min_r2=self.baseline_min_r2,
            detection=self.detection,
        )

    def analyze(
        self,
        temperature: NDArray[np.float64],
        strain: NDArray[np.float64],
        method: str = "lever",
    ) -> ReturnDict:
        """Run the analysis and keep what it went through.

        The return value is the same mapping
        :func:`analyze_dilatometry_curve` returns. What the analyzer adds is
        that the inputs, the ramp direction and the transformation limits stay
        reachable afterwards, as :attr:`temperature`, :attr:`strain`,
        :attr:`is_cooling` and :attr:`limits`.

        Args:
            temperature: Array of temperature values (°C).
            strain: Array of strain or relative length change values.
            method: ``"lever"`` or ``"tangent"``.

        Returns:
            The analysis results.

        Raises:
            ValueError: If the method is unsupported or the data insufficient.
        """
        temperature = np.asarray(temperature, dtype=np.float64)
        strain = np.asarray(strain, dtype=np.float64)

        result = analyze_dilatometry_curve(
            temperature,
            strain,
            method=method,
            margin_percent=self.baseline_margin,
            find_inflection_margin=self.limits_margin,
            min_points_fit=self.min_points_fit,
            min_r2_optimal_margin=self.min_r2,
            deviation_fraction=self.deviation_fraction,
            detection=self.detection,
        )

        self.temperature = temperature
        self.strain = strain
        self.is_cooling = bool(result["is_cooling"])
        # Taken from the result rather than recomputed: each method locates the
        # limits with its own margin, so a second call here could disagree with
        # the temperatures the analysis actually reported.
        self.limits = (
            self._nearest(temperature, result, "start_temperature"),
            self._nearest(temperature, result, "end_temperature"),
        )
        self.result = result
        return result

    @staticmethod
    def _nearest(temperature: NDArray[np.float64], result: ReturnDict, key: str) -> int:
        """Index of the point at a temperature the result reports.

        The cast is needed because the result mapping is typed as the union of
        everything it can hold; these two keys are always floats.
        """
        return int(np.argmin(np.abs(temperature - cast(float, result[key]))))
