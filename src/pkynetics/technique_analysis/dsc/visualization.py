"""Plotting functions for DSC analysis results.

All functions draw on a given matplotlib Axes (like the other pkynetics
plotting modules) except :func:`plot_dsc_analysis`, which creates a figure.
Temperatures are stored in K; pass ``celsius=True`` to plot in degC.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .core import DSCAnalyzer
from .types import CpResult, DSCPeak, OperationMode

FloatArray = NDArray[np.float64]

KELVIN_OFFSET = 273.15


def _temperature(values: Any, celsius: bool) -> Any:
    """Convert K to degC if requested."""
    return np.asarray(values) - KELVIN_OFFSET if celsius else values


def _temperature_label(celsius: bool) -> str:
    return "Temperature (°C)" if celsius else "Temperature (K)"


def _heat_flow_label(exo_up: bool) -> str:
    return "Heat flow (mW, exo ↑)" if exo_up else "Heat flow (mW, endo ↑)"


def plot_dsc_curve(
    ax: plt.Axes,
    temperature: FloatArray,
    heat_flow: FloatArray,
    baseline: Optional[FloatArray] = None,
    peaks: Optional[Sequence[DSCPeak]] = None,
    exo_up: bool = False,
    celsius: bool = False,
    label: str = "Heat flow",
) -> None:
    """
    Plot a DSC curve with optional baseline and characterized peaks.

    Peaks (from PeakAnalyzer, found in the baseline-corrected signal) are
    shaded between the curve and the baseline, with their extrapolated
    onset/endset, peak temperature and enthalpy (if known) marked.

    Args:
        ax: Matplotlib axes object
        temperature: Temperature array (K)
        heat_flow: Heat flow array (mW)
        baseline: Optional baseline array (mW)
        peaks: Optional peaks to mark
        exo_up: Sign convention of the data (for the axis label)
        celsius: Plot temperatures in degC
        label: Legend label of the curve
    """
    t = _temperature(temperature, celsius)
    ax.plot(t, heat_flow, color="C0", label=label)
    base = np.zeros_like(heat_flow) if baseline is None else baseline
    if baseline is not None:
        ax.plot(t, baseline, "--", color="C7", label="Baseline")

    for i, peak in enumerate(peaks or []):
        lo, hi = peak.peak_indices
        region = slice(lo, hi + 1)
        ax.fill_between(
            t[region],
            base[region],
            heat_flow[region],
            color="C1",
            alpha=0.25,
            label="Peak area" if i == 0 else None,
        )
        for value, style in (
            (peak.onset_temperature, ":"),
            (peak.endset_temperature, ":"),
        ):
            ax.axvline(_temperature(value, celsius), color="C1", linestyle=style)

        peak_t = peak.peak_temperature
        peak_idx = int(np.argmin(np.abs(np.asarray(temperature) - peak_t)))
        text = f"{_temperature(peak_t, celsius):.1f}"
        if np.isfinite(peak.enthalpy):
            text += f"\nΔH = {peak.enthalpy:.1f} J/g"
        ax.annotate(
            text,
            (_temperature(peak_t, celsius), heat_flow[peak_idx]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
        )

    ax.set_xlabel(_temperature_label(celsius))
    ax.set_ylabel(_heat_flow_label(exo_up))
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_thermal_events(
    ax: plt.Axes,
    temperature: FloatArray,
    heat_flow: FloatArray,
    events: Dict[str, List[Any]],
    exo_up: bool = False,
    celsius: bool = False,
) -> None:
    """
    Plot a DSC curve with detected thermal events.

    Args:
        ax: Matplotlib axes object
        temperature: Temperature array (K)
        heat_flow: Heat flow array (mW)
        events: Output of ThermalEventDetector.detect_events
        exo_up: Sign convention of the data (for the axis label)
        celsius: Plot temperatures in degC
    """
    t = _temperature(temperature, celsius)
    ax.plot(t, heat_flow, color="C0", label="Heat flow")

    def value_at(temp: float) -> float:
        """Heat flow at the sample nearest to a temperature."""
        return float(heat_flow[int(np.argmin(np.abs(temperature - temp)))])

    for i, gt in enumerate(events.get("glass_transitions", [])):
        ax.axvspan(
            _temperature(gt.onset_temperature, celsius),
            _temperature(gt.endpoint_temperature, celsius),
            color="C2",
            alpha=0.15,
            label="Glass transition" if i == 0 else None,
        )
        mid = gt.midpoint_temperature
        ax.plot(_temperature(mid, celsius), value_at(mid), "o", color="C2")
        ax.annotate(
            f"Tg = {_temperature(mid, celsius):.1f}",
            (_temperature(mid, celsius), value_at(mid)),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=8,
        )

    for kind, color, name in (
        ("crystallization", "C3", "Crystallization"),
        ("melting", "C1", "Melting"),
    ):
        for i, event in enumerate(events.get(kind, [])):
            peak = event.peak_temperature
            ax.axvspan(
                _temperature(event.onset_temperature, celsius),
                _temperature(event.endpoint_temperature, celsius),
                color=color,
                alpha=0.15,
                label=name if i == 0 else None,
            )
            text = f"{_temperature(peak, celsius):.1f}"
            if np.isfinite(event.enthalpy):
                text += f"\nΔH = {event.enthalpy:.1f} J/g"
            ax.annotate(
                text,
                (_temperature(peak, celsius), value_at(peak)),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
            )

    ax.set_xlabel(_temperature_label(celsius))
    ax.set_ylabel(_heat_flow_label(exo_up))
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_cp(
    ax: plt.Axes,
    result: CpResult,
    celsius: bool = False,
    label: Optional[str] = None,
) -> None:
    """
    Plot specific heat capacity with its uncertainty.

    Continuous results are drawn as a line with an uncertainty band,
    stepped results as points with error bars.

    Args:
        ax: Matplotlib axes object
        result: CpResult from CpCalculator
        celsius: Plot temperatures in degC
        label: Legend label (default: method name)
    """
    t = _temperature(result.temperature, celsius)
    cp, u = result.specific_heat, result.uncertainty
    label = label or f"Cp ({result.method.value.replace('_', '-')})"

    if result.operation_mode == OperationMode.STEPPED:
        ax.errorbar(t, cp, yerr=u, fmt="o", capsize=4, label=label)
    else:
        (line,) = ax.plot(t, cp, label=label)
        ax.fill_between(t, cp - u, cp + u, color=line.get_color(), alpha=0.25)

    ax.set_xlabel(_temperature_label(celsius))
    ax.set_ylabel("Specific heat capacity (J/(g·K))")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_dsc_analysis(
    analyzer: DSCAnalyzer,
    celsius: bool = False,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """
    Plot the results of DSCAnalyzer.analyze(): raw data with baseline and
    peaks, and the baseline-corrected curve with thermal events.

    Args:
        analyzer: DSCAnalyzer after analyze() has been called
        celsius: Plot temperatures in degC
        figsize: Figure size

    Returns:
        Matplotlib figure

    Raises:
        ValueError: If analyze() has not been called
    """
    if analyzer.baseline is None or analyzer.corrected_heat_flow is None:
        raise ValueError("Call analyze() before plotting")

    exp = analyzer.experiment
    exo_up = analyzer.event_detector.exo_up
    fig, (ax_raw, ax_events) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    plot_dsc_curve(
        ax_raw,
        exp.temperature,
        exp.heat_flow,
        baseline=analyzer.baseline,
        peaks=analyzer.peaks,
        exo_up=exo_up,
        celsius=celsius,
    )
    ax_raw.set_title(f"{exp.sample_name}: DSC curve and baseline")

    plot_thermal_events(
        ax_events,
        exp.temperature,
        analyzer.corrected_heat_flow,
        analyzer.events,
        exo_up=exo_up,
        celsius=celsius,
    )
    ax_events.set_title("Baseline-corrected curve and thermal events")

    fig.tight_layout()
    return fig
