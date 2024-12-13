from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_raw_and_smoothed(
    ax: plt.Axes,
    temperature: np.ndarray,
    strain: np.ndarray,
    smooth_strain: np.ndarray,
    method: str,
) -> None:
    """
    Plot raw and smoothed dilatometry data.

    Args:
        ax: Matplotlib axes object
        temperature: Temperature data array
        strain: Raw strain data array
        smooth_strain: Smoothed strain data array
        method: Analysis method name for plot title
    """
    ax.plot(temperature, strain, label="Raw data", alpha=0.5)
    ax.plot(temperature, smooth_strain, label="Smoothed data", color="r")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Relative Change")
    ax.set_title(f"Raw and Smoothed Dilatometry Data ({method.capitalize()} Method)")
    ax.legend()
    ax.grid(True)


def plot_transformation_points(
    ax: plt.Axes, temperature: np.ndarray, smooth_strain: np.ndarray, results: Dict
) -> None:
    """
    Plot strain data with transformation points and extrapolations.

    Args:
        ax: Matplotlib axes object
        temperature: Temperature data array
        smooth_strain: Smoothed strain data array
        results: Dictionary containing analysis results
    """
    ax.plot(temperature, smooth_strain, label="Strain")
    ax.plot(
        temperature, results["before_extrapolation"], "--", label="Before extrapolation"
    )
    ax.plot(
        temperature, results["after_extrapolation"], "--", label="After extrapolation"
    )

    points = {
        "Start": ("start_temperature", "green"),
        "End": ("end_temperature", "red"),
        "Mid": ("mid_temperature", "blue"),
    }

    y_range = ax.get_ylim()
    text_y_positions = np.linspace(y_range[0], y_range[1], len(points) + 2)[1:-1]

    for (label, (temp_key, color)), y_pos in zip(points.items(), text_y_positions):
        temp = results[temp_key]
        ax.axvline(temp, color=color, linestyle="--", label=label)
        ax.annotate(
            f"{label}: {temp:.1f}°C",
            xy=(temp, y_pos),
            xytext=(10, 0),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
            ha="left",
            va="center",
        )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Relative Change")
    ax.set_title("Dilatometry Curve with Transformation Points and Extrapolations")
    ax.legend()
    ax.grid(True)


def plot_lever_rule(
    ax: plt.Axes, temperature: np.ndarray, smooth_strain: np.ndarray, results: Dict
) -> None:
    """
    Plot lever rule representation.

    Args:
        ax: Matplotlib axes object
        temperature: Temperature data array
        smooth_strain: Smoothed strain data array
        results: Dictionary containing analysis results
    """
    ax.plot(temperature, smooth_strain, label="Strain")
    ax.plot(
        temperature, results["before_extrapolation"], "--", label="Before extrapolation"
    )
    ax.plot(
        temperature, results["after_extrapolation"], "--", label="After extrapolation"
    )

    mid_temp = results["mid_temperature"]
    mid_strain = np.interp(mid_temp, temperature, smooth_strain)
    mid_before = np.interp(mid_temp, temperature, results["before_extrapolation"])
    mid_after = np.interp(mid_temp, temperature, results["after_extrapolation"])

    ax.plot([mid_temp, mid_temp], [mid_before, mid_after], "k-", label="Lever")
    ax.plot(mid_temp, mid_strain, "ro", label="Mid point")
    ax.annotate(
        f"Mid point: {mid_temp:.1f}°C",
        xy=(mid_temp, mid_strain),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Relative Change")
    ax.set_title("Lever Rule Representation")
    ax.legend()
    ax.grid(True)


def plot_transformed_fraction(
    ax: plt.Axes, temperature: np.ndarray, results: Dict
) -> None:
    """
    Plot transformed fraction vs temperature.

    Args:
        ax: Matplotlib axes object
        temperature: Temperature data array
        results: Dictionary containing analysis results
    """
    ax.plot(temperature, results["transformed_fraction"], label="Transformed Fraction")

    points = {
        "Start": ("start_temperature", "green", 0.0),
        "Mid": ("mid_temperature", "blue", 0.5),
        "End": ("end_temperature", "red", 1.0),
    }

    for label, (temp_key, color, fraction) in points.items():
        temp = results[temp_key]
        ax.axvline(temp, color=color, linestyle="--", label=f"{label}")
        ax.plot(temp, fraction, "o", color=color)
        ax.annotate(
            f"{label}: {temp:.1f}°C\n{fraction * 100:.1f}%",
            xy=(temp, fraction),
            xytext=(10, 0),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
        )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Transformed Fraction")
    ax.set_title("Transformed Fraction vs Temperature")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True)


def plot_dilatometry_analysis(
    temperature: np.ndarray,
    strain: np.ndarray,
    smooth_strain: np.ndarray,
    results: Dict,
    method: str,
) -> plt.Figure:
    """
    Create complete visualization of dilatometry analysis.

    Args:
        temperature: Temperature data array
        strain: Raw strain data array
        smooth_strain: Smoothed strain data array
        results: Dictionary containing analysis results
        method: Analysis method name

    Returns:
        matplotlib.figure.Figure: Complete figure with all plots
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

    plot_raw_and_smoothed(ax1, temperature, strain, smooth_strain, method)
    plot_transformation_points(ax2, temperature, smooth_strain, results)
    plot_lever_rule(ax3, temperature, smooth_strain, results)
    plot_transformed_fraction(ax4, temperature, results)

    plt.tight_layout()
    return fig
