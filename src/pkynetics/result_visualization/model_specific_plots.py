"""Model-specific plotting functions for Pkynetics."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_coats_redfern(
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    e_a: float,
    a: float,
    r_squared: float,
) -> None:
    """Generate a Coats-Redfern plot for kinetic analysis.

    Args:
        x (np.ndarray): Temperature data (1000/T in K^-1)
        y (np.ndarray): Transformed conversion data (ln(-ln(1-α)/T^2))
        x_fit (np.ndarray): Fitted temperature data subset
        y_fit (np.ndarray): Fitted conversion data subset
        e_a (float): Activation energy in J/mol
        a (float): Pre-exponential factor in min^-1
        r_squared (float): Coefficient of determination

    Returns:
        None: Displays the plot using matplotlib

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values
    """
    plt.figure(figsize=(10, 6))

    # Plot all data points
    plt.scatter(x, y, label="All Data", alpha=0.3, s=10, color="lightblue")

    # Highlight the fitted region
    plt.scatter(x_fit, y_fit, label="Fitted Data", alpha=0.7, s=10, color="blue")

    # Calculate and plot the fit line
    fit = np.polyfit(x_fit, y_fit, 1)
    fit_line = np.poly1d(fit)
    plt.plot(x_fit, fit_line(x_fit), "r-", label="Fit", linewidth=2)

    plt.xlabel("1000/T (K^-1)")
    plt.ylabel("ln(-ln(1-α)/T^2)")
    plt.title("Coats-Redfern Plot")
    plt.legend()
    plt.grid(True)

    # Add text box with results
    textstr = (
        f"E_a = {e_a / 1000:.2f} kJ/mol\nA = {a:.2e} min^-1\nR^2 = {r_squared:.4f}"
    )
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.show()


def plot_freeman_carroll(
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    e_a: float,
    n: float,
    r_squared: float,
) -> None:
    """Generate a Freeman-Carroll plot for kinetic analysis.

    Args:
        x (np.ndarray): Δ(1/T)/Δln(1-α) data
        y (np.ndarray): Δln(dα/dt)/Δln(1-α) data
        x_fit (np.ndarray): Fitted x data subset
        y_fit (np.ndarray): Fitted y data subset
        e_a (float): Activation energy in J/mol
        n (float): Reaction order
        r_squared (float): Coefficient of determination

    Returns:
        None: Displays the plot using matplotlib

    Raises:
        ValueError: If input arrays have different lengths or contain invalid values
    """
    plt.figure(figsize=(10, 6))

    # Plot all data points
    plt.scatter(x, y, label="All Data", alpha=0.3, s=10, color="lightblue")

    # Highlight the fitted region
    plt.scatter(x_fit, y_fit, label="Fitted Data", alpha=0.7, s=10, color="blue")

    # Calculate and plot the fit line
    fit = np.polyfit(x_fit, y_fit, 1)
    fit_line = np.poly1d(fit)
    x_range = np.linspace(min(x_fit), max(x_fit), 100)
    plt.plot(x_range, fit_line(x_range), "r-", label="Fit", linewidth=2)

    plt.xlabel("Δ(1/T) / Δln(1-α)")
    plt.ylabel("Δln(dα/dt) / Δln(1-α)")
    plt.title("Freeman-Carroll Plot")
    plt.legend()
    plt.grid(True)

    # Set axis limits to focus on the relevant data
    x_margin = (max(x_fit) - min(x_fit)) * 0.1
    y_margin = (max(y_fit) - min(y_fit)) * 0.1
    plt.xlim(min(x_fit) - x_margin, max(x_fit) + x_margin)
    plt.ylim(min(y_fit) - y_margin, max(y_fit) + y_margin)

    # Add text box with results
    textstr = f"E_a = {e_a / 1000:.2f} kJ/mol\nn = {n:.2f}\nR^2 = {r_squared:.4f}"
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.show()


def plot_horowitz_metzger(
    temperature: np.ndarray,
    alpha: np.ndarray,
    n: float = 1,
    show_plot: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a Horowitz-Metzger plot for kinetic analysis.

    Args:
        temperature (np.ndarray): Temperature data in Kelvin.
        alpha (np.ndarray): Conversion data.
        n (float): Reaction order. Default is 1.
        show_plot (bool): Whether to show the plot. Default is True.
        ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates a new figure.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes objects of the plot.
    """
    from pkynetics.model_fitting_methods.horowitz_metzger import horowitz_metzger_plot

    # Get plot data
    theta, y, e_a, a, t_s, r_squared, theta_selected, y_selected = (
        horowitz_metzger_plot(temperature, alpha, n)
    )

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot all data points
    ax.scatter(theta, y, label="All Data", alpha=0.3, s=10, color="lightblue")

    # Highlight the fitted region
    ax.scatter(
        theta_selected, y_selected, label="Fitted Data", alpha=0.7, s=10, color="blue"
    )

    # Calculate and plot the fit line
    fit = np.polyfit(theta_selected, y_selected, 1)
    fit_line = np.poly1d(fit)
    ax.plot(theta_selected, fit_line(theta_selected), "r-", label="Fit", linewidth=2)

    ax.set_xlabel("θ (K)")
    ax.set_ylabel("ln(-ln(1-α))")
    ax.set_title("Horowitz-Metzger Plot")
    ax.legend()
    ax.grid(True)

    # Add text box with results
    textstr = (
        f"E_a = {e_a / 1000:.2f} kJ/mol\n"
        f"A = {a:.2e} min^-1\n"
        f"T_s = {t_s:.2f} K\n"
        f"R^2 = {r_squared:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if show_plot:
        plt.show()

    return fig, ax
