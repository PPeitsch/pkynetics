"""
DSC Signal Smoothing Effects Example
====================================

This example illustrates the effect of applying different levels of smoothing
to a noisy DSC signal. It uses the `SignalProcessor` utility to apply a
Savitzky-Golay filter with varying window sizes.

The workflow includes:
1.  Generating a synthetic signal with a known clean shape and adding significant noise.
2.  Applying three different levels of smoothing (low, medium, high).
3.  Creating a multi-panel plot to clearly visualize the effect of each smoothing
    level on both the signal and its first derivative, avoiding clutter.
"""

import matplotlib.pyplot as plt
import numpy as np

from pkynetics.technique_analysis.dsc import SignalProcessor


def generate_noisy_data(temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generates a clean signal with a sharp peak and adds noise."""
    clean_signal = 1.5 * np.exp(-(((temperature - 400) / 15.0) ** 2))
    noise = np.random.normal(0, 0.1, len(temperature))
    noisy_signal = clean_signal + noise
    return noisy_signal, clean_signal


def main():
    """Main function to run the smoothing effects example."""
    print("--- DSC Signal Smoothing Effects Example ---")

    temperature = np.linspace(300, 500, 401)
    noisy_signal, clean_signal = generate_noisy_data(temperature)

    processor = SignalProcessor()

    # --- Define smoothing levels ---
    smoothing_levels = {
        "Low Smoothing (window=5)": 5,
        "Medium Smoothing (window=21)": 21,
        "High Smoothing (window=51)": 51,
    }

    # --- Visualization with individual subplots for clarity ---
    # Create a figure with a row for each smoothing level, plus one for the original
    fig, axes = plt.subplots(
        len(smoothing_levels) + 1, 2, figsize=(14, 16), sharex=True
    )
    fig.suptitle(
        "Effect of Different Smoothing Levels on DSC Signal and its Derivative",
        fontsize=16,
    )

    # --- Plot 1: Original Noisy Data ---
    ax_sig, ax_deriv = axes[0]
    ax_sig.plot(
        temperature, noisy_signal, label="Noisy Signal", color="gray", alpha=0.8
    )
    ax_sig.plot(temperature, clean_signal, "k--", label="True Clean Signal")
    ax_sig.set_title("Original Signal")
    ax_sig.set_ylabel("Heat Flow (a.u.)")
    ax_sig.legend()
    ax_sig.grid(True)

    ax_deriv.plot(
        temperature,
        np.gradient(noisy_signal, temperature),
        label="Derivative of Noisy Signal",
        color="gray",
        alpha=0.8,
    )
    ax_deriv.plot(
        temperature,
        np.gradient(clean_signal, temperature),
        "k--",
        label="Derivative of True Signal",
    )
    ax_deriv.set_title("Original Derivative")
    ax_deriv.set_ylabel("d(HF)/dT")
    ax_deriv.legend()
    ax_deriv.grid(True)

    # --- Plot each smoothing level in its own row ---
    for i, (name, window) in enumerate(smoothing_levels.items()):
        row_idx = i + 1
        ax_sig, ax_deriv = axes[row_idx]

        # Apply smoothing
        smoothed_signal = processor.smooth_signal(
            noisy_signal, window_length=window, method="savgol"
        )

        # Plot smoothed signal
        ax_sig.plot(
            temperature, noisy_signal, color="gray", alpha=0.3, label="_nolegend_"
        )
        ax_sig.plot(
            temperature,
            smoothed_signal,
            color=f"C{i}",
            linewidth=2,
            label="Smoothed Signal",
        )
        ax_sig.plot(temperature, clean_signal, "k--", label="True Signal")
        ax_sig.set_title(name)
        ax_sig.set_ylabel("Heat Flow (a.u.)")
        ax_sig.legend()
        ax_sig.grid(True)

        # Plot derivative of smoothed signal
        ax_deriv.plot(
            temperature,
            np.gradient(noisy_signal, temperature),
            color="gray",
            alpha=0.3,
            label="_nolegend_",
        )
        ax_deriv.plot(
            temperature,
            np.gradient(smoothed_signal, temperature),
            color=f"C{i}",
            linewidth=2,
            label="Derivative of Smoothed",
        )
        ax_deriv.plot(
            temperature,
            np.gradient(clean_signal, temperature),
            "k--",
            label="Derivative of True",
        )
        ax_deriv.set_title(f"Derivative ({name})")
        ax_deriv.set_ylabel("d(HF)/dT")
        ax_deriv.legend()
        ax_deriv.grid(True)

    # Set common X label
    axes[-1, 0].set_xlabel("Temperature (K)")
    axes[-1, 1].set_xlabel("Temperature (K)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
