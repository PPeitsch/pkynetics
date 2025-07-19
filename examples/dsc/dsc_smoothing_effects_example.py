"""
DSC Signal Smoothing Effects Example
====================================

This example illustrates the effect of applying different levels of smoothing
to a noisy DSC signal. It uses the `SignalProcessor` utility to apply a
Savitzky-Golay filter with varying window sizes.

The workflow includes:
1.  Generating a synthetic signal with a known clean shape and adding significant noise.
2.  Applying three different levels of smoothing (low, medium, high).
3.  Plotting the original noisy signal, the true clean signal, and the smoothed
    results.
4.  Plotting the first derivatives of each signal to show how smoothing is
    critical for derivative-based analysis (like Tg detection).
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

    # --- Apply smoothing with different window sizes ---
    smoothing_levels = {
        "Low Smoothing (window=5)": 5,
        "Medium Smoothing (window=21)": 21,
        "High Smoothing (window=51)": 51,
    }

    smoothed_signals = {}
    for name, window in smoothing_levels.items():
        smoothed_signals[name] = processor.smooth_signal(
            noisy_signal, window_length=window, method="savgol"
        )

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Smoothed Signals
    ax1.plot(temperature, noisy_signal, label="Noisy Signal", color="gray", alpha=0.5)
    ax1.plot(temperature, clean_signal, "k--", label="True Clean Signal")
    for name, signal_data in smoothed_signals.items():
        ax1.plot(temperature, signal_data, label=name, linewidth=2)

    ax1.set_title("Effect of Smoothing on Heat Flow Signal")
    ax1.set_ylabel("Heat Flow (a.u.)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Derivatives of Smoothed Signals
    ax2.plot(
        temperature,
        np.gradient(clean_signal, temperature),
        "k--",
        label="Derivative of True Signal",
    )
    for name, signal_data in smoothed_signals.items():
        ax2.plot(
            temperature,
            np.gradient(signal_data, temperature),
            label=f"Derivative of {name}",
            linewidth=2,
        )

    ax2.set_title("Effect of Smoothing on First Derivative")
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("d(HF)/dT (a.u.)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
