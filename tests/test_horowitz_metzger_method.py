"""Unit tests for the Horowitz-Metzger method."""

import unittest

import numpy as np
from scipy.constants import R
from scipy.integrate import cumulative_trapezoid

from pkynetics.model_fitting_methods import (
    horowitz_metzger_equation,
    horowitz_metzger_method,
)


def first_order_conversion(temperature, e_a, a, heating_rate):
    """Exact first-order conversion for a linear temperature ramp.

    alpha = 1 - exp(-int k dt), not 1 - exp(-(k t)^n): the latter evaluates
    an isothermal solution at a moving temperature and is not a solution of
    the non-isothermal rate equation (see K6).
    """
    time = (temperature - temperature[0]) / heating_rate
    k = a * np.exp(-e_a / (R * temperature))
    return 1 - np.exp(-cumulative_trapezoid(k, time, initial=0))


class TestHorowitzMetzgerMethod(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        self.temperature = np.linspace(300, 800, 1000)
        self.true_e_a = 120000  # J/mol
        self.true_a = 1e10  # min^-1
        self.true_n = 1  # Reaction order
        self.heating_rate = 10  # K/min

        # Calculate conversion
        r = 8.314  # Gas constant in J/(mol·K)
        k = self.true_a * np.exp(-self.true_e_a / (r * self.temperature))
        time = (self.temperature - self.temperature[0]) / self.heating_rate
        self.alpha = 1 - np.exp(-((k * time) ** self.true_n))
        self.alpha = np.clip(
            self.alpha, 0.001, 0.999
        )  # Ensure alpha is within valid range

    def test_horowitz_metzger_method_accuracy(self):
        e_a, a, t_s, r_squared = horowitz_metzger_method(
            self.temperature, self.alpha, self.heating_rate, n=self.true_n
        )

        self.assertGreater(e_a, 0)  # Activation energy should be positive
        self.assertLess(
            abs(e_a / 1000 - self.true_e_a / 1000), 30
        )  # Compare in kJ/mol, allow 30 kJ/mol difference
        self.assertLess(
            abs(np.log10(a) - np.log10(self.true_a)), 3
        )  # Compare order of magnitude
        self.assertGreater(r_squared, 0.95)  # R-squared should be high for perfect data

    def test_horowitz_metzger_method_with_noise(self):
        np.random.seed(42)  # for reproducibility
        noise = np.random.normal(0, 0.01, len(self.alpha))
        noisy_alpha = np.clip(self.alpha + noise, 0.001, 0.999)

        e_a, a, t_s, r_squared = horowitz_metzger_method(
            self.temperature, noisy_alpha, self.heating_rate, n=self.true_n
        )

        self.assertGreater(e_a, 0)  # Activation energy should be positive
        self.assertLess(
            abs(e_a / 1000 - self.true_e_a / 1000), 40
        )  # Allow larger difference with noise
        self.assertLess(
            abs(np.log10(a) - np.log10(self.true_a)), 4
        )  # Allow larger difference in order of magnitude
        self.assertGreater(r_squared, 0.9)  # R-squared should still be relatively high

    def test_horowitz_metzger_equation(self):
        r = R
        theta = np.linspace(-100, 100, 100)
        y = horowitz_metzger_equation(theta, self.true_e_a, r, 500)

        # Check if the equation produces the expected linear relationship
        slope, intercept = np.polyfit(theta, y, 1)
        self.assertAlmostEqual(slope, self.true_e_a / (r * 500**2), delta=1e-6)
        self.assertAlmostEqual(intercept, 0, delta=1e-6)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            horowitz_metzger_method(
                self.temperature[:-1], self.alpha, self.heating_rate
            )  # Different lengths

        with self.assertRaises(ValueError):
            horowitz_metzger_method(
                self.temperature, np.ones_like(self.alpha), self.heating_rate
            )  # Alpha values = 1

        with self.assertRaises(ValueError):
            horowitz_metzger_method(
                -self.temperature, self.alpha, self.heating_rate
            )  # Negative temperatures

        with self.assertRaises(ValueError):
            horowitz_metzger_method(
                self.temperature, -self.alpha, self.heating_rate
            )  # Negative alpha values

    def test_pre_exponential_satisfies_the_rate_maximum(self):
        """A must satisfy the Kissinger condition at T_s (K9).

        A was computed as exp(intercept + E_a/(R*T_s)), which ignores the
        condition that fixes it and does not involve the heating rate at
        all; the example came out 340x too large.
        """
        temperature = np.linspace(400, 800, 2000)
        alpha = first_order_conversion(temperature, 120000.0, 1e10, 10.0)
        usable = (alpha > 1e-3) & (alpha < 1 - 1e-3)

        e_a, a, t_s, _ = horowitz_metzger_method(
            temperature[usable], alpha[usable], 10.0
        )

        expected = 10.0 * e_a / (R * t_s**2) * np.exp(e_a / (R * t_s))
        self.assertAlmostEqual(np.log10(a), np.log10(expected), places=9)

    def test_pre_exponential_scales_with_the_heating_rate(self):
        """A is proportional to beta at a fixed T_s (K9).

        The old formula did not receive the heating rate, so the same curve
        analysed as 5 or 20 K/min returned the same A.
        """
        temperature = np.linspace(400, 800, 2000)
        alpha = first_order_conversion(temperature, 120000.0, 1e10, 10.0)
        usable = (alpha > 1e-3) & (alpha < 1 - 1e-3)

        _, a_5, _, _ = horowitz_metzger_method(temperature[usable], alpha[usable], 5.0)
        _, a_20, _, _ = horowitz_metzger_method(
            temperature[usable], alpha[usable], 20.0
        )

        self.assertAlmostEqual(a_20 / a_5, 4.0, places=9)

    def test_pre_exponential_is_exact_given_the_true_activation_energy(self):
        """Fed the true E_a, the formula reproduces A to better than 1 %.

        This separates the two error sources: the formula is right, and what
        is left over is the method's own bias in E_a (~+11 %), which A is
        exponential in.
        """
        e_a_true, a_true, beta = 120000.0, 1e10, 10.0
        temperature = np.linspace(400, 800, 2000)
        alpha = first_order_conversion(temperature, e_a_true, a_true, beta)
        usable = (alpha > 1e-3) & (alpha < 1 - 1e-3)

        _, _, t_s, _ = horowitz_metzger_method(temperature[usable], alpha[usable], beta)

        a = beta * e_a_true / (R * t_s**2) * np.exp(e_a_true / (R * t_s))
        self.assertLess(abs(a / a_true - 1), 0.01)

    def test_heating_rate_must_be_positive(self):
        with self.assertRaises(ValueError):
            horowitz_metzger_method(self.temperature, self.alpha, 0.0)

    def test_non_first_order_reaction(self):
        """
        TODO: This test has been temporarily removed due to implementation uncertainties.
        Need to review the H-M method implementation for non-first order reactions.
        """
        pass

    def test_edge_cases(self):
        """
        TODO: This test has been temporarily removed due to implementation uncertainties.
        Need to review the theoretical basis for testing low Ea cases and implement
        appropriate assertions.
        """
        pass


if __name__ == "__main__":
    unittest.main()
