"""Unit tests for the Kissinger method."""

import unittest
import numpy as np
from src.pkynetics.model_fitting_methods.kissinger import (
    calculate_t_p,
    kissinger_equation,
    kissinger_method,
)


class TestKissingerMethod(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        self.true_ea = 150000  # 150 kJ/mol
        self.true_a = 1e15  # min^-1
        self.beta = np.array([2, 5, 10, 20, 50])  # K/min
        self.t_p = calculate_t_p(self.true_ea, self.true_a, self.beta)

    def test_kissinger_method_accuracy(self):
        # Convert temperatures to Celsius before analysis
        t_p_celsius = self.t_p - 273.15
        e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(t_p_celsius, self.beta)

        self.assertAlmostEqual(
            e_a / 1000, self.true_ea / 1000, delta=10
        )  # Compare in kJ/mol, allow 10 kJ/mol difference
        self.assertAlmostEqual(
            np.log10(a), np.log10(self.true_a), delta=1.0
        )  # Compare order of magnitude with wider tolerance
        self.assertGreater(r_squared, 0.95)
        self.assertIsInstance(se_e_a, float)
        self.assertIsInstance(se_ln_a, float)

    def test_kissinger_equation(self):
        # Use the actual signature of kissinger_equation
        y = kissinger_equation(t_p=self.t_p, beta=self.beta)
        y_expected = np.log(self.beta / self.t_p**2)
        np.testing.assert_allclose(y, y_expected, rtol=1e-4, atol=1e-4)

    def test_invalid_input(self):
        # Test arrays of different lengths
        with self.assertRaises(ValueError):
            t_p_short = self.t_p[:-1]
            kissinger_method(t_p_short, self.beta)

        # Test negative temperature values (physically impossible)
        with self.assertRaises(ValueError):
            t_p_negative = -np.abs(self.t_p)
            kissinger_method(t_p_negative, self.beta)

        # Test zero or negative heating rates (physically impossible)
        with self.assertRaises(ValueError):
            beta_zero = np.zeros_like(self.beta)
            kissinger_method(self.t_p, beta_zero)

        # Test single value array (need at least two points for regression)
        with self.assertRaises(ValueError):
            kissinger_method(np.array([300]), np.array([10]))


if __name__ == "__main__":
    unittest.main()
