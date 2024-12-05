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
        e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(self.t_p, self.beta)

        self.assertAlmostEqual(
            e_a / 1000, self.true_ea / 1000, delta=5
        )  # Compare in kJ/mol, allow 5 kJ/mol difference
        self.assertAlmostEqual(
            np.log10(a), np.log10(self.true_a), delta=0.5
        )  # Compare order of magnitude
        self.assertGreater(r_squared, 0.99)
        self.assertIsInstance(se_e_a, float)
        self.assertIsInstance(se_ln_a, float)

    def test_kissinger_equation(self):
        y = kissinger_equation(self.t_p, self.true_ea, self.true_a, self.beta)
        y_expected = np.log(self.beta / self.t_p**2)
        np.testing.assert_allclose(y, y_expected, rtol=1e-4, atol=1e-4)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            kissinger_method(self.t_p[:-1], self.beta)  # Different lengths

        with self.assertRaises(ValueError):
            kissinger_method(-self.t_p, self.beta)  # Negative temperatures

        with self.assertRaises(ValueError):
            kissinger_method(self.t_p, -self.beta)  # Negative heating rates


if __name__ == "__main__":
    unittest.main()
