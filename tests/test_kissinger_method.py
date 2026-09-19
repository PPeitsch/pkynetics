"""Unit tests for the Kissinger method."""

import unittest

import numpy as np

from pkynetics.model_fitting_methods.kissinger import (
    calculate_t_p,
    kissinger_equation,
    kissinger_method,
)
from pkynetics.synthetic_data import generate_basic_kinetic_data


class TestKissingerMethod(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        self.true_ea = 150000  # 150 kJ/mol
        self.true_a = 1e15  # min^-1
        self.beta = np.array([2, 5, 10, 20, 50])  # K/min
        self.t_p = calculate_t_p(self.true_ea, self.true_a, self.beta)

    def test_kissinger_method_accuracy(self):
        # Peak temperatures in K, straight from calculate_t_p
        e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(self.t_p, self.beta)

        # calculate_t_p solves the same equation: the fit is exact
        self.assertAlmostEqual(e_a / self.true_ea, 1, delta=1e-3)
        self.assertAlmostEqual(np.log(a), np.log(self.true_a), delta=0.01)
        self.assertGreater(r_squared, 0.9999)
        self.assertIsInstance(se_e_a, float)
        self.assertIsInstance(se_ln_a, float)

    def test_peaks_of_first_order_curves(self):
        """Kissinger is exact for first order: use the peaks of d(alpha)/dT."""
        a_per_s = 1e12
        beta = np.array([5.0, 10.0, 20.0, 40.0])
        temperature, alpha = generate_basic_kinetic_data(
            self.true_ea, a_per_s, beta, (350, 700), num_points=20000
        )
        t_p = np.array(
            [t[np.argmax(np.gradient(conv, t))] for t, conv in zip(temperature, alpha)]
        )

        e_a, a, *_ = kissinger_method(t_p, beta)

        self.assertAlmostEqual(e_a / self.true_ea, 1, delta=0.01)
        self.assertAlmostEqual(np.log(a / 60), np.log(a_per_s), delta=0.2)

    def test_rejects_nonpositive_temperatures(self):
        with self.assertRaises(ValueError):
            kissinger_method(np.array([-10.0, 5.0]), np.array([5.0, 10.0]))

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
