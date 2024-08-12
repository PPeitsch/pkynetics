"""Unit tests for the Kissinger method."""

import unittest
import numpy as np
from model_fitting_methods.kissinger import kissinger_method, kissinger_equation, calculate_t_p


class TestKissingerMethod(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        self.true_ea = 150000  # 150 kJ/mol
        self.true_a = 1e15  # min^-1
        self.beta = np.array([2, 5, 10, 20, 50])  # K/min
        self.t_p = calculate_t_p(self.true_ea, self.true_a, self.beta)

    def test_kissinger_method_accuracy(self):
        e_a, a, r_squared = kissinger_method(self.t_p, self.beta)

        self.assertAlmostEqual(e_a / 1000, self.true_ea / 1000, delta=1)  # Compare in kJ/mol
        self.assertAlmostEqual(np.log10(a), np.log10(self.true_a), delta=0.1)  # Compare order of magnitude
        self.assertGreater(r_squared, 0.99)

    def test_kissinger_equation(self):
        r = 8.314  # Gas constant in J/(mol·K)
        ln_ar_ea = np.log(self.true_a * r / self.true_ea)
        y = kissinger_equation(self.t_p, self.true_ea, ln_ar_ea)
        y_expected = np.log(self.beta / self.t_p ** 2)
        np.testing.assert_allclose(y, y_expected, rtol=1e-5, atol=1e-5)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            kissinger_method(self.t_p[:-1], self.beta)  # Different lengths

        with self.assertRaises(ValueError):
            kissinger_method(-self.t_p, self.beta)  # Negative temperatures

        with self.assertRaises(ValueError):
            kissinger_method(self.t_p, -self.beta)  # Negative heating rates


if __name__ == '__main__':
    unittest.main()
