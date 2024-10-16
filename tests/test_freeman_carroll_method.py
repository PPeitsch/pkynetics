"""Unit tests for the Freeman-Carroll method."""

import unittest
import numpy as np
from model_fitting_methods import freeman_carroll_method, freeman_carroll_equation


class TestFreemanCarrollMethod(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        self.time = np.linspace(0, 100, 1000)
        self.temperature = 300 + 5 * self.time
        self.true_e_a = 100000  # J/mol
        self.true_a = 1e10  # min^-1
        self.true_n = 1.5

        # Calculate conversion
        r = 8.314  # Gas constant in J/(mol·K)
        k = self.true_a * np.exp(-self.true_e_a / (r * self.temperature))
        self.alpha = 1 - np.exp(-(k * self.time) ** self.true_n)
        self.alpha = np.clip(self.alpha, 0.001, 0.999)  # Ensure alpha is within valid range

    def test_freeman_carroll_method_accuracy(self):
        e_a, n, r_squared = freeman_carroll_method(self.temperature, self.alpha, self.time)

        self.assertGreater(e_a, 0)  # Activation energy should be positive
        self.assertLess(abs(e_a / 1000 - self.true_e_a / 1000), 50)  # Compare in kJ/mol, allow 50 kJ/mol difference
        self.assertLess(abs(n - self.true_n), 0.5)  # Allow 0.5 difference in reaction order
        self.assertGreater(r_squared, 0.8)  # R-squared should be relatively high for this data

    def test_freeman_carroll_method_with_noise(self):
        np.random.seed(42)  # for reproducibility
        noise = np.random.normal(0, 0.005, len(self.alpha))
        noisy_alpha = np.clip(self.alpha + noise, 0.001, 0.999)

        e_a, n, r_squared = freeman_carroll_method(self.temperature, noisy_alpha, self.time)

        self.assertGreater(e_a, 0)  # Activation energy should be positive
        self.assertLess(abs(e_a / 1000 - self.true_e_a / 1000), 70)  # Allow larger difference with noise
        self.assertLess(abs(n - self.true_n), 0.7)  # Allow larger difference in reaction order
        self.assertGreater(r_squared, 0.7)  # R-squared should still be relatively high

    def test_freeman_carroll_equation(self):
        x = np.linspace(0, 1, 100)
        y = freeman_carroll_equation(x, self.true_e_a, self.true_n)
        
        # Check if the equation produces the expected linear relationship
        slope, intercept = np.polyfit(x, y, 1)
        self.assertAlmostEqual(slope, -self.true_e_a / 8.314, delta=1)
        self.assertAlmostEqual(intercept, self.true_n, delta=0.01)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            freeman_carroll_method(self.temperature[:-1], self.alpha, self.time)  # Different lengths

        with self.assertRaises(ValueError):
            freeman_carroll_method(self.temperature, np.ones_like(self.alpha), self.time)  # Alpha values = 1

        with self.assertRaises(ValueError):
            freeman_carroll_method(-self.temperature, self.alpha, self.time)  # Negative temperatures

        with self.assertRaises(ValueError):
            freeman_carroll_method(self.temperature, self.alpha, -self.time)  # Negative time


if __name__ == '__main__':
    unittest.main()
