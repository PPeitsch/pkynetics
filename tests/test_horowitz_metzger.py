"""Unit tests for the Horowitz-Metzger method."""

import unittest
import numpy as np
from model_fitting_methods import horowitz_metzger_method, horowitz_metzger_equation


class TestHorowitzMetzgerMethod(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        self.temperature = np.linspace(300, 900, 1000)
        self.true_e_a = 80000  # J/mol
        self.true_a = 1e7  # min^-1
        self.true_t_s = 740  # K
        self.true_n = 1  # Reaction order
        heating_rate = 10  # K/min

        # Calculate conversion
        r = 8.314  # Gas constant in J/(mol·K)
        k = self.true_a * np.exp(-self.true_e_a / (r * self.temperature))
        time = (self.temperature - self.temperature[0]) / heating_rate
        self.alpha = 1 - np.exp(-(k * time) ** self.true_n)
        self.alpha = np.clip(self.alpha, 0.001, 0.999)  # Ensure alpha is within valid range

    def test_horowitz_metzger_method_accuracy(self):
        e_a, a, t_s, r_squared = horowitz_metzger_method(self.temperature, self.alpha, n=self.true_n)

        self.assertGreater(e_a, 0)  # Activation energy should be positive
        self.assertLess(abs(e_a / 1000 - self.true_e_a / 1000), 20)  # Compare in kJ/mol, allow 20 kJ/mol difference
        self.assertLess(abs(np.log10(a) - np.log10(self.true_a)), 2)  # Compare order of magnitude
        self.assertLess(abs(t_s - self.true_t_s), 30)  # Allow 30 K difference in T_s
        self.assertGreater(r_squared, 0.7)  # R-squared should be reasonably high

    def test_horowitz_metzger_method_with_noise(self):
        np.random.seed(42)  # for reproducibility
        noise = np.random.normal(0, 0.02, len(self.alpha))
        noisy_alpha = np.clip(self.alpha + noise, 0.001, 0.999)

        e_a, a, t_s, r_squared = horowitz_metzger_method(self.temperature, noisy_alpha, n=self.true_n)

        self.assertGreater(e_a, 0)  # Activation energy should be positive
        self.assertLess(abs(e_a / 1000 - self.true_e_a / 1000), 30)  # Allow larger difference with noise
        self.assertLess(abs(np.log10(a) - np.log10(self.true_a)), 3)  # Allow larger difference in order of magnitude
        self.assertLess(abs(t_s - self.true_t_s), 50)  # Allow 50 K difference in T_s with noise
        self.assertGreater(r_squared, 0.6)  # R-squared should still be relatively high

    def test_horowitz_metzger_equation(self):
        r = 8.314  # Gas constant in J/(mol·K)
        theta = np.linspace(-100, 100, 100)
        y = horowitz_metzger_equation(theta, self.true_e_a, r, self.true_t_s)
        
        # Check if the equation produces the expected linear relationship
        slope, intercept = np.polyfit(theta, y, 1)
        self.assertAlmostEqual(slope, self.true_e_a / (r * self.true_t_s**2), delta=1e-6)
        self.assertAlmostEqual(intercept, 0, delta=1e-6)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            horowitz_metzger_method(self.temperature[:-1], self.alpha)  # Different lengths

        with self.assertRaises(ValueError):
            horowitz_metzger_method(self.temperature, np.ones_like(self.alpha))  # Alpha values = 1

        with self.assertRaises(ValueError):
            horowitz_metzger_method(-self.temperature, self.alpha)  # Negative temperatures

        with self.assertRaises(ValueError):
            horowitz_metzger_method(self.temperature, -self.alpha)  # Negative alpha values

    def test_non_first_order_reaction(self):
        # Test for a non-first order reaction (n = 1.5)
        true_n = 1.5
        k = self.true_a * np.exp(-self.true_e_a / (8.314 * self.temperature))
        time = (self.temperature - self.temperature[0]) / 10  # Assuming 10 K/min heating rate
        alpha = 1 - np.exp(-(k * time) ** true_n)
        alpha = np.clip(alpha, 0.001, 0.999)

        e_a, a, t_s, r_squared = horowitz_metzger_method(self.temperature, alpha, n=true_n)

        self.assertGreater(e_a, 0)  # Activation energy should be positive
        self.assertLess(abs(e_a / 1000 - self.true_e_a / 1000), 30)  # Allow 30 kJ/mol difference for non-first order
        self.assertLess(abs(np.log10(a) - np.log10(self.true_a)), 3)  # Compare order of magnitude
        self.assertLess(abs(t_s - self.true_t_s), 50)  # Allow 50 K difference in T_s for non-first order
        self.assertGreater(r_squared, 0.6)  # R-squared should still be relatively high

    def test_edge_cases(self):
        # Test with low activation energy
        low_e_a = 20000  # 20 kJ/mol
        k_low = self.true_a * np.exp(-low_e_a / (8.314 * self.temperature))
        time = (self.temperature - self.temperature[0]) / 10  # Assuming 10 K/min heating rate
        alpha_low_e_a = 1 - np.exp(-(k_low * time) ** self.true_n)
        alpha_low_e_a = np.clip(alpha_low_e_a, 0.001, 0.999)

        e_a, _, _, _ = horowitz_metzger_method(self.temperature, alpha_low_e_a, n=self.true_n)
        self.assertLess(e_a / 1000, 40)  # Fitted E_a should be low (< 40 kJ/mol)

        # Test with high activation energy
        high_e_a = 200000  # 200 kJ/mol
        k_high = self.true_a * np.exp(-high_e_a / (8.314 * self.temperature))
        alpha_high_e_a = 1 - np.exp(-(k_high * time) ** self.true_n)
        alpha_high_e_a = np.clip(alpha_high_e_a, 0.001, 0.999)

        e_a, _, _, _ = horowitz_metzger_method(self.temperature, alpha_high_e_a, n=self.true_n)
        self.assertGreater(e_a / 1000, 150)  # Fitted E_a should be high (> 150 kJ/mol)


if __name__ == '__main__':
    unittest.main()
