"""Unit tests for the JMAK method."""

import unittest

import numpy as np

from src.pkynetics.model_fitting_methods import jmak_equation, jmak_method


class TestJMAKMethod(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        self.time = np.linspace(0, 100, 100)
        self.true_n, self.true_k = 2.5, 0.01
        self.transformed_fraction = jmak_equation(self.time, self.true_k, self.true_n)

    def test_jmak_method_accuracy(self):
        n, k, r_squared = jmak_method(self.time, self.transformed_fraction)

        self.assertAlmostEqual(n, self.true_n, delta=0.05)
        self.assertAlmostEqual(k, self.true_k, delta=0.001)
        self.assertGreater(r_squared, 0.99)

    def test_jmak_method_with_noise(self):
        np.random.seed(42)  # for reproducibility
        noise = np.random.normal(0, 0.01, len(self.time))
        noisy_transformed_fraction = np.clip(self.transformed_fraction + noise, 0, 1)

        n, k, r_squared = jmak_method(self.time, noisy_transformed_fraction)

        self.assertAlmostEqual(n, self.true_n, delta=0.1)
        self.assertAlmostEqual(k, self.true_k, delta=0.005)
        self.assertGreater(r_squared, 0.95)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            jmak_method(self.time[:-1], self.transformed_fraction)  # Different lengths

        with self.assertRaises(ValueError):
            jmak_method(self.time, self.transformed_fraction * 2)  # Values > 1


if __name__ == "__main__":
    unittest.main()
