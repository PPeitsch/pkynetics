"""Unit tests for the Avrami method."""

import unittest
import numpy as np
from model_fitting_methods import avrami_method, avrami_equation


class TestAvramiMethod(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        self.time = np.linspace(0, 100, 100)
        self.true_n, self.true_k = 2.5, 0.01
        self.relative_crystallinity = avrami_equation(self.time, self.true_k, self.true_n)

    def test_avrami_method_accuracy(self):
        n, k, r_squared = avrami_method(self.time, self.relative_crystallinity)

        self.assertAlmostEqual(n, self.true_n, delta=0.05)
        self.assertAlmostEqual(k, self.true_k, delta=0.001)
        self.assertGreater(r_squared, 0.99)

    def test_avrami_method_with_noise(self):
        np.random.seed(42)  # for reproducibility
        noise = np.random.normal(0, 0.01, len(self.time))
        noisy_crystallinity = np.clip(self.relative_crystallinity + noise, 0, 1)

        n, k, r_squared = avrami_method(self.time, noisy_crystallinity)

        self.assertAlmostEqual(n, self.true_n, delta=0.1)
        self.assertAlmostEqual(k, self.true_k, delta=0.005)
        self.assertGreater(r_squared, 0.95)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            avrami_method(self.time[:-1], self.relative_crystallinity)  # Different lengths

        with self.assertRaises(ValueError):
            avrami_method(self.time, self.relative_crystallinity * 2)  # Values > 1


if __name__ == '__main__':
    unittest.main()
