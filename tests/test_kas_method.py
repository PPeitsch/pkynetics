"""Unit tests for the Kissinger-Akahira-Sunose (KAS) method."""

import unittest

import numpy as np

from pkynetics.model_free_methods import kas_method
from pkynetics.synthetic_data import generate_basic_kinetic_data


class TestKASMethod(unittest.TestCase):
    def setUp(self):
        # Non-isothermal first-order data, integrated over temperature
        self.e_a_true = 150000  # J/mol
        self.a_true = 1e12  # 1/s
        self.heating_rates = [5, 10, 20, 40]  # K/min

        self.temperature_data, self.conversion_data = generate_basic_kinetic_data(
            self.e_a_true,
            self.a_true,
            np.array(self.heating_rates, dtype=np.float64),
            (350, 700),
            num_points=5000,
        )

    def test_kas_method_accuracy(self):
        activation_energy, pre_exp_factor, conv_levels, r_squared = kas_method(
            self.temperature_data, self.conversion_data, self.heating_rates
        )

        np.testing.assert_allclose(activation_energy, self.e_a_true, rtol=0.02)
        self.assertGreater(np.min(r_squared), 0.999)

        # pre_exp_factor is A/g(alpha) in 1/min; first order: g = -ln(1 - alpha)
        a_estimated = pre_exp_factor * -np.log(1 - conv_levels) / 60
        np.testing.assert_allclose(np.log(a_estimated), np.log(self.a_true), atol=0.5)

    def test_kas_method_with_noise(self):
        rng = np.random.default_rng(42)
        noisy_conversion_data = [
            np.clip(conv + rng.normal(0, 0.01, size=conv.shape), 0, 1)
            for conv in self.conversion_data
        ]

        activation_energy, _, _, r_squared = kas_method(
            self.temperature_data, noisy_conversion_data, self.heating_rates
        )

        self.assertLess(
            abs(np.mean(activation_energy) - self.e_a_true) / self.e_a_true, 0.05
        )
        self.assertGreater(np.mean(r_squared), 0.9)

    def test_invalid_input(self):
        # Test with inconsistent number of datasets
        with self.assertRaises(ValueError):
            kas_method(
                self.temperature_data[:-1], self.conversion_data, self.heating_rates
            )

        # Test with temperature and conversion arrays of different lengths
        invalid_temp_data = self.temperature_data.copy()
        invalid_temp_data[0] = invalid_temp_data[0][:-1]
        with self.assertRaises(ValueError):
            kas_method(invalid_temp_data, self.conversion_data, self.heating_rates)

        # Test with negative temperature values
        invalid_temp_data = self.temperature_data.copy()
        invalid_temp_data[0][0] = -1
        with self.assertRaises(ValueError):
            kas_method(invalid_temp_data, self.conversion_data, self.heating_rates)

        # Test with conversion values outside [0, 1]
        invalid_conv_data = self.conversion_data.copy()
        invalid_conv_data[0][0] = 1.1
        with self.assertRaises(ValueError):
            kas_method(self.temperature_data, invalid_conv_data, self.heating_rates)


if __name__ == "__main__":
    unittest.main()
