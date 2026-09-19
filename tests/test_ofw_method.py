"""Unit tests for the Ozawa-Flynn-Wall method."""

import unittest

import numpy as np

from pkynetics.model_free_methods import ofw_method
from pkynetics.synthetic_data import generate_basic_kinetic_data


class TestOFWMethod(unittest.TestCase):
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

    def test_ofw_method_accuracy(self):
        activation_energy, pre_exp_factor, conv_levels, r_squared = ofw_method(
            self.temperature_data, self.conversion_data, self.heating_rates
        )

        # Doyle's approximation overestimates E_a by about 1 % at E/RT ~ 33
        np.testing.assert_allclose(activation_energy, self.e_a_true, rtol=0.02)
        self.assertGreater(np.min(r_squared), 0.999)

        # pre_exp_factor is A/g(alpha) in 1/min; first order: g = -ln(1 - alpha)
        a_estimated = pre_exp_factor * -np.log(1 - conv_levels) / 60
        np.testing.assert_allclose(np.log(a_estimated), np.log(self.a_true), atol=0.5)

    def test_ofw_method_exported(self):
        """The package exports the function, not the module of the same name."""
        self.assertTrue(callable(ofw_method))

    def test_ofw_method_with_noise(self):
        # Add noise to conversion data
        np.random.seed(42)  # for reproducibility
        noisy_conversion_data = []
        for conv in self.conversion_data:
            noise = np.random.normal(0, 0.01, size=conv.shape)
            noisy_conv = np.clip(conv + noise, 0, 1)
            noisy_conversion_data.append(noisy_conv)

        activation_energy, pre_exp_factor, conv_levels, r_squared = ofw_method(
            self.temperature_data, noisy_conversion_data, self.heating_rates
        )

        # Check if mean activation_energy is still within a reasonable range
        self.assertGreater(np.nanmean(activation_energy), self.e_a_true * 0.6)
        self.assertLess(np.nanmean(activation_energy), self.e_a_true * 1.4)

        # Check if R-squared values are still relatively high, but lower due to noise
        self.assertGreater(np.nanmean(r_squared), 0.7)

        # Check if we have a reasonable number of valid (non-nan) results
        valid_results = np.sum(np.isfinite(activation_energy))
        self.assertGreaterEqual(valid_results / len(activation_energy), 0.8)

    def test_invalid_input(self):
        # Test with inconsistent number of datasets
        with self.assertRaises(ValueError):
            ofw_method(
                self.temperature_data[:-1], self.conversion_data, self.heating_rates
            )

        # Test with temperature and conversion arrays of different lengths
        invalid_temp_data = self.temperature_data.copy()
        invalid_temp_data[0] = invalid_temp_data[0][:-1]
        with self.assertRaises(ValueError):
            ofw_method(invalid_temp_data, self.conversion_data, self.heating_rates)

        # Test with negative temperature values
        invalid_temp_data = self.temperature_data.copy()
        invalid_temp_data[0][0] = -1
        with self.assertRaises(ValueError):
            ofw_method(invalid_temp_data, self.conversion_data, self.heating_rates)

        # Test with conversion values outside [0, 1]
        invalid_conv_data = self.conversion_data.copy()
        invalid_conv_data[0][0] = 1.1
        with self.assertRaises(ValueError):
            ofw_method(self.temperature_data, invalid_conv_data, self.heating_rates)


if __name__ == "__main__":
    unittest.main()
