"""Unit tests for the Friedman method."""

import unittest

import numpy as np

from src.pkynetics.model_free_methods import friedman_method


class TestFriedmanMethod(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        self.e_a_true = 150000  # J/mol
        self.a_true = 1e15  # 1/s
        self.heating_rates = [5, 10, 20]  # K/min
        self.r = 8.314  # Gas constant in J/(mol·K)

        self.temperature_data = []
        self.conversion_data = []

        for beta in self.heating_rates:
            t = np.linspace(400, 800, 1000)
            time = (t - t[0]) / beta
            k = self.a_true * np.exp(-self.e_a_true / (self.r * t))
            alpha = 1 - np.exp(-k * time)

            self.temperature_data.append(t)
            self.conversion_data.append(alpha)

    def test_friedman_method_accuracy(self):
        activation_energy, pre_exp_factor, conv_levels, r_squared = friedman_method(
            self.temperature_data, self.conversion_data, self.heating_rates
        )

        # Check if mean activation_energy is close to true e_a
        relative_error_e_a = (
            abs(np.nanmean(activation_energy) - self.e_a_true) / self.e_a_true
        )
        self.assertLess(relative_error_e_a, 0.1)  # Allow for up to 10% relative error

        # Check if mean ln(pre_exp_factor) is within a reasonable range of true ln(a)
        ln_a_estimated = np.nanmean(np.log(pre_exp_factor))
        ln_a_true = np.log(self.a_true)
        relative_error_ln_a = abs(ln_a_estimated - ln_a_true) / abs(ln_a_true)
        self.assertLess(relative_error_ln_a, 0.2)  # Allow for up to 20% relative error

        # Check if R-squared values are high
        self.assertGreater(np.nanmean(r_squared), 0.95)

        # Print diagnostic information
        print(f"True E_a: {self.e_a_true}")
        print(f"Mean estimated E_a: {np.nanmean(activation_energy)}")
        print(f"Relative error E_a: {relative_error_e_a}")
        print(f"True ln(A): {ln_a_true}")
        print(f"Mean estimated ln(A): {ln_a_estimated}")
        print(f"Relative error ln(A): {relative_error_ln_a}")
        print(f"R-squared: {np.nanmean(r_squared)}")

    def test_friedman_method_with_noise(self):
        # Add noise to conversion data
        np.random.seed(42)  # for reproducibility
        noisy_conversion_data = []
        for conv in self.conversion_data:
            noise = np.random.normal(0, 0.01, size=conv.shape)
            noisy_conv = np.clip(conv + noise, 0, 1)
            noisy_conversion_data.append(noisy_conv)

        activation_energy, pre_exp_factor, conv_levels, r_squared = friedman_method(
            self.temperature_data, noisy_conversion_data, self.heating_rates
        )

        # Check if mean activation_energy is still within a reasonable range
        self.assertGreater(np.nanmean(activation_energy), self.e_a_true * 0.7)
        self.assertLess(np.nanmean(activation_energy), self.e_a_true * 1.3)

        # Check if R-squared values are still relatively high, but lower due to noise
        self.assertGreater(np.nanmean(r_squared), 0.7)

        # Check if we have a reasonable number of valid (non-nan) results
        valid_results = np.sum(np.isfinite(activation_energy))
        self.assertGreaterEqual(valid_results / len(activation_energy), 0.8)

    def test_invalid_input(self):
        # Test with inconsistent number of datasets
        with self.assertRaises(ValueError):
            friedman_method(
                self.temperature_data[:-1], self.conversion_data, self.heating_rates
            )

        # Test with temperature and conversion arrays of different lengths
        invalid_temp_data = self.temperature_data.copy()
        invalid_temp_data[0] = invalid_temp_data[0][:-1]
        with self.assertRaises(ValueError):
            friedman_method(invalid_temp_data, self.conversion_data, self.heating_rates)

        # Test with negative temperature values
        invalid_temp_data = self.temperature_data.copy()
        invalid_temp_data[0][0] = -1
        with self.assertRaises(ValueError):
            friedman_method(invalid_temp_data, self.conversion_data, self.heating_rates)

        # Test with conversion values outside [0, 1]
        invalid_conv_data = self.conversion_data.copy()
        invalid_conv_data[0][0] = 1.1
        with self.assertRaises(ValueError):
            friedman_method(
                self.temperature_data, invalid_conv_data, self.heating_rates
            )


if __name__ == "__main__":
    unittest.main()
