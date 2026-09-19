"""Unit tests for the Coats-Redfern method."""

import unittest

import numpy as np

from pkynetics.model_fitting_methods import coats_redfern_method
from pkynetics.synthetic_data import generate_coats_redfern_data


class TestCoatsRedfernMethod(unittest.TestCase):
    e_a_true = 150000  # J/mol
    a_true = 1e12  # 1/s

    def check_parameters(self, heating_rate, n):
        temperature, alpha = generate_coats_redfern_data(
            self.e_a_true,
            self.a_true,
            heating_rate,
            (350, 700),
            n=n,
        )

        e_a, a, r_squared, *_ = coats_redfern_method(
            temperature, alpha, heating_rate, n=n
        )

        self.assertAlmostEqual(e_a / self.e_a_true, 1, delta=0.02)
        # A comes in 1/min (heating rate in K/min). Neglecting (1 - 2RT/E_a)
        # underestimates it by about 10 %.
        self.assertAlmostEqual(np.log(a / 60), np.log(self.a_true), delta=0.3)
        self.assertGreater(r_squared, 0.999)

    def test_first_order(self):
        for heating_rate in [5, 10, 20, 40]:
            with self.subTest(heating_rate=heating_rate):
                self.check_parameters(heating_rate, n=1)

    def test_nth_order(self):
        self.check_parameters(10, n=1.5)

    def test_activation_energy_in_joules(self):
        """E_a is in J/mol, not kJ/mol (x is 1000/T)."""
        temperature, alpha = generate_coats_redfern_data(
            self.e_a_true, self.a_true, 10, (350, 700), n=1
        )
        e_a, *_ = coats_redfern_method(temperature, alpha, 10, n=1)
        self.assertGreater(e_a, 1e4)

    def test_invalid_input(self):
        temperature = np.linspace(400, 600, 100)
        with self.assertRaises(ValueError):
            coats_redfern_method(temperature, np.linspace(0, 1, 99), 10)


if __name__ == "__main__":
    unittest.main()
