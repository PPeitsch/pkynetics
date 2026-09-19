"""Unit tests for the synthetic kinetic data generators."""

import unittest

import numpy as np
from scipy.constants import R
from scipy.integrate import solve_ivp

from pkynetics.synthetic_data import generate_basic_kinetic_data


class TestBasicKineticData(unittest.TestCase):
    e_a = 150000  # J/mol
    a = 1e12  # 1/s

    def reference(self, beta, temperature, reaction_model):
        """Integrate d(alpha)/dT = (A / beta) exp(-E_a / RT) f(alpha) numerically."""

        def rate(t, y):
            remaining = np.clip(1 - y, 0, 1)
            f = remaining if reaction_model == "first_order" else remaining**1.5
            return self.a / (beta / 60) * np.exp(-self.e_a / (R * t)) * f

        solution = solve_ivp(
            rate,
            (temperature[0], temperature[-1]),
            [0.0],
            t_eval=temperature,
            rtol=1e-10,
            atol=1e-12,
        )
        return solution.y[0]

    def test_matches_integrated_rate_equation(self):
        heating_rates = np.array([5.0, 40.0])
        for reaction_model in ["first_order", "nth_order"]:
            temperature, alpha = generate_basic_kinetic_data(
                self.e_a,
                self.a,
                heating_rates,
                (350, 700),
                reaction_model=reaction_model,
                num_points=2000,
                n=1.5,
            )
            for beta, t, conv in zip(heating_rates, temperature, alpha):
                with self.subTest(reaction_model=reaction_model, beta=beta):
                    np.testing.assert_allclose(
                        conv, self.reference(beta, t, reaction_model), atol=1e-6
                    )

    def test_peak_shifts_with_heating_rate(self):
        temperature, alpha = generate_basic_kinetic_data(
            self.e_a, self.a, np.array([5.0, 10.0, 20.0]), (350, 700)
        )
        t_half = [np.interp(0.5, conv, t) for t, conv in zip(temperature, alpha)]
        self.assertTrue(np.all(np.diff(t_half) > 0))
        self.assertAlmostEqual(t_half[1], 554.0, delta=0.5)

    def test_unsupported_model(self):
        with self.assertRaises(ValueError):
            generate_basic_kinetic_data(
                self.e_a, self.a, np.array([10.0]), (350, 700), reaction_model="x"
            )


if __name__ == "__main__":
    unittest.main()
