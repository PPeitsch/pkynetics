Coats-Redfern Method
====================

.. py:function:: coats_redfern_method(temperature: np.ndarray, alpha: np.ndarray, heating_rate: float, n: float = 1) -> Tuple[float, float, float]

   Perform Coats-Redfern analysis to determine kinetic parameters.

   :param temperature: Temperature data in Kelvin.
   :type temperature: np.ndarray
   :param alpha: Conversion data.
   :type alpha: np.ndarray
   :param heating_rate: Heating rate in K/min.
   :type heating_rate: float
   :param n: Reaction order. Default is 1.
   :type n: float
   :return: Tuple containing activation energy (E_a in J/mol), pre-exponential factor (A in min^-1), and R-squared value.
   :rtype: Tuple[float, float, float]

   The Coats-Redfern method is an integral method used for the determination of kinetic parameters from thermogravimetric data. It is based on the following equation:

   .. math::

      \ln\left(\frac{g(\alpha)}{T^2}\right) = \ln\left(\frac{AR}{\beta E_a}\right) - \frac{E_a}{RT}

   where:
   - g(α) is the integral form of the reaction model
   - T is the temperature
   - β is the heating rate
   - E_a is the activation energy
   - A is the pre-exponential factor
   - R is the gas constant

   Example:
   --------
   .. code-block:: python

      import numpy as np
      from pkynetics.model_fitting_methods import coats_redfern_method

      # Generate sample data
      temperature = np.linspace(300, 800, 500)
      alpha = 1 - np.exp(-0.01 * (temperature - 300))
      heating_rate = 10  # K/min

      # Perform Coats-Redfern analysis
      e_a, a, r_squared = coats_redfern_method(temperature, alpha, heating_rate)

      print(f"Activation energy (E_a): {e_a/1000:.2f} kJ/mol")
      print(f"Pre-exponential factor (A): {a:.2e} min^-1")
      print(f"R-squared: {r_squared:.4f}")

   Note:
   -----
   The Coats-Redfern method assumes that the reaction follows a specific reaction model, which is determined by the value of n (reaction order).

   Raises:
   -------
   - ValueError: If input arrays have different lengths or contain invalid values.
