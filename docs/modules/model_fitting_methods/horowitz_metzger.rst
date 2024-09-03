Horowitz-Metzger Method
=======================

.. py:function:: horowitz_metzger_method(temperature: np.ndarray, alpha: np.ndarray) -> Tuple[float, float, float, float]

   Perform Horowitz-Metzger analysis to determine kinetic parameters.

   :param temperature: Temperature data in Kelvin.
   :type temperature: np.ndarray
   :param alpha: Conversion data.
   :type alpha: np.ndarray
   :return: Tuple containing activation energy (E_a in J/mol), pre-exponential factor (A in min^-1), temperature of maximum decomposition rate (T_s in K), and R-squared value.
   :rtype: Tuple[float, float, float, float]

   The Horowitz-Metzger method is used for determining kinetic parameters from thermogravimetric data. It is based on the following equation:

   .. math::

      \ln[-\ln(1-\alpha)] = \frac{E_a\theta}{RT_s^2}

   where:
   - α is the conversion
   - E_a is the activation energy
   - θ is T - T_s
   - T_s is the temperature at the maximum rate of decomposition
   - R is the gas constant

   Example:
   --------
   .. code-block:: python

      import numpy as np
      from pkynetics.model_fitting_methods import horowitz_metzger_method

      # Generate sample data
      temperature = np.linspace(300, 800, 1000)
      alpha = 1 - np.exp(-0.01 * (temperature - 300))

      # Perform Horowitz-Metzger analysis
      e_a, a, t_s, r_squared = horowitz_metzger_method(temperature, alpha)

      print(f"Activation energy (E_a): {e_a/1000:.2f} kJ/mol")
      print(f"Pre-exponential factor (A): {a:.2e} min^-1")
      print(f"Temperature of max decomposition rate (T_s): {t_s:.2f} K")
      print(f"R-squared: {r_squared:.4f}")

   Note:
   -----
   The Horowitz-Metzger method assumes that the reaction follows first-order kinetics.

   Raises:
   -------
   - ValueError: If input arrays have different lengths or contain invalid values.
