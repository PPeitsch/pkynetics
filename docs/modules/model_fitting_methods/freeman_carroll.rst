Freeman-Carroll Method
======================

.. py:function:: freeman_carroll_method(temperature: np.ndarray, alpha: np.ndarray, time: np.ndarray) -> Tuple[float, float, float]

   Perform Freeman-Carroll analysis to determine kinetic parameters.

   :param temperature: Temperature data in Kelvin.
   :type temperature: np.ndarray
   :param alpha: Conversion data.
   :type alpha: np.ndarray
   :param time: Time data in minutes.
   :type time: np.ndarray
   :return: Tuple containing activation energy (E_a in J/mol), reaction order (n), and R-squared value.
   :rtype: Tuple[float, float, float]

   The Freeman-Carroll method is used for determining the activation energy and reaction order from thermogravimetric data. It is based on the following equation:

   .. math::

      \frac{\Delta \ln(d\alpha/dt)}{\Delta \ln(1-\alpha)} = n - \frac{E_a}{R} \cdot \frac{\Delta(1/T)}{\Delta \ln(1-\alpha)}

   where:
   - dα/dt is the rate of conversion
   - α is the conversion
   - T is the temperature
   - E_a is the activation energy
   - n is the reaction order
   - R is the gas constant

   Example:
   --------
   .. code-block:: python

      import numpy as np
      from pkynetics.model_fitting_methods import freeman_carroll_method

      # Generate sample data
      time = np.linspace(0, 100, 1000)
      temperature = 300 + 5 * time
      alpha = 1 - np.exp(-0.01 * time)

      # Perform Freeman-Carroll analysis
      e_a, n, r_squared = freeman_carroll_method(temperature, alpha, time)

      print(f"Activation energy (E_a): {e_a/1000:.2f} kJ/mol")
      print(f"Reaction order (n): {n:.2f}")
      print(f"R-squared: {r_squared:.4f}")

   Note:
   -----
   The Freeman-Carroll method allows for the simultaneous determination of both activation energy and reaction order.

   Raises:
   -------
   - ValueError: If input arrays have different lengths or contain invalid values.
