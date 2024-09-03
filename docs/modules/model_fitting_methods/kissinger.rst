Kissinger Method
================

.. py:function:: kissinger_method(t_p: np.ndarray, beta: np.ndarray) -> Tuple[float, float, float]

   Perform Kissinger analysis for non-isothermal kinetics.

   :param t_p: Peak temperatures for different heating rates.
   :type t_p: np.ndarray
   :param beta: Heating rates corresponding to the peak temperatures.
   :type beta: np.ndarray
   :return: Tuple containing activation energy (E_a), pre-exponential factor (A), and coefficient of determination (R^2).
   :rtype: Tuple[float, float, float]

   The Kissinger method is used for determining the activation energy and pre-exponential factor in non-isothermal kinetic analysis. It is based on the following equation:

   .. math::

      \ln\left(\frac{\beta}{T_p^2}\right) = \ln\left(\frac{AR}{E_a}\right) - \frac{E_a}{RT_p}

   where:
   - β is the heating rate
   - T_p is the peak temperature
   - E_a is the activation energy
   - A is the pre-exponential factor
   - R is the gas constant

   Example:
   --------
   .. code-block:: python

      import numpy as np
      from pkynetics.model_fitting_methods import kissinger_method

      # Sample data
      t_p = np.array([450, 460, 470, 480])  # Peak temperatures in K
      beta = np.array([5, 10, 15, 20])  # Heating rates in K/min

      # Perform Kissinger analysis
      e_a, a, r_squared = kissinger_method(t_p, beta)

      print(f"Activation energy (E_a): {e_a/1000:.2f} kJ/mol")
      print(f"Pre-exponential factor (A): {a:.2e} min^-1")
      print(f"R-squared: {r_squared:.4f}")

   Note:
   -----
   The Kissinger method assumes that the reaction rate is maximum at the peak temperature and that the reaction follows first-order kinetics.

   Raises:
   -------
   - ValueError: If input arrays have different lengths or contain invalid values.
