Coats-Redfern Method
====================

.. py:function:: coats_redfern_method(temperature: np.ndarray, alpha: np.ndarray, heating_rate: float, n: float = 1) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

   Perform Coats-Redfern analysis to determine kinetic parameters for solid-state reactions.

   :param temperature: Temperature data in Kelvin.
   :type temperature: np.ndarray
   :param alpha: Conversion data.
   :type alpha: np.ndarray
   :param heating_rate: Heating rate in K/min.
   :type heating_rate: float
   :param n: Reaction order. Default is 1.
   :type n: float
   :return: Tuple containing activation energy (E_a in J/mol), pre-exponential factor (A in min^-1), R-squared value, and arrays for plotting (x, y, x_filtered, y_filtered).
   :rtype: Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Theory
------

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

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from pkynetics.model_fitting_methods import coats_redfern_method
   from pkynetics.result_visualization import plot_coats_redfern

   # Generate sample data
   temperature = np.linspace(300, 800, 500)
   alpha = 1 - np.exp(-0.01 * (temperature - 300))
   heating_rate = 10  # K/min

   # Perform Coats-Redfern analysis
   e_a, a, r_squared, x, y, x_filtered, y_filtered = coats_redfern_method(temperature, alpha, heating_rate)

   print(f"Activation energy (E_a): {e_a/1000:.2f} kJ/mol")
   print(f"Pre-exponential factor (A): {a:.2e} min^-1")
   print(f"R-squared: {r_squared:.4f}")

   # Visualize results
   plot_coats_redfern(x, y, x_filtered, y_filtered, e_a, a, r_squared)

Parameters
----------

- **temperature** (np.ndarray): Temperature data in Kelvin. Must be in ascending order.
- **alpha** (np.ndarray): Conversion data. Must be between 0 and 1.
- **heating_rate** (float): Heating rate in K/min. Must be positive.
- **n** (float, optional): Reaction order. Default is 1.

Returns
-------

A tuple containing:
   1. **e_a** (float): Activation energy in J/mol.
   2. **a** (float): Pre-exponential factor in min^-1.
   3. **r_squared** (float): Coefficient of determination (R^2) of the fit.
   4. **x** (np.ndarray): x-values for plotting (1000/T).
   5. **y** (np.ndarray): y-values for plotting (ln(g(α)/T^2)).
   6. **x_filtered** (np.ndarray): Filtered x-values used for linear regression.
   7. **y_filtered** (np.ndarray): Filtered y-values used for linear regression.

Raises
------

- **ValueError**: If input arrays have different lengths or contain invalid values.

Notes
-----

- The method focuses on the most linear part of the data, typically between 20% to 80% conversion.
- The function automatically removes any invalid points (NaN or inf) before performing the linear regression.
- Use the returned x, y, x_filtered, and y_filtered arrays with the `plot_coats_redfern` function for visualization.

See Also
--------

- :func:`freeman_carroll_method`: For non-isothermal decomposition kinetics
- :func:`plot_coats_redfern`: For visualizing Coats-Redfern analysis results
