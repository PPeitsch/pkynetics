Freeman-Carroll Method
======================

.. py:function:: freeman_carroll_method(temperature: np.ndarray, alpha: np.ndarray, time: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

   Perform Freeman-Carroll analysis to determine kinetic parameters for non-isothermal decomposition reactions.

   :param temperature: Temperature data in Kelvin.
   :type temperature: np.ndarray
   :param alpha: Conversion data.
   :type alpha: np.ndarray
   :param time: Time data in minutes.
   :type time: np.ndarray
   :return: Tuple containing activation energy (E_a in J/mol), reaction order (n), R-squared value, and arrays for plotting (x, y, x_filtered, y_filtered).
   :rtype: Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Theory
------

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

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from pkynetics.model_fitting_methods import freeman_carroll_method
   from pkynetics.result_visualization import plot_freeman_carroll

   # Generate sample data
   time = np.linspace(0, 100, 1000)
   temperature = 300 + 5 * time
   alpha = 1 - np.exp(-0.01 * time)

   # Perform Freeman-Carroll analysis
   e_a, n, r_squared, x, y, x_filtered, y_filtered = freeman_carroll_method(temperature, alpha, time)

   print(f"Activation energy (E_a): {e_a/1000:.2f} kJ/mol")
   print(f"Reaction order (n): {n:.2f}")
   print(f"R-squared: {r_squared:.4f}")

   # Visualize results
   plot_freeman_carroll(x, y, x_filtered, y_filtered, e_a, n, r_squared)

Parameters
----------

- **temperature** (np.ndarray): Temperature data in Kelvin. Must be in ascending order.
- **alpha** (np.ndarray): Conversion data. Must be between 0 and 1.
- **time** (np.ndarray): Time data in minutes. Must be in ascending order.

Returns
-------

A tuple containing:
   1. **e_a** (float): Activation energy in J/mol.
   2. **n** (float): Reaction order.
   3. **r_squared** (float): Coefficient of determination (R^2) of the fit.
   4. **x** (np.ndarray): x-values for plotting (Δ(1/T) / Δln(1-α)).
   5. **y** (np.ndarray): y-values for plotting (Δln(dα/dt) / Δln(1-α)).
   6. **x_filtered** (np.ndarray): Filtered x-values used for linear regression.
   7. **y_filtered** (np.ndarray): Filtered y-values used for linear regression.

Raises
------

- **ValueError**: If input arrays have different lengths or contain invalid values.

Notes
-----

- The method uses smoothed and differentiated data to improve accuracy.
- The analysis focuses on the most relevant part of the reaction, typically between 20% to 80% conversion.
- Outliers are removed using the Interquartile Range (IQR) method to improve the fit.
- Use the returned x, y, x_filtered, and y_filtered arrays with the `plot_freeman_carroll` function for visualization.

See Also
--------

- :func:`coats_redfern_method`: For solid-state reaction kinetics
- :func:`plot_freeman_carroll`: For visualizing Freeman-Carroll analysis results
