Horowitz-Metzger Method
=======================

.. py:function:: horowitz_metzger_method(temperature: np.ndarray, alpha: np.ndarray, heating_rate: float, n: float = 1) -> Tuple[float, float, float, float]

   Perform Horowitz-Metzger analysis to determine kinetic parameters for thermal decomposition reactions.

   :param temperature: Temperature data in Kelvin.
   :type temperature: np.ndarray
   :param alpha: Conversion data.
   :type alpha: np.ndarray
   :param heating_rate: Heating rate in K/min. Required for the pre-exponential factor.
   :type heating_rate: float
   :param n: Reaction order. Default is 1.
   :type n: float
   :return: Tuple containing activation energy (E_a in J/mol), pre-exponential factor (A in min^-1), temperature of maximum decomposition rate (T_s in K), and R-squared value.
   :rtype: Tuple[float, float, float, float]

Theory
------

The Horowitz-Metzger method is used for determining kinetic parameters from thermogravimetric data. It is based on the following equation:

.. math::

   \ln[-\ln(1-\alpha)] = \frac{E_a\theta}{RT_s^2}

where:
   - α is the conversion
   - E_a is the activation energy
   - θ is T - T_s
   - T_s is the temperature at the maximum rate of decomposition
   - R is the gas constant

The slope of that plot gives :math:`E_a`. The pre-exponential factor is not
given by the intercept: it follows from the rate being at its maximum at
:math:`T_s`, which is the Kissinger condition,

.. math::

   \frac{\beta E_a}{RT_s^2} = A \exp\left(-\frac{E_a}{RT_s}\right)

so :math:`A = \frac{\beta E_a}{RT_s^2} \exp\left(\frac{E_a}{RT_s}\right)`,
which is why the heating rate :math:`\beta` has to be supplied.

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from pkynetics.model_fitting_methods import horowitz_metzger_method
   from pkynetics.result_visualization import plot_horowitz_metzger

   # Generate sample data
   temperature = np.linspace(300, 800, 1000)
   alpha = 1 - np.exp(-0.01 * (temperature - 300))
   heating_rate = 10.0  # K/min

   # Perform Horowitz-Metzger analysis
   e_a, a, t_s, r_squared = horowitz_metzger_method(temperature, alpha, heating_rate)

   print(f"Activation energy (E_a): {e_a/1000:.2f} kJ/mol")
   print(f"Pre-exponential factor (A): {a:.2e} min^-1")
   print(f"Temperature of max decomposition rate (T_s): {t_s:.2f} K")
   print(f"R-squared: {r_squared:.4f}")

   # Visualize results (assuming a plot_horowitz_metzger function exists)
   plot_horowitz_metzger(temperature, alpha, heating_rate)

Parameters
----------

- **temperature** (np.ndarray): Temperature data in Kelvin. Must be in ascending order.
- **alpha** (np.ndarray): Conversion data. Must be between 0 and 1 (exclusive).
- **heating_rate** (float): Heating rate in K/min. Must be positive.
- **n** (float, optional): Reaction order. Default is 1.

Returns
-------

A tuple containing:
   1. **e_a** (float): Activation energy in J/mol.
   2. **a** (float): Pre-exponential factor in min^-1.
   3. **t_s** (float): Temperature of maximum decomposition rate in K.
   4. **r_squared** (float): Coefficient of determination (R^2) of the fit.

Raises
------

- **ValueError**: If input arrays have different lengths or contain invalid values.

Notes
-----

- The method uses smoothed derivative data to determine the temperature of maximum decomposition rate (T_s).
- The analysis focuses on the most linear region of the data, typically between 20-80% conversion.
- If not enough points are available in the 20-80% conversion range, the range is expanded to 10-90%.
- The method assumes first-order kinetics by default, but other reaction orders can be specified.
- The method approximates the temperature integral by expanding :math:`1/T` around :math:`T_s`, which biases :math:`E_a` high: on exact first-order non-isothermal data it returns 133.6 kJ/mol for a true 120 kJ/mol (+11 %). Since :math:`A` is exponential in :math:`E_a`, it comes out over an order of magnitude high even though the formula above is exact. Prefer a model-free method (Friedman, KAS, OFW) when the absolute value matters.

See Also
--------

- :func:`kissinger_method`: For non-isothermal kinetics analysis using multiple heating rates
- :func:`coats_redfern_method`: For solid-state reaction kinetics
