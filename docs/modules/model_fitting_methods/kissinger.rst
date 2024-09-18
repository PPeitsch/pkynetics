Kissinger Method
================

.. py:function:: kissinger_method(t_p: np.ndarray, beta: np.ndarray) -> Tuple[float, float, float, float, float]

   Perform Kissinger analysis for non-isothermal kinetics using peak temperature data from multiple heating rates.

   :param t_p: Peak temperatures for different heating rates in K.
   :type t_p: np.ndarray
   :param beta: Heating rates corresponding to the peak temperatures in K/min.
   :type beta: np.ndarray
   :return: Tuple containing activation energy (E_a in J/mol), pre-exponential factor (A in min^-1), standard error of E_a, standard error of ln(A), and coefficient of determination (R^2).
   :rtype: Tuple[float, float, float, float, float]

Theory
------

The Kissinger method is used for determining the activation energy and pre-exponential factor in non-isothermal kinetic analysis. It is based on the following equation:

.. math::

   \ln\left(\frac{\beta}{T_p^2}\right) = \ln\left(\frac{AR}{E_a}\right) - \frac{E_a}{RT_p}

where:
   - β is the heating rate
   - T_p is the peak temperature
   - E_a is the activation energy
   - A is the pre-exponential factor
   - R is the gas constant

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from pkynetics.model_fitting_methods import kissinger_method
   from pkynetics.result_visualization import plot_kissinger

   # Sample data
   t_p = np.array([450, 460, 470, 480])  # Peak temperatures in K
   beta = np.array([5, 10, 15, 20])  # Heating rates in K/min

   # Perform Kissinger analysis
   e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(t_p, beta)

   print(f"Activation energy (E_a): {e_a/1000:.2f} ± {se_e_a/1000:.2f} kJ/mol")
   print(f"Pre-exponential factor (A): {a:.2e} min^-1")
   print(f"95% CI for ln(A): [{np.log(a)-1.96*se_ln_a:.2f}, {np.log(a)+1.96*se_ln_a:.2f}]")
   print(f"R-squared: {r_squared:.4f}")

   # Visualize results
   plot_kissinger(t_p, beta, e_a, a, r_squared)

Parameters
----------

- **t_p** (np.ndarray): Peak temperatures for different heating rates in K.
- **beta** (np.ndarray): Heating rates corresponding to the peak temperatures in K/min.

Returns
-------

A tuple containing:
   1. **e_a** (float): Activation energy in J/mol.
   2. **a** (float): Pre-exponential factor in min^-1.
   3. **se_e_a** (float): Standard error of the activation energy in J/mol.
   4. **se_ln_a** (float): Standard error of ln(A).
   5. **r_squared** (float): Coefficient of determination (R^2) of the fit.

Raises
------

- **ValueError**: If input arrays have different lengths or contain invalid values (e.g., negative temperatures or heating rates).

Notes
-----

- The method assumes that the reaction rate is maximum at the peak temperature.
- It is typically applied to reactions that follow first-order kinetics, but can be used as an approximation for other reaction orders.
- The method provides standard errors for both E_a and ln(A), allowing for the calculation of confidence intervals.
- Use the `plot_kissinger` function to visualize the results and assess the quality of the fit.

See Also
--------

- :func:`horowitz_metzger_method`: For single heating rate thermal decomposition kinetics
- :func:`coats_redfern_method`: For solid-state reaction kinetics
- :func:`plot_kissinger`: For visualizing Kissinger analysis results
