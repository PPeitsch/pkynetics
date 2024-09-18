Avrami Method
=============

.. py:function:: avrami_method(time: np.ndarray, relative_crystallinity: np.ndarray) -> Tuple[float, float, float]

   Perform Avrami analysis for isothermal crystallization kinetics.

   :param time: Time data.
   :type time: np.ndarray
   :param relative_crystallinity: Relative crystallinity data.
   :type relative_crystallinity: np.ndarray
   :return: Tuple containing Avrami exponent (n), rate constant (k), and coefficient of determination (R^2).
   :rtype: Tuple[float, float, float]

Theory
------

The Avrami equation describes the kinetics of phase transformations, particularly crystallization processes. It relates the fraction of transformed material to time:

.. math::

   X(t) = 1 - \exp(-(kt)^n)

where:
   - X(t) is the fraction of transformed material at time t
   - k is the rate constant
   - n is the Avrami exponent

The Avrami exponent (n) provides information about the nucleation and growth mechanisms:

- n = 1: One-dimensional growth from instantaneous nuclei
- n = 2: Two-dimensional growth from instantaneous nuclei
- n = 3: Three-dimensional growth from instantaneous nuclei
- n = 4: Three-dimensional growth from sporadic nuclei

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from pkynetics.model_fitting_methods import avrami_method

   # Generate sample data
   time = np.linspace(0, 100, 100)
   relative_crystallinity = 1 - np.exp(-(0.01 * time) ** 2.5)

   # Perform Avrami analysis
   n, k, r_squared = avrami_method(time, relative_crystallinity)

   print(f"Avrami exponent (n): {n:.2f}")
   print(f"Rate constant (k): {k:.4e}")
   print(f"R-squared: {r_squared:.4f}")

Parameters
----------

- **time** (np.ndarray): An array of time values. Must be in ascending order and start from a non-negative value.
- **relative_crystallinity** (np.ndarray): An array of relative crystallinity values. Must be between 0 and 1.

Returns
-------

A tuple containing:
   1. **n** (float): The Avrami exponent.
   2. **k** (float): The rate constant.
   3. **r_squared** (float): The coefficient of determination (R^2) of the fit.

Raises
------

- **ValueError**: If input arrays have different lengths or contain invalid values.

Notes
-----

- The function uses non-linear least squares to fit the Avrami equation to the data.
- Ensure that your time and relative crystallinity data are properly normalized and preprocessed before using this function.
- The R-squared value provides a measure of how well the model fits the data. Values closer to 1 indicate a better fit.

See Also
--------

- :func:`kissinger_method`: For non-isothermal kinetics analysis
- :func:`coats_redfern_method`: For solid-state reaction kinetics
