Model Fitting Methods
=====================

The model fitting methods module implements various kinetic analysis techniques based on fitting experimental data to specific kinetic models. These methods are crucial for determining kinetic parameters in thermal analysis.

.. toctree::
   :maxdepth: 2

   avrami
   kissinger
   coats_redfern
   freeman_carroll
   horowitz_metzger

Overview
--------

This module provides implementations of several widely used model fitting methods in thermal analysis kinetics. These methods are used to determine kinetic parameters such as activation energy, pre-exponential factor, and reaction order from experimental data.

Key Features
^^^^^^^^^^^^

- Implementation of popular model fitting methods for various kinetic models
- Support for both isothermal and non-isothermal kinetics analysis
- Robust error handling and input validation to ensure reliable results
- Calculation of goodness-of-fit parameters (e.g., R-squared) for model evaluation
- Consistent interface across different methods for ease of use

Available Methods
^^^^^^^^^^^^^^^^^

1. **Avrami Method**: For isothermal crystallization kinetics
2. **Kissinger Method**: For non-isothermal kinetics analysis
3. **Coats-Redfern Method**: For solid-state reaction kinetics
4. **Freeman-Carroll Method**: For non-isothermal decomposition kinetics
5. **Horowitz-Metzger Method**: For thermal decomposition kinetics

Usage Example
-------------

Here's a basic example of using the Avrami method:

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

For detailed information on each method, please refer to their respective documentation pages.
