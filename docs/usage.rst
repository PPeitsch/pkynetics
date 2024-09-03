Usage
=====

This page provides a quick overview of how to use the Pkynetics library for thermal analysis kinetic methods, focusing on the currently implemented features.

Importing Data
--------------

Pkynetics provides importers for TGA and DSC data from various manufacturers:

.. code-block:: python

    from pkynetics.data_import import tga_importer, dsc_importer

    # Import TGA data
    tga_data = tga_importer('path/to/tga_data.csv', manufacturer='auto')

    # Import DSC data
    dsc_data = dsc_importer('path/to/dsc_data.csv', manufacturer='auto')

The `manufacturer` parameter can be set to 'auto', 'TA', 'Mettler', 'Netzsch', or 'Setaram'.

Model Fitting Methods
---------------------

Pkynetics currently implements several model fitting methods:

Avrami Method
^^^^^^^^^^^^^

For isothermal crystallization kinetics:

.. code-block:: python

    from pkynetics.model_fitting_methods import avrami_method

    n, k, r_squared = avrami_method(time, relative_crystallinity)
    print(f"Avrami exponent (n): {n}")
    print(f"Rate constant (k): {k}")
    print(f"R-squared: {r_squared}")

Kissinger Method
^^^^^^^^^^^^^^^^

For non-isothermal kinetics analysis:

.. code-block:: python

    from pkynetics.model_fitting_methods import kissinger_method

    e_a, a, r_squared = kissinger_method(t_p, beta)
    print(f"Activation energy (E_a): {e_a}")
    print(f"Pre-exponential factor (A): {a}")
    print(f"R-squared: {r_squared}")

Coats-Redfern Method
^^^^^^^^^^^^^^^^^^^^

For kinetic analysis:

.. code-block:: python

    from pkynetics.model_fitting_methods import coats_redfern_method

    e_a, a, r_squared = coats_redfern_method(temperature, alpha, heating_rate, n=1)
    print(f"Activation energy (E_a): {e_a}")
    print(f"Pre-exponential factor (A): {a}")
    print(f"R-squared: {r_squared}")

Freeman-Carroll Method
^^^^^^^^^^^^^^^^^^^^^^

For non-isothermal kinetics analysis:

.. code-block:: python

    from pkynetics.model_fitting_methods import freeman_carroll_method

    e_a, n, r_squared = freeman_carroll_method(temperature, alpha, time)
    print(f"Activation energy (E_a): {e_a}")
    print(f"Reaction order (n): {n}")
    print(f"R-squared: {r_squared}")

Horowitz-Metzger Method
^^^^^^^^^^^^^^^^^^^^^^^

For kinetic analysis:

.. code-block:: python

    from pkynetics.model_fitting_methods import horowitz_metzger_method

    e_a, a, t_s, r_squared = horowitz_metzger_method(temperature, alpha)
    print(f"Activation energy (E_a): {e_a}")
    print(f"Pre-exponential factor (A): {a}")
    print(f"Temperature of maximum decomposition rate (T_s): {t_s}")
    print(f"R-squared: {r_squared}")

For more detailed usage instructions and examples, please refer to the API documentation and the Examples section.
