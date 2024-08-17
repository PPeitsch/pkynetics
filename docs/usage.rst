Usage
=====

This page provides a quick overview of how to use the Pkynetics library for thermal analysis kinetic methods.

Importing Data
--------------

Pkynetics provides importers for TGA and DSC data:

.. code-block:: python

    from pkynetics.data_import import tga_importer, dsc_importer

    # Import TGA data
    tga_data = tga_importer('path/to/tga_data.csv')

    # Import DSC data
    dsc_data = dsc_importer('path/to/dsc_data.csv')

Model Fitting Methods
---------------------

Here's an example of using the Avrami method for isothermal crystallization kinetics:

.. code-block:: python

    from pkynetics.model_fitting_methods import avrami_method

    n, k, r_squared = avrami_method(time_data, crystallinity_data)
    print(f"Avrami exponent (n): {n}")
    print(f"Rate constant (k): {k}")
    print(f"R-squared: {r_squared}")

Model Free Methods
------------------

Here's an example of using the Friedman method:

.. code-block:: python

    from pkynetics.model_free_methods import friedman_method

    activation_energy, pre_exp_factor = friedman_method(temp_data, conversion_data, heating_rates)
    print(f"Activation Energy: {activation_energy}")
    print(f"Pre-exponential factor: {pre_exp_factor}")

For more detailed usage instructions and examples, please refer to the API documentation and the Examples section.
