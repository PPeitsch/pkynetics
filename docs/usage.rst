Usage
=====

This page provides an overview of how to use the Pkynetics library for thermal analysis kinetic methods, focusing on the latest implemented features.

Data Import
-----------

Pkynetics provides importers for TGA and DSC data from various manufacturers:

.. code-block:: python

    from pkynetics.data_import import tga_importer, dsc_importer

    # Import TGA data
    tga_data = tga_importer('path/to/tga_data.csv', manufacturer='auto')

    # Import DSC data
    dsc_data = dsc_importer('path/to/dsc_data.csv', manufacturer='auto')

The ``manufacturer`` parameter can be set to 'auto', 'TA', 'Mettler', 'Netzsch', or 'Setaram'.

Model Fitting Methods
---------------------

Pkynetics implements several model fitting methods with improved data handling:

Coats-Redfern Method
^^^^^^^^^^^^^^^^^^^^

For kinetic analysis:

.. code-block:: python

    from pkynetics.model_fitting_methods import coats_redfern_method
    from pkynetics.result_visualization import plot_coats_redfern

    e_a, a, r_squared, x, y, x_fit, y_fit = coats_redfern_method(temperature, alpha, heating_rate, n=1)
    plot_coats_redfern(x, y, x_fit, y_fit, e_a, a, r_squared)

    print(f"Activation energy (E_a): {e_a/1000:.2f} kJ/mol")
    print(f"Pre-exponential factor (A): {a:.2e} min^-1")
    print(f"R-squared: {r_squared:.4f}")

Result Visualization
--------------------

Pkynetics now offers enhanced visualization capabilities:

.. code-block:: python

    from pkynetics.result_visualization import (
        plot_arrhenius,
        plot_conversion_vs_temperature,
        plot_derivative_thermogravimetry,
        plot_activation_energy_vs_conversion
    )

    # Example: Plot conversion vs temperature
    plot_conversion_vs_temperature([temperature1, temperature2], [alpha1, alpha2], [heating_rate1, heating_rate2])

For more detailed usage instructions and examples of other methods, please refer to the API documentation and the Examples section.
