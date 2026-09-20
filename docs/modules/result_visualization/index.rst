Result Visualization Module
===========================

.. currentmodule:: pkynetics.result_visualization

Plotting helpers for the output of the kinetic methods and the technique
analyses. Each one draws the diagnostic plot its method is read from — an
Arrhenius plot for rate constants, a Kissinger plot for peak temperatures,
a lever-rule plot for a dilatometry curve — rather than a generic chart.

Kinetic Plots
-------------

.. autofunction:: plot_arrhenius
.. autofunction:: plot_conversion_vs_temperature
.. autofunction:: plot_derivative_thermogravimetry
.. autofunction:: plot_activation_energy_vs_conversion
.. autofunction:: plot_kissinger
.. autofunction:: plot_jmak_results
.. autofunction:: plot_modified_jmak_results

Model-Specific Plots
--------------------

.. autofunction:: plot_coats_redfern
.. autofunction:: plot_freeman_carroll
.. autofunction:: plot_horowitz_metzger

Dilatometry Plots
-----------------

``plot_dilatometry_analysis`` assembles the four panels; the individual
functions draw onto an axes you supply, for building a custom layout.

.. autofunction:: plot_dilatometry_analysis
.. autofunction:: plot_raw_and_smoothed
.. autofunction:: plot_transformation_points
.. autofunction:: plot_lever_rule
.. autofunction:: plot_transformed_fraction

DSC Plots
---------

The DSC plotting helpers live with the DSC module; see
:doc:`../technique_analysis/dsc`.

Usage Example
-------------

.. code-block:: python

   from pkynetics.result_visualization import plot_dilatometry_analysis
   from pkynetics.technique_analysis.dilatometry import analyze_dilatometry_curve

   results = analyze_dilatometry_curve(temperature, strain, method="lever")
   fig = plot_dilatometry_analysis(
       temperature, strain, smooth_strain, results, "lever"
   )
   fig.savefig("dilatometry.png", dpi=150)

Most of these call ``plt.show()`` and return ``None``; the dilatometry
ones return the figure so it can be saved or embedded.
