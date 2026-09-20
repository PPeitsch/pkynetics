Dilatometry Analysis
====================

.. currentmodule:: pkynetics.technique_analysis.dilatometry

Locates the start, midpoint and end of a phase transformation on a
dilatometry curve, and computes the transformed fraction from it.

Unlike the DSC module, dilatometry reports temperatures in **°C**, the
convention of the metallurgical literature it serves.

Quick Start
-----------

.. code-block:: python

   from pkynetics.technique_analysis.dilatometry import analyze_dilatometry_curve

   results = analyze_dilatometry_curve(temperature, strain, method="lever")

   print(f"Start: {results['start_temperature']:.1f} degC")
   print(f"Mid:   {results['mid_temperature']:.1f} degC")
   print(f"End:   {results['end_temperature']:.1f} degC")

Cooling segments are detected automatically from the temperature trend,
and reported under ``is_cooling``.

Methods
-------

``lever``
   The lever rule. Extrapolates the linear expansion of the phases before
   and after the transformation and reads the transformed fraction as the
   relative position of the curve between the two extrapolations. The
   transformation limits come from where the curve departs from those
   tangents.

``tangent``
   Fits tangents to the linear segments and locates the transformation
   limits where the curve deviates from them by more than a threshold.
   ``margin_percent=None`` searches for the margin that maximizes the R² of
   the linear fits.

.. warning::

   The **midpoint** is accurate — on a synthetic transformation centred at
   750 °C both methods return it to within 1 K. The **limits** are not yet
   trustworthy: on the same curve the lever method overshoots the
   transformation by ~15 K on each side, and the tangent method returns
   the first and last temperature of the data, i.e. it does not locate
   them at all. Treat ``start_temperature`` and ``end_temperature`` as
   indicative until this is resolved, and prefer ``mid_temperature`` and
   the transformed fraction.

API
---

.. automodule:: pkynetics.technique_analysis.dilatometry
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: pkynetics.technique_analysis.utilities
   :members:
   :undoc-members:
