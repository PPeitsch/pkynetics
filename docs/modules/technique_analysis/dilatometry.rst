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
   relative position of the curve between the two extrapolations.

``tangent``
   Fits tangents to the linear segments and reads the transformed fraction
   against them. ``margin_percent=None`` searches for the widest margin
   whose linear fits still reach the required R².

Both methods locate the transformation limits the same way, on the
derivative ``dS/dT`` rather than on the deviation of the strain from the
extrapolated tangents: a real baseline is never exactly straight, and the
integral of a slight curvature is indistinguishable from the start of a
transformation. ``deviation_fraction`` (default 0.05) sets how much of the
peak excursion of the derivative still counts as transforming — raise it
for a tighter bracket, lower it for a wider one.

On the Zry-4 heating run shipped in ``pkynetics/data`` both methods put the
alpha->beta contraction at 840-929 °C, and on a synthetic transformation
running over 705-795 °C both return 707-793 °C.

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
