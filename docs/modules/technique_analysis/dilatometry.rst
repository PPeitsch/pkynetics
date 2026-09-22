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

For repeated analyses, :class:`DilatometryAnalyzer` holds the settings once
instead of taking them again on every call, and keeps what the analysis went
through:

.. code-block:: python

   from pkynetics.technique_analysis import DilatometryAnalyzer

   analyzer = DilatometryAnalyzer(limits_margin=0.15)

   for run in runs:
       results = analyzer.analyze(run.temperature, run.strain)
       start_idx, end_idx = analyzer.limits

It returns the same mapping the function does. The two are interchangeable:
the analyzer calls the same code, and the free functions are not going
anywhere.

Module layout
-------------

The analysis is a package of focused modules rather than one file:

================================ ==============================================
Module                           What it holds
================================ ==============================================
``core``                         :func:`analyze_dilatometry_curve`,
                                 :class:`DilatometryAnalyzer`
``transformation_points``        Where a transformation starts and ends
``linear_segments``              The baselines on either side, and their fits
``transformed_fraction``         How far the transformation has gone
``methods.lever``,               The two analysis methods
``methods.tangent``
``curve_features``               The derivative and the noise of the curve
``utilities``, ``types``         Numerical helpers and type definitions
================================ ==============================================

Everything public is importable from
``pkynetics.technique_analysis.dilatometry`` and from
``pkynetics.technique_analysis``, exactly as before the split.

One thing worth knowing when reading the signatures: the free functions name
the same setting differently depending on where they grew up. The margin used
to *locate* a transformation appears as ``margin``, ``find_inflection_margin``
and ``limits_margin``; the margin used to *fit the baselines* appears as
``margin_percent`` and ``margin_percent_fraction``. They are two settings, and
:class:`DilatometryAnalyzer` names them once, as ``limits_margin`` and
``baseline_margin``.

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

``dS/dT`` itself comes from a local polynomial fit rather than from
differencing a smoothed signal, because the noise of a two-point difference
grows as the sample spacing shrinks — on a run recorded every 0.25 K it was
the noise, not the transformation, that set the threshold.

On the Zry-4 heating run both methods put the alpha->beta contraction at
839-936 °C; on a synthetic transformation running over 705-795 °C both return
706-794 °C; and on a cooling run windowed to 1040-700 °C, against a
transformation from ~945 to ~760 °C, they return 938-757 °C.

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
