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
the analyzer calls the same code.

Which one is the API
--------------------

Both. The free functions are the documented entry point, and they are **not
deprecated**: for a one-shot analysis, :func:`analyze_dilatometry_curve` is the
shorter way to say it, and the pieces underneath it
(:func:`find_transformation_limits`, :func:`fit_linear_segments`,
:func:`calculate_transformed_fraction`) are public so that a step can be used
on its own. :class:`DilatometryAnalyzer` is a second way in for when the same
settings are used repeatedly, or when the inputs, the ramp direction and the
limits are wanted after the fact. Neither is scheduled to replace the other.

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
``detection``                    The rules for *where* a transformation is
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

Two questions, two parameters
-----------------------------

An analysis answers two independent questions, and each has its own
parameter:

============= ============================== =================================
Parameter     Question                       Values
============= ============================== =================================
``method``    How far has it transformed?    ``lever``, ``tangent``
``detection`` Where is the transformation?   ``derivative``, ``offset``,
                                             ``second_derivative``,
                                             ``double_tangent``,
                                             ``statistical``
============= ============================== =================================

They cross freely — any detector combines with either method:

.. code-block:: python

   analyze_dilatometry_curve(temperature, strain,
                             method="lever", detection="derivative")

``detection="derivative"`` is the default and is what the module has done
since the limits were unified, so nothing changes for callers who leave it
alone. A detector can also be used on its own through
:func:`get_detector`, which takes a :class:`DetectionContext`.

Detectors
---------

``derivative`` (default)
   The transformation on ``dS/dT``: the limits are where the derivative comes
   back to within ``deviation_fraction`` of its peak excursion. Reproduces
   both shipped runs and the synthetic curve, and is the one to compare
   against.

``offset``
   A fixed departure from the extrapolated baseline, the analogue of the
   0.2 % offset of a tensile test, set by ``offset_fraction`` as a fraction
   of the transformation excursion. Reproducible between runs, and
   **systematically conservative**: it reports where the curve has departed
   measurably, not where the departure begins.

   It also drifts. The departure from an extrapolated line is an integral,
   and the integral of a baseline that is only slightly bowed grows with no
   transformation happening. The Zry-4 heating run fits its initial baseline
   at R² = 0.999 and still departs from it by 1.3 % of the transformation
   excursion *inside the fitting window*, so a 2 % offset puts the start at
   722 °C against a real ~839. The detector measures that floor and warns
   when the offset does not clear it; a result worth trusting is one that
   agrees with ``derivative``.

``second_derivative``
   The extrema of ``d²S/dT²``, where the curve bends away from one baseline
   and onto the other. It answers a genuinely different question: those
   extrema are the points of **maximum curvature**, which lie inside the feet
   of the transformation rather than at them. On a logistic transformation of
   width *w* centred on *T₀* they are at *T₀ ± w·ln(2+√3)*, which is what the
   tests check. Sharp on a clean transition, and noisy on anything else —
   curvature amplifies noise twice over — so ``prominence_fraction`` warns
   when the two extrema are too uneven to be the two ends of one feature.

``double_tangent``
   Three lines: the two baselines, extrapolated, and a tangent through the
   steepest part of the transformation. The limits are the two crossings —
   the extrapolated onset of the thermal-analysis literature, and the
   construction ASTM E228 describes for reading a transformation off a chart.

   It needs no threshold, which is its advantage over ``offset``: nothing to
   tune per run, and its one parameter barely matters — doubling
   ``tangent_window_fraction`` moves the limits by about a kelvin, where a
   comparable change to ``offset_fraction`` moves them by tens. What it
   assumes instead is that the transformation has a *single* steepest part a
   straight line describes; on one that proceeds in two stages the tangent is
   fitted across both and belongs to neither.

   On the Zry-4 heating run it returns 860-926 °C, which is the 855-935 that
   used to be the reference value — the reading off the plot, recovered from
   the data instead of from someone's eye. See :func:`find_transformation_limits`
   for why that differs from the 839 °C the derivative detector reports.

``statistical``
   A departure beyond the prediction interval of the baseline regression.
   Its threshold is the only one here with a stated meaning — a
   false-positive rate — rather than a fraction someone picked.

   It is the most *sensitive* of the five, which is not the same as the most
   accurate. It answers "where does the curve stop being explainable by the
   scatter of the baseline", and on a clean run that is far from where the
   transformation begins: on the synthetic sigmoid the baseline scatter is
   3·10⁻⁸ while the tail of the transformation already reaches 2·10⁻⁷ at
   658 °C — a genuine six-sigma departure, and 0.017 % of the excursion.

   It does **not** fix the drift that limits ``offset``, which is the obvious
   thing to expect of it. The interval widens with the square of the distance
   from the fitting window, but a baseline bow grows faster: on the Zry-4
   heating run it reports 717 °C where ``offset`` reports 739 and the foot is
   at 839.

   Since the method rests on the residuals being scatter rather than
   structure, that is tested rather than assumed, with a Wald-Wolfowitz runs
   test on each baseline. On real dilatometry data the assumption is usually
   violated, and the detector says so. A violation does not by itself mean
   the answer is wrong — the cooling run scores −12.2 on its final baseline
   and still brackets well — it means the stated confidence is no longer the
   false-positive rate it claims to be.

   Use it on a noisy run, where "is this more than noise" is the question
   worth asking.

Choosing between them: ``derivative`` unless there is a reason, ``offset``
when reproducibility between runs matters more than hitting the feet, and
``second_derivative`` when the transition is sharp and clean and what is
wanted is its steepest part, and ``double_tangent`` when the number has to
match what a metallurgist would read off the chart by hand, and
``statistical`` when the data are noisy and the question is whether a
departure is real at all.

What they give on the same three runs:

======================= ================= ================= =================
Detector                Synthetic         Zry-4 heating     Zry-4 cooling
                        (705-795)         (foot at 839-936) (938-757)
======================= ================= ================= =================
``derivative``          706-794           839-936           938-757
``offset``              721-779           739-923           921-783
``second_derivative``   735-765           878-926           931-837
``double_tangent``      728-772           860-926           919-818
``statistical``         658-842           717-927           931-766
======================= ================= ================= =================

They do not agree, and they are not meant to: each is bracketing a
different definition of where a transformation is.

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

The baseline margin, and its dead zones
---------------------------------------

``limits_margin`` (default 0.2) is the fraction of the data at each end taken
as baseline. It carries one consequence that is easy to state: a curve whose
transformation falls inside the first or last 20 % of its range cannot be
analysed in one pass. The full 1050 → 71 °C cooling run is such a case — the
first 20 % of that range is 1050-854 °C, which already contains the
transformation. That is reported as an explicit ``UserWarning``, and the fix
is to window the data to a range with a linear baseline on either side.

It carries a second one that is not obvious at all. The margin has **narrow
dead zones**, and nothing about a curve says where they are. On the Zry-4
heating run the derivative detector returns:

========================= ==========================
Margin                    Limits
========================= ==========================
0.10 - 0.13               702-936 °C
0.14                      835-936 °C
**0.15 - 0.17**           **822-837 °C** (a 15 K
                          bracket on a 97 K
                          transformation)
0.18 - 0.25               839-936 °C
0.26 and up               narrowing progressively
========================= ==========================

Nothing distinguishes 0.16 from 0.20 from the outside, and linearity does not
separate them either: the baselines inside the dead zone fit a line at
R² = 0.9989 against 0.9991 outside it. The cooling run, by contrast, is stable
across 0.10-0.31 — so the width of the safe range is itself a property of the
curve.

What does separate them is stability, which is what ``margin="auto"`` uses:

.. code-block:: python

   find_transformation_limits(temperature, strain, margin="auto")

It runs the detector across the whole range of margins, groups the ones giving
the same answer, and takes the middle of the widest group — the margin whose
answer does not depend on the margin. A dead zone is narrow by nature, which
is what makes it a trap rather than a region, so the answer holding over the
widest stretch is the one to take. Warnings from margins that were tried and
discarded are suppressed; the ones belonging to the chosen margin are raised
normally. It costs about 25 runs of the detector, which is milliseconds, and
it is worth it on data whose safe range is not already known.

The transformed fraction is a reading, not a model
--------------------------------------------------

``transformed_fraction`` runs from 0 at the start limit to 1 at the end limit,
but it is **not guaranteed to be monotonic in between**: it is computed
point-by-point from the strain, so noise in the strain shows up in it. It is
left that way on purpose, since forcing it to rise would make it something
other than a direct reading of the curve.

What the result reports instead is how large the effect is:
``max_backward_step`` is the largest single step backwards, as a fraction of
full scale. On the shipped runs it is 1.03 % (heating) and 0.28 % (cooling).
Above 5 % the tangent method records a warning in ``fit_quality["warnings"]``,
which says the strain is noisy relative to the transformation being measured.

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
