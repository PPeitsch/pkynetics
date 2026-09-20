DSC Analysis
============

.. currentmodule:: pkynetics.technique_analysis.dsc

Differential scanning calorimetry: baseline correction, peak
characterization, thermal event detection and heat capacity.

Quick Start
-----------

:class:`DSCAnalyzer` runs the whole pipeline — it corrects the baseline,
finds the peaks and detects the thermal events:

.. code-block:: python

   import numpy as np
   from pkynetics.data_import import dsc_importer
   from pkynetics.technique_analysis.dsc import (
       DSCAnalyzer,
       DSCExperiment,
       ThermalEventDetector,
   )

   data = dsc_importer("sample.txt")

   experiment = DSCExperiment(
       temperature=data["temperature"] + 273.15,  # K
       heat_flow=data["heat_flow"],               # mW
       time=data["time"] * 60,                    # s
       mass=9.0,                                  # mg
       heating_rate=10.0,                         # K/min
       sample_name="Eicosane",
   )

   analyzer = DSCAnalyzer(
       experiment, event_detector=ThermalEventDetector(exo_up=True)
   )
   results = analyzer.analyze(baseline_method="polynomial", degree=2)

   for peak in results["peaks"]:
       print(f"{peak.peak_temperature - 273.15:.1f} degC, {peak.enthalpy:.1f} J/g")

``mass`` and ``heating_rate`` are what make the enthalpy a J/g figure
rather than an area; without them it is ``NaN``.

Baselines
---------

Five methods, selected with ``baseline_method``:

``linear``
   A straight line through the event-free regions. The safe default when
   the run is short and the drift is small.

``polynomial``
   A polynomial of the given ``degree`` through the event-free regions.

``spline``
   A smoothing spline. ``smoothing`` is **relative to the noise**, not an
   absolute residual budget: ``1.0`` means "follow the trend, not the
   noise", and lower values fit more tightly. It follows curvature a
   polynomial cannot, at the cost of being able to absorb a broad peak.

``asymmetric``
   Asymmetric least squares (Eilers & Boelens). Needs no event-free
   regions, which is what to reach for when events cover most of the scan.

``auto``
   Fits linear and polynomial baselines to the detected event-free regions
   and keeps the one with the lowest BIC, so a higher degree is only chosen
   when it clearly earns it. Spline and asymmetric are deliberately not
   candidates: they are not constrained to event-free regions.

Glass transitions and stepped baselines
---------------------------------------

A glass transition is a **step** in the baseline, and a baseline fitted
across the whole curve cuts through that step and distorts every event
after it. ``analyze()`` therefore locates the transition on the raw curve
and fits the baseline to each side of the step separately, interpolating
within it, the way ASTM and ISO draw it. Pass ``detect_steps=False`` to
fit one baseline over the whole curve instead.

Because the stepped baseline removes the step by construction, the
reported ΔCp is the one measured on the raw curve.

API
---

.. automodule:: pkynetics.technique_analysis.dsc.core
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pkynetics.technique_analysis.dsc.baseline
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pkynetics.technique_analysis.dsc.peak_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pkynetics.technique_analysis.dsc.thermal_events
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pkynetics.technique_analysis.dsc.heat_capacity
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pkynetics.technique_analysis.dsc.signal_stability
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pkynetics.technique_analysis.dsc.utilities
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pkynetics.technique_analysis.dsc.visualization
   :members:
   :undoc-members:

Types
-----

.. automodule:: pkynetics.technique_analysis.dsc.types
   :members:
   :undoc-members:
   :show-inheritance:
