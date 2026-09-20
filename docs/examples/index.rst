Examples
========

The repository ships runnable examples under `examples/
<https://github.com/PPeitsch/pkynetics/tree/main/examples>`_, each one a
self-contained script that resolves the bundled sample data by itself:

.. code-block:: bash

   python examples/dsc/eicosane_melting_example.py

Data Import
-----------

``examples/data_import/``

- ``importer_examples.py`` — the TGA, DSC and dilatometry importers with
  automatic manufacturer detection.
- ``custom_importer_example.py`` — importing a non-standard format.

DSC
---

``examples/dsc/``

- ``eicosane_melting_example.py`` — melting of eicosane from a TA
  Instruments export: extrapolated onset, peak temperature and enthalpy of
  fusion, against literature values.
- ``polymer_analysis_example.py`` — a glass transition, cold
  crystallization and melting on one curve, which is what the stepped
  baseline exists for.
- ``dsc_baseline_comparison_example.py`` — the five baseline methods on the
  same data.
- ``dsc_smoothing_effects_example.py`` — what smoothing does to the
  detected events.
- ``dsc_heat_capacity_example.py`` and ``stepped_cp_real_data_example.py``
  — Cp by the three-step and stepped methods.
- ``dsc_real_data_analysis_example.py`` — the full pipeline on a real run.

Dilatometry
-----------

``examples/dilatometry/dilatometry_example.py`` — transformation points and
transformed fraction by the lever and tangent methods.

Kinetic Methods
---------------

``examples/kinetic_methods/`` — one script per method: Friedman, KAS, OFW,
Kissinger, Coats-Redfern, Freeman-Carroll, Horowitz-Metzger and JMAK. Each
generates data with known kinetic parameters and reports how well the
method recovers them, which is the honest way to see a method's bias
before trusting it on real data.

Synthetic Data
--------------

``examples/synthetic_data/synthetic_data_example.py`` — generating test
curves with known parameters.
