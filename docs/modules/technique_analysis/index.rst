Technique Analysis Module
=========================

The technique analysis module turns a raw thermal-analysis curve into the
quantities an experiment is run for: transition temperatures, enthalpies,
heat capacities and transformed fractions. It sits downstream of
:doc:`../data_import/index` and upstream of the kinetic methods.

.. toctree::
   :maxdepth: 2

   dsc
   dilatometry

Units
-----

The module works in **kelvin** and **seconds** throughout, with heat flow
in **mW**, sample mass in **mg** and heating rates in **K/min**. Instrument
exports rarely use these, so convert on the way in:

.. code-block:: python

   temperature = data["temperature"] + 273.15   # degC -> K
   time = data["time"] * 60                     # min -> s

Dilatometry is the exception: it reports transformation temperatures in
**°C**, since that is the convention of the metallurgical literature it
serves.

Sign convention
---------------

DSC instruments disagree about which way an exotherm points, and the file
header is what says so ("Exotherm Up" for TA Instruments). Pass it
explicitly rather than relying on a default:

.. code-block:: python

   from pkynetics.technique_analysis.dsc import ThermalEventDetector

   detector = ThermalEventDetector(exo_up=True)

Available Analyses
------------------

1. :doc:`dsc`: Differential Scanning Calorimetry — baselines, peaks,
   thermal events and heat capacity.
2. :doc:`dilatometry`: Dilatometry — transformation points and transformed
   fraction by the lever and tangent methods.
