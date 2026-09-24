Data Preprocessing Module
=========================

.. currentmodule:: pkynetics.data_preprocessing

Smoothing, normalisation and conversion of a raw signal before it goes into
an analysis.

Smoothing
---------

Every smoothing in the package that is not part of a derivative goes through
:func:`smooth_data`: the DSC and TGA preprocessing, the Freeman-Carroll
method, and ``SignalProcessor.smooth_signal`` in the DSC analysis. The method
is picked with ``method=``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``method``
     - When to use it
   * - ``savgol``
     - The default. Savitzky-Golay keeps the height and position of a peak
       better than an average of the same width.
   * - ``moving_average``
     - A plain centred average, for when the shape of the peak matters less
       than simplicity. Near the ends the window narrows instead of being
       padded, so the ends are not pulled toward zero.
   * - ``lowess``
     - Robust to isolated spikes, and the only one that fits against the real
       abscissa: pass it as ``x``. Where the sampling changes rate, fitting
       against ``x`` rather than the index cuts the error by 10-30 % next to
       the change; elsewhere the two agree. The slowest of the three.

All three are centred, so none of them moves a feature along the temperature
axis. There is no exponential or weighted moving average for that reason:
being causal, they lag the signal.

Derivatives are not smoothing. Smoothing and then differencing point to
point leaves the noise of a two-point difference, which grows as the sample
spacing shrinks; the dilatometry analysis differentiates a local fit instead
(see :doc:`../technique_analysis/dilatometry`).

.. autofunction:: smooth_data
.. autofunction:: available_smoothing_methods

Technique Preprocessing
-----------------------

.. autofunction:: preprocess_dilatometry_data
.. autofunction:: normalize_strain
.. autofunction:: detect_noise_level
.. autofunction:: remove_outliers
.. autofunction:: calculate_dsc_transformed_fraction
.. autofunction:: calculate_tga_transformed_fraction
