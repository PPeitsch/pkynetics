Data Import Module
==================

The data import module provides functions to import thermal analysis data from various instruments and manufacturers.

.. toctree::
   :maxdepth: 2

   tga_importer
   dsc_importer
   custom_importer

Overview
--------

This module supports the import of Thermogravimetric Analysis (TGA) and Differential Scanning Calorimetry (DSC) data from major manufacturers, including TA Instruments, Mettler Toledo, Netzsch, and Setaram. It also provides a custom importer for flexible data import from other sources.

Key Features:
^^^^^^^^^^^^^

- Automatic detection of manufacturer formats
- Support for multiple file encodings
- Consistent data structure output across different manufacturers
- Custom importer for non-standard data formats
