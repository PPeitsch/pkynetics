Data Import Module
==================

The data import module provides robust functionality for importing thermal analysis data from various instruments and manufacturers. It offers a consistent interface for handling different file formats and data structures.

.. toctree::
   :maxdepth: 2

   tga_importer
   dsc_importer
   custom_importer

Key Features
------------

1. **Manufacturer-specific Importers**:
   - Support for major thermal analysis instrument manufacturers
   - Automatic detection of manufacturer formats
   - Consistent output structure across different input formats

2. **Flexible Custom Importer**:
   - Handling of non-standard data formats
   - Customizable import parameters
   - Automatic delimiter detection and column name suggestion

3. **Robust Error Handling**:
   - Clear error messages for unsupported formats or missing files
   - Graceful handling of unexpected data structures

4. **Data Preprocessing**:
   - Automatic conversion of units (e.g., temperature to Kelvin)
   - Calculation of derived quantities (e.g., weight percent for TGA data)

Available Importers
-------------------

1. :doc:`tga_importer`: For Thermogravimetric Analysis (TGA) data
2. :doc:`dsc_importer`: For Differential Scanning Calorimetry (DSC) data
3. :doc:`custom_importer`: For flexible import of custom data formats

Supported Manufacturers
-----------------------

- TA Instruments
- Mettler Toledo
- Netzsch
- Setaram

Usage Example
-------------

Here's a quick example of how to use the TGA importer:

.. code-block:: python

   from pkynetics.data_import import tga_importer

   # Import TGA data with automatic manufacturer detection
   tga_data = tga_importer("path/to/tga_data.csv")

   # Access the imported data
   temperature = tga_data['temperature']
   time = tga_data['time']
   weight = tga_data['weight']
   weight_percent = tga_data['weight_percent']

For more detailed information on each importer, please refer to their respective documentation pages.
