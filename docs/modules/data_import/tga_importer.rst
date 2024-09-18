TGA Importer
============

.. py:function:: tga_importer(file_path: str, manufacturer: str = "auto") -> Dict[str, np.ndarray]

   Import Thermogravimetric Analysis (TGA) data from common file formats.

   :param file_path: Path to the TGA data file.
   :type file_path: str
   :param manufacturer: Instrument manufacturer. Options: "auto", "TA", "Mettler", "Netzsch", "Setaram". Default is "auto" for automatic detection.
   :type manufacturer: str
   :return: Dictionary containing temperature, time, weight, and weight_percent data.
   :rtype: Dict[str, np.ndarray]

Functionality
-------------

This function imports TGA data from various manufacturer formats, providing a consistent output structure regardless of the input format. It supports automatic detection of the manufacturer format or allows manual specification.

Supported Manufacturers
-----------------------

- TA Instruments
- Mettler Toledo
- Netzsch
- Setaram

Output Dictionary
-----------------

The function returns a dictionary with the following keys:

- 'temperature': Temperature data in Kelvin
- 'time': Time data in minutes
- 'weight': Weight data in milligrams
- 'weight_percent': Weight percent data (calculated if not provided in the original data)

Usage Example
-------------

.. code-block:: python

   from pkynetics.data_import import tga_importer

   # Import TGA data with automatic manufacturer detection
   tga_data = tga_importer("path/to/tga_data.csv")

   # Import TGA data specifying the manufacturer
   tga_data = tga_importer("path/to/tga_data.csv", manufacturer="TA")

   # Access the imported data
   temperature = tga_data['temperature']
   time = tga_data['time']
   weight = tga_data['weight']
   weight_percent = tga_data['weight_percent']

   print(f"Number of data points: {len(temperature)}")
   print(f"Temperature range: {temperature.min():.2f} K to {temperature.max():.2f} K")
   print(f"Total weight loss: {weight[0] - weight[-1]:.2f} mg")

Error Handling
--------------

- Raises `ValueError` if the file format is not recognized or supported.
- Raises `FileNotFoundError` if the specified file does not exist.

Notes
-----

- When using "auto" for manufacturer detection, the function attempts to determine the manufacturer based on the file content. If automatic detection fails, you may need to specify the manufacturer manually.
- The function automatically converts temperature to Kelvin if the original data is in Celsius.
- Weight percent is calculated if not provided in the original data, using the initial weight as 100%.

See Also
--------

- :func:`dsc_importer`: For importing Differential Scanning Calorimetry (DSC) data
- :class:`CustomImporter`: For handling custom data formats
