DSC Importer
============

.. py:function:: dsc_importer(file_path: str, manufacturer: str = "auto") -> Dict[str, np.ndarray]

   Import Differential Scanning Calorimetry (DSC) data from common file formats.

   :param file_path: Path to the DSC data file.
   :type file_path: str
   :param manufacturer: Instrument manufacturer. Options: "auto", "TA", "Mettler", "Netzsch", "Setaram". Default is "auto" for automatic detection.
   :type manufacturer: str
   :return: Dictionary containing temperature, time, heat_flow, and heat_capacity data.
   :rtype: Dict[str, np.ndarray]

Functionality
-------------

This function imports DSC data from various manufacturer formats, providing a consistent output structure regardless of the input format. It supports automatic detection of the manufacturer format or allows manual specification.

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
- 'heat_flow': Heat flow data in mW
- 'heat_capacity': Heat capacity data in J/(g·K) (if available, otherwise None)

Usage Example
-------------

.. code-block:: python

   from pkynetics.data_import import dsc_importer

   # Import DSC data with automatic manufacturer detection
   dsc_data = dsc_importer("path/to/dsc_data.csv")

   # Import DSC data specifying the manufacturer
   dsc_data = dsc_importer("path/to/dsc_data.csv", manufacturer="TA")

   # Access the imported data
   temperature = dsc_data['temperature']
   time = dsc_data['time']
   heat_flow = dsc_data['heat_flow']
   heat_capacity = dsc_data['heat_capacity']

   print(f"Number of data points: {len(temperature)}")
   print(f"Temperature range: {temperature.min():.2f} K to {temperature.max():.2f} K")
   print(f"Maximum heat flow: {heat_flow.max():.2f} mW")
   if heat_capacity is not None:
       print(f"Average heat capacity: {heat_capacity.mean():.2f} J/(g·K)")
   else:
       print("Heat capacity data not available")

Error Handling
--------------

- Raises `ValueError` if the file format is not recognized or supported.
- Raises `FileNotFoundError` if the specified file does not exist.

Notes
-----

- When using "auto" for manufacturer detection, the function attempts to determine the manufacturer based on the file content. If automatic detection fails, you may need to specify the manufacturer manually.
- The function automatically converts temperature to Kelvin if the original data is in Celsius.
- Heat capacity data may not be available for all DSC measurements. In such cases, the 'heat_capacity' key in the output dictionary will be None.

See Also
--------

- :func:`tga_importer`: For importing Thermogravimetric Analysis (TGA) data
- :class:`CustomImporter`: For handling custom data formats
