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

   This function imports TGA data from various manufacturer formats, providing a consistent output structure regardless of the input format.

   Supported Manufacturers:
   ------------------------
   - TA Instruments
   - Mettler Toledo
   - Netzsch
   - Setaram

   Output Dictionary Keys:
   -----------------------
   - 'temperature': Temperature data in Kelvin
   - 'time': Time data in minutes
   - 'weight': Weight data in milligrams
   - 'weight_percent': Weight percent data

   Example:
   --------
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

   Note:
   -----
   When using "auto" for manufacturer detection, the function attempts to determine the manufacturer based on the file content. If automatic detection fails, you may need to specify the manufacturer manually.

   Raises:
   -------
   - ValueError: If the file format is not recognized or supported.
   - FileNotFoundError: If the specified file does not exist.
