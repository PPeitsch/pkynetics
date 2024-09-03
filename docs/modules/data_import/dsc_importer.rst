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

   This function imports DSC data from various manufacturer formats, providing a consistent output structure regardless of the input format.

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
   - 'heat_flow': Heat flow data in mW
   - 'heat_capacity': Heat capacity data in J/(g·K) (if available)

   Example:
   --------
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
      heat_capacity = dsc_data['heat_capacity']  # May be None if not available

   Note:
   -----
   When using "auto" for manufacturer detection, the function attempts to determine the manufacturer based on the file content. If automatic detection fails, you may need to specify the manufacturer manually.

   Raises:
   -------
   - ValueError: If the file format is not recognized or supported.
   - FileNotFoundError: If the specified file does not exist.
