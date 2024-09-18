Custom Importer
===============

.. py:class:: CustomImporter

   A flexible importer for custom data formats in thermal analysis.

   This class provides methods to import data from non-standard file formats, offering customizable options for data parsing and preprocessing.

   .. py:method:: __init__(file_path: str, column_names: List[str], separator: str = ',', decimal: str = '.', encoding: str = 'utf-8', skiprows: int = 0)

      Initialize the CustomImporter.

      :param file_path: Path to the data file.
      :param column_names: List of column names in the order they appear in the file.
      :param separator: Column separator in the file. Defaults to ','.
      :param decimal: Decimal separator used in the file. Defaults to '.'.
      :param encoding: File encoding. Defaults to 'utf-8'.
      :param skiprows: Number of rows to skip at the beginning of the file. Defaults to 0.

   .. py:method:: import_data() -> Dict[str, np.ndarray]

      Import data from the file.

      :return: Dictionary containing the imported data.
      :rtype: Dict[str, np.ndarray]
      :raises ValueError: If the file format is not recognized or supported.
      :raises FileNotFoundError: If the specified file does not exist.

   .. py:staticmethod:: detect_delimiter(file_path: str, num_lines: int = 5) -> str

      Attempt to detect the delimiter used in the file.

      :param file_path: Path to the data file.
      :param num_lines: Number of lines to check. Defaults to 5.
      :return: Detected delimiter.
      :rtype: str
      :raises ValueError: If unable to detect the delimiter.

   .. py:staticmethod:: suggest_column_names(file_path: str, delimiter: Optional[str] = None) -> List[str]

      Suggest column names based on the first row of the file.

      :param file_path: Path to the data file.
      :param delimiter: Delimiter to use. If None, will attempt to detect.
      :return: Suggested column names.
      :rtype: List[str]
      :raises ValueError: If unable to suggest column names.

Usage Example
-------------

.. code-block:: python

   from pkynetics.data_import import CustomImporter

   # Detect delimiter and suggest column names
   delimiter = CustomImporter.detect_delimiter('path/to/custom_data.csv')
   suggested_columns = CustomImporter.suggest_column_names('path/to/custom_data.csv', delimiter=delimiter)

   print(f"Detected delimiter: {delimiter}")
   print(f"Suggested columns: {suggested_columns}")

   # Initialize the CustomImporter
   importer = CustomImporter(
       'path/to/custom_data.csv',
       suggested_columns,
       separator=delimiter,
       decimal='.',
       encoding='utf-8',
       skiprows=1
   )

   # Import the data
   data = importer.import_data()

   # Access the imported data
   for column in suggested_columns:
       print(f"{column}: {data[column][:5]}...")  # Print first 5 values of each column

Key Features
------------

1. Flexible data import for non-standard formats
2. Automatic delimiter detection
3. Column name suggestion
4. Customizable import parameters (separator, decimal format, encoding, etc.)
5. Robust error handling

Notes
-----

- The CustomImporter is particularly useful when dealing with data formats not covered by the standard TGA and DSC importers.
- It's recommended to use the `detect_delimiter` and `suggest_column_names` methods before initializing the CustomImporter to ensure correct data parsing.
- Make sure to specify the correct decimal separator and encoding to avoid data misinterpretation.

See Also
--------

- :func:`tga_importer`: For importing standard Thermogravimetric Analysis (TGA) data
- :func:`dsc_importer`: For importing standard Differential Scanning Calorimetry (DSC) data
