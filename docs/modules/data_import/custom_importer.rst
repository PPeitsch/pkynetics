Custom Importer
===============

.. py:class:: CustomImporter

   A flexible importer for custom data formats.

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

Example:
--------
.. code-block:: python

   from pkynetics.data_import import CustomImporter

   # Initialize the CustomImporter
   importer = CustomImporter(
       'path/to/custom_data.csv',
       ['Time', 'Temperature', 'Weight'],
       separator=',',
       decimal='.',
       encoding='utf-8',
       skiprows=1
   )

   # Import the data
   data = importer.import_data()

   # Access the imported data
   time = data['Time']
   temperature = data['Temperature']
   weight = data['Weight']

   # Detect delimiter
   delimiter = CustomImporter.detect_delimiter('path/to/custom_data.csv')

   # Suggest column names
   column_names = CustomImporter.suggest_column_names('path/to/custom_data.csv', delimiter=',')

Note:
-----
The CustomImporter is useful when dealing with non-standard data formats or when you need more control over the import process.
