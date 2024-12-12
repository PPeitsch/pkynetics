"""Custom data importer for Pkynetics."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray

logger = logging.getLogger(__name__)

# Type aliases
DataArray = Union[np.ndarray, ExtensionArray]
ReturnDict = Dict[str, Optional[DataArray]]


class CustomImporter:
    """
    A flexible importer for custom data formats.
    """

    def __init__(
        self,
        file_path: str,
        column_names: List[str],
        separator: str = ",",
        decimal: str = ".",
        encoding: str = "utf-8",
        skiprows: int = 0,
    ):
        """
        Initialize the CustomImporter.

        Args:
            file_path (str): Path to the data file.
            column_names (List[str]): List of column names in the order they appear in the file.
            separator (str, optional): Column separator in the file. Defaults to ','.
            decimal (str, optional): Decimal separator used in the file. Defaults to '.'.
            encoding (str, optional): File encoding. Defaults to 'utf-8'.
            skiprows (int, optional): Number of rows to skip at the beginning of the file. Defaults to 0.
        """
        self.file_path = file_path
        self.column_names = column_names
        self.separator = separator
        self.decimal = decimal
        self.encoding = encoding
        self.skiprows = skiprows

    def import_data(self) -> ReturnDict:
        """
        Import data from the file.

        Returns:
            Dict[str, Optional[np.ndarray]]: Dictionary containing the imported data.

        Raises:
            ValueError: If the file format is not recognized or supported.
            FileNotFoundError: If the specified file does not exist.
        """
        logger.info(f"Importing data from {self.file_path}")

        try:
            df = pd.read_csv(
                self.file_path,
                sep=self.separator,
                decimal=self.decimal,
                encoding=self.encoding,
                skiprows=self.skiprows,
                names=self.column_names,
            )

            # Initialize an empty dictionary with all possible fields as None
            data: ReturnDict = {col: None for col in self.column_names}

            # Fill in the values that exist
            for col in df.columns:
                data[col] = df[col].values

            return data

        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error importing data: {str(e)}")
            raise ValueError(f"Unable to import data. Error: {str(e)}")

    @staticmethod
    def detect_delimiter(file_path: str, num_lines: int = 5) -> str:
        """
        Attempt to detect the delimiter used in the file.

        Args:
            file_path (str): Path to the data file.
            num_lines (int, optional): Number of lines to check. Defaults to 5.

        Returns:
            str: Detected delimiter.

        Raises:
            ValueError: If unable to detect the delimiter.
        """
        common_delimiters = [",", ";", "\t", "|"]

        try:
            with open(file_path, "r") as file:
                lines = file.readlines()[:num_lines]

            for delimiter in common_delimiters:
                if all(delimiter in line for line in lines):
                    return delimiter

            raise ValueError("Unable to detect delimiter")
        except Exception as e:
            logger.error(f"Error detecting delimiter: {str(e)}")
            raise

    @staticmethod
    def suggest_column_names(
        file_path: str, delimiter: Optional[str] = None
    ) -> List[str]:
        """
        Suggest column names based on the first row of the file.

        Args:
            file_path (str): Path to the data file.
            delimiter (Optional[str], optional): Delimiter to use. If None, will attempt to detect. Defaults to None.

        Returns:
            List[str]: Suggested column names.

        Raises:
            ValueError: If unable to suggest column names.
        """
        try:
            if delimiter is None:
                delimiter = CustomImporter.detect_delimiter(file_path)

            with open(file_path, "r") as file:
                first_line = file.readline().strip()

            return first_line.split(delimiter)
        except Exception as e:
            logger.error(f"Error suggesting column names: {str(e)}")
            raise ValueError(f"Unable to suggest column names. Error: {str(e)}")
