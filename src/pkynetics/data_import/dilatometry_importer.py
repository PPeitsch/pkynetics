import logging
from typing import Dict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._encoding import detect_encoding

logger = logging.getLogger(__name__)


def dilatometry_importer(file_path: str) -> Dict[str, NDArray[np.float64]]:
    """
    Import dilatometry data from the specified file format.

    Args:
        file_path (str): Path to the dilatometry data file.

    Returns:
        Dict[str, NDArray[np.float64]]: Dictionary containing time, temperature,
        relative_change, and differential_change data.

    Raises:
        ValueError: If the file format is not recognized or supported.
        FileNotFoundError: If the specified file does not exist.
    """
    logger.info(f"Importing dilatometry data from {file_path}")

    try:
        encoding = detect_encoding(file_path)

        logger.info(f"Detected file encoding: {encoding}")

        # Read the file with detected encoding
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            encoding=encoding,
            engine="python",
            # Line 2 holds the column names and line 3 the units; data starts at 4
            skiprows=lambda x: x < 2 or x == 3,
            index_col=0,
        )

        # Clean column names and rename
        df.columns = df.columns.str.strip()
        column_mapping = {
            df.columns[0]: "time",
            df.columns[1]: "temperature",
            df.columns[2]: "relative_change",
            df.columns[3]: "differential_change",
        }
        df = df.rename(columns=column_mapping)

        # Convert values to float, handling both comma and dot as decimal separators.
        # Text columns are "object" in pandas 2 and "str" in pandas 3: test for
        # numeric instead of matching the dtype name.
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
            else:
                df[col] = df[col].astype(float)

        # Create result dictionary
        result_data: Dict[str, NDArray[np.float64]] = {
            "time": np.array(df["time"].values, dtype=np.float64),
            "temperature": np.array(df["temperature"].values, dtype=np.float64),
            "relative_change": np.array(df["relative_change"].values, dtype=np.float64),
            "differential_change": np.array(
                df["differential_change"].values, dtype=np.float64
            ),
        }

        return result_data

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error importing dilatometry data: {str(e)}")
        raise ValueError(f"Unable to import dilatometry data. Error: {str(e)}")
