import logging
from typing import Dict, Optional, TypedDict

import chardet
import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DetectionResult(TypedDict):
    encoding: str
    confidence: float
    language: Optional[str]


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
        # Detect file encoding
        with open(file_path, "rb") as file:
            raw_data = file.read()
            detection_result = chardet.detect(raw_data)  # Use chardet's type
            encoding = detection_result["encoding"]

        logger.info(f"Detected file encoding: {encoding}")

        # Read the file with detected encoding
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            encoding=encoding,
            engine="python",
            skiprows=lambda x: x < 2 or (2 < x < 5),
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

        # Convert values to float, handling both comma and dot as decimal separators
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.replace(",", ".").astype(float)
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
