import pandas as pd
import numpy as np
from typing import Dict
import logging
import chardet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dilatometry_importer(file_path: str) -> Dict[str, np.ndarray]:
    """
    Import dilatometry data from the specified file format.

    Args:
        file_path (str): Path to the dilatometry data file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing time, temperature, relative_change, and differential_change data.

    Raises:
        ValueError: If the file format is not recognized or supported.
        FileNotFoundError: If the specified file does not exist.
    """
    logger.info(f"Importing dilatometry data from {file_path}")

    try:
        # Detect file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        logger.info(f"Detected file encoding: {encoding}")

        # Read the file with detected encoding
        df = pd.read_csv(file_path, sep=r'\s+', encoding=encoding, engine='python',
                         skiprows=lambda x: x < 2 or (2 < x < 5), index_col=0)

        # Clean column names and rename
        df.columns = df.columns.str.strip()
        column_mapping = {
            df.columns[0]: 'time',
            df.columns[1]: 'temperature',
            df.columns[2]: 'relative_change',
            df.columns[3]: 'differential_change'
        }
        df = df.rename(columns=column_mapping)

        # Convert values to float, handling both comma and dot as decimal separators
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
            else:
                df[col] = df[col].astype(float)

        # Create result dictionary
        result = {
            'time': df['time'].values,
            'temperature': df['temperature'].values,
            'relative_change': df['relative_change'].values,
            'differential_change': df['differential_change'].values
        }

        return result

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error importing dilatometry data: {str(e)}")
        raise ValueError(f"Unable to import dilatometry data. Error: {str(e)}")
