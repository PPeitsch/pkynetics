import pandas as pd
import numpy as np
from typing import Dict, Union
import logging
import chardet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dsc_importer(file_path: str, manufacturer: str = "auto") -> Dict[str, np.ndarray]:
    """
    Import DSC data from common file formats.

    Args:
        file_path (str): Path to the DSC data file.
        manufacturer (str): Instrument manufacturer. Options: "auto", "TA", "Mettler", "Netzsch", "Setaram".
            Default is "auto" for automatic detection.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing temperature, time, heat_flow, and heat_capacity data.

    Raises:
        ValueError: If the file format is not recognized or supported.
        FileNotFoundError: If the specified file does not exist.
    """
    logger.info(f"Importing DSC data from {file_path}")

    try:
        if manufacturer == "auto":
            manufacturer = _detect_manufacturer(file_path)
            logger.info(f"Detected manufacturer: {manufacturer}")
        elif manufacturer == "TA":
            data = _import_ta_instruments(file_path)
        elif manufacturer == "Mettler":
            data = _import_mettler_toledo(file_path)
        elif manufacturer == "Netzsch":
            data = _import_netzsch(file_path)
        elif manufacturer == "Setaram":
            data = import_setaram(file_path)
        else:
            raise ValueError(f"Unsupported manufacturer: {manufacturer}")

        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error importing DSC data: {str(e)}")
        raise


def import_setaram(file_path: str) -> Dict[str, Union[np.ndarray, None]]:
    """
    Import Setaram DSC or simultaneous DSC-TGA data.

    Args:
        file_path (str): Path to the Setaram data file.

    Returns:
        Dict[str, Union[np.ndarray, None]]: Dictionary containing time, temperature,
        sample_temperature, heat_flow, and weight (if available) data.

    Raises:
        ValueError: If the file format is not recognized as a valid Setaram format.
        FileNotFoundError: If the specified file does not exist.
    """
    logger.info(f"Importing Setaram data from {file_path}")

    try:
        # Detect file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        logger.info(f"Detected file encoding: {encoding}")

        # Read the file with detected encoding
        df = pd.read_csv(file_path, delim_whitespace=True, decimal='.', encoding=encoding,
                         dtype=str, skiprows=12)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Rename columns to match expected format
        column_mapping = {
            'Index': 'index',
            'Time': 'time',
            'Furnace': 'temperature',
            'Sample': 'sample_temperature',
            'TG ': 'weight',
            'HeatFlow': 'heat_flow'
        }
        df = df.rename(columns=column_mapping)

        # Convert string values to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.').str.strip(), errors='coerce')

        result = {
            'time': df['time'].values,
            'temperature': df['temperature'].values,
            'sample_temperature': df['sample_temperature'].values,
        }

        if 'heat_flow' in df.columns:
            result['heat_flow'] = df['heat_flow'].values
        else:
            result['heat_flow'] = None

        if 'weight' in df.columns:
            result['weight'] = df['weight'].values
        else:
            result['weight'] = None

        return result

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Setaram file: {str(e)}")
        raise ValueError(f"Unable to read Setaram file. Error: {str(e)}")


def _detect_manufacturer(file_path: str) -> str:
    """
    Detect the instrument manufacturer based on file content.

    Args:
        file_path (str): Path to the data file.

    Returns:
        str: Detected manufacturer name.

    Raises:
        ValueError: If unable to detect the manufacturer automatically.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        # Detect file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        logger.info(f"Detected file encoding: {encoding}")

        with open(file_path, 'r', encoding=encoding) as f:
            header = f.read(1000)  # Read first 1000 characters

        if "TA Instruments" in header:
            return "TA"
        elif "METTLER TOLEDO" in header:
            return "Mettler"
        elif "NETZSCH" in header:
            return "Netzsch"
        elif "Setaram" in header or ("Time (s)" in header and "Furnace Temperature (°C)" in header):
            return "Setaram"
        else:
            raise ValueError("Unable to detect manufacturer automatically. Please specify manually.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error detecting manufacturer: {str(e)}")
        raise ValueError(f"Unable to detect manufacturer. Error: {str(e)}")


def _import_ta_instruments(file_path: str) -> Dict[str, np.ndarray]:
    """
    Import DSC data from TA Instruments format.

    Args:
        file_path (str): Path to the TA Instruments data file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing temperature, time, heat_flow, and heat_capacity data.

    Raises:
        ValueError: If the file format is not recognized as a valid TA Instruments format.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path, skiprows=1, encoding='iso-8859-1')
        return {
            "temperature": df["Temperature (°C)"].values,
            "time": df["Time (min)"].values,
            "heat_flow": df["Heat Flow (mW)"].values,
            "heat_capacity": df["Heat Capacity (J/(g·°C))"].values if "Heat Capacity (J/(g·°C))" in df.columns else None
        }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading TA Instruments file: {str(e)}")
        raise ValueError(f"Unable to read TA Instruments file. Error: {str(e)}")


def _import_mettler_toledo(file_path: str) -> Dict[str, np.ndarray]:
    """
    Import DSC data from Mettler Toledo format.

    Args:
        file_path (str): Path to the Mettler Toledo data file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing temperature, time, heat_flow, and heat_capacity data.

    Raises:
        ValueError: If the file format is not recognized as a valid Mettler Toledo format.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path, skiprows=2, delimiter='\t', encoding='iso-8859-1')
        return {
            "temperature": df["Temperature [°C]"].values,
            "time": df["Time [min]"].values,
            "heat_flow": df["Heat Flow [mW]"].values,
            "heat_capacity": df[
                "Specific Heat Capacity [J/(g·K)]"].values if "Specific Heat Capacity [J/(g·K)]" in df.columns else None
        }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Mettler Toledo file: {str(e)}")
        raise ValueError(f"Unable to read Mettler Toledo file. Error: {str(e)}")


def _import_netzsch(file_path: str) -> Dict[str, np.ndarray]:
    """
    Import DSC data from Netzsch format.

    Args:
        file_path (str): Path to the Netzsch data file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing temperature, time, heat_flow, and heat_capacity data.

    Raises:
        ValueError: If the file format is not recognized as a valid Netzsch format.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path, skiprows=10, delimiter='\t', encoding='iso-8859-1')
        return {
            "temperature": df["Temperature/°C"].values,
            "time": df["Time/min"].values,
            "heat_flow": df["DSC/(mW/mg)"].values,
            "heat_capacity": df[
                "Specific Heat Capacity/(J/(g·K))"].values if "Specific Heat Capacity/(J/(g·K))" in df.columns else None
        }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Netzsch file: {str(e)}")
        raise ValueError(f"Unable to read Netzsch file. Error: {str(e)}")
