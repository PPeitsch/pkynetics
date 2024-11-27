"""DSC data importer for Pkynetics."""

import pandas as pd
import numpy as np
from typing import Dict, Union
import logging

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

    Examples:
        >>> data = dsc_importer("path/to/dsc_data.csv")
        >>> print(data.keys())
        dict_keys(['temperature', 'time', 'heat_flow', 'heat_capacity'])
    """
    logger.info(f"Importing DSC data from {file_path}")

    try:
        if manufacturer == "auto":
            manufacturer = _detect_manufacturer(file_path)

        if manufacturer == "TA":
            data = _import_ta_instruments(file_path)
        elif manufacturer == "Mettler":
            data = _import_mettler_toledo(file_path)
        elif manufacturer == "Netzsch":
            data = _import_netzsch(file_path)
        elif manufacturer == "Setaram":
            setaram_data = import_setaram(file_path)
            return {
                "temperature": setaram_data['temperature'],
                "time": setaram_data['time'],
                "heat_flow": setaram_data['heat_flow'],
                "heat_capacity": None  # Setaram data doesn't include heat capacity
            }
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
        # Try different encodings
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=';', decimal=',', encoding=encoding, dtype=str)
                break  # If successful, exit the loop
            except UnicodeDecodeError:
                continue  # Try the next encoding
        
        if df is None:
            raise ValueError(f"Unable to read file with any of the attempted encodings: {encodings}")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Rename columns to match expected format
        column_mapping = {
            'Time (s)': 'time',
            'Furnace Temperature (°C)': 'temperature',
            'Sample Temperature (°C)': 'sample_temperature',
            'TG (mg)': 'weight',
            'HeatFlow (mW)': 'heat_flow'
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
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
        header = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    header = f.read(1000)  # Read first 1000 characters
                break  # If successful, exit the loop
            except UnicodeDecodeError:
                continue  # Try the next encoding
        
        if header is None:
            raise ValueError(f"Unable to read file with any of the attempted encodings: {encodings}")

        if "TA Instruments" in header:
            return "TA"
        elif "METTLER TOLEDO" in header:
            return "Mettler"
        elif "NETZSCH" in header:
            return "Netzsch"
        elif "Time (s)" in header and "Furnace Temperature (°C)" in header:
            return "Setaram"
        else:
            raise ValueError("Unable to detect manufacturer automatically. Please specify manually.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise


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
