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
        Dict[str, Union[np.ndarray, None]]: Dictionary containing time, furnace_temperature, 
        sample_temperature, heat_flow, and weight (if available) data.

    Raises:
        ValueError: If the file format is not recognized as a valid Setaram format.
        FileNotFoundError: If the specified file does not exist.

    Examples:
        >>> data = import_setaram("path/to/setaram_data.txt")
        >>> print(data.keys())
        dict_keys(['time', 'furnace_temperature', 'sample_temperature', 'heat_flow', 'weight'])
    """
    logger.info(f"Importing Setaram data from {file_path}")

    try:
        df = pd.read_csv(file_path, sep=r'\s+', engine='python',
                         names=['Time', 'Furnace_Temperature', 'Sample_Temperature', 'TG', 'HeatFlow'])

        if 'TG' in df.columns:
            logger.info("Detected simultaneous DSC-TGA data")
            return {
                'time': df['Time'].values,
                'furnace_temperature': df['Furnace_Temperature'].values,
                'sample_temperature': df['Sample_Temperature'].values,
                'heat_flow': df['HeatFlow'].values,
                'weight': df['TG'].values
            }
        else:
            logger.info("Detected DSC-only data")
            return {
                'time': df['Time'].values,
                'furnace_temperature': df['Furnace_Temperature'].values,
                'sample_temperature': df['Sample_Temperature'].values,
                'heat_flow': df['HeatFlow'].values,
                'weight': None
            }
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
        with open(file_path, 'r') as f:
            header = f.read(1000)  # Read first 1000 characters

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
        df = pd.read_csv(file_path, skiprows=1)
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
        df = pd.read_csv(file_path, skiprows=2, delimiter='\t')
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
        df = pd.read_csv(file_path, skiprows=10, delimiter='\t')
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
