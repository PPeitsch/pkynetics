"""Import functions for TGA data."""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray

from ._encoding import detect_encoding
from ._manufacturer import detect_manufacturer

logger = logging.getLogger(__name__)

# Type aliases
DataArray = Union[np.ndarray, ExtensionArray]
ReturnDict = Dict[str, Optional[DataArray]]


def tga_importer(file_path: str, manufacturer: str = "auto") -> ReturnDict:
    """
    Import TGA data from common file formats.

    Args:
        file_path (str): Path to the TGA data file.
        manufacturer (str): Instrument manufacturer. Options: "auto", "TA", "Mettler", "Netzsch", "Setaram".
            Default is "auto" for automatic detection.

    Returns:
        Dict[str, Optional[np.ndarray]]: Dictionary containing temperature, time, weight, and weight_percent data.

    Raises:
        ValueError: If the file format is not recognized or supported.
        FileNotFoundError: If the specified file does not exist.
    """
    logger.info(f"Importing TGA data from {file_path}")

    try:
        if manufacturer == "auto":
            manufacturer = detect_manufacturer(file_path)
            logger.info(f"Detected manufacturer: {manufacturer}")
            if manufacturer == "TA":
                data = _import_ta_instruments(file_path)
            elif manufacturer == "Mettler":
                data = _import_mettler_toledo(file_path)
            elif manufacturer == "Netzsch":
                data = _import_netzsch(file_path)
            else:  # Setaram
                return import_setaram(file_path)
        elif manufacturer == "TA":
            data = _import_ta_instruments(file_path)
        elif manufacturer == "Mettler":
            data = _import_mettler_toledo(file_path)
        elif manufacturer == "Netzsch":
            data = _import_netzsch(file_path)
        elif manufacturer == "Setaram":
            return import_setaram(file_path)
        else:
            raise ValueError(f"Unsupported manufacturer: {manufacturer}")

        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error importing TGA data: {str(e)}")
        raise


def import_setaram(file_path: str) -> ReturnDict:
    """
    Import Setaram TGA or simultaneous DSC-TGA data.

    Args:
        file_path (str): Path to the Setaram data file.

    Returns:
        Dict[str, Optional[np.ndarray]]: Dictionary containing time, temperature,
        sample_temperature, weight, and heat_flow (if available) data.

    Raises:
        ValueError: If the file format is not recognized as a valid Setaram format.
        FileNotFoundError: If the specified file does not exist.
    """
    logger.info(f"Importing Setaram data from {file_path}")

    try:
        encoding = detect_encoding(file_path)
        logger.info(f"Detected file encoding: {encoding}")

        # Read the file with detected encoding
        df = pd.read_csv(file_path, sep=";", decimal=",", encoding=encoding, dtype=str)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Rename columns to match expected format
        column_mapping = {
            "Time (s)": "time",
            "Furnace Temperature (°C)": "temperature",
            "Sample Temperature (°C)": "sample_temperature",
            "TG (mg)": "weight",
            "HeatFlow (mW)": "heat_flow",
        }
        df = df.rename(columns=column_mapping)

        # Convert string values to float
        for col in df.columns:
            df[col] = pd.to_numeric(
                df[col].str.replace(",", ".").str.strip(), errors="coerce"
            )

        # Initialize all fields as None
        data: ReturnDict = {
            "time": None,
            "temperature": None,
            "sample_temperature": None,
            "heat_flow": None,
            "weight": None,
            "weight_percent": None,
        }

        # Fill in available data
        if "time" in df.columns:
            data["time"] = df["time"].values
        if "temperature" in df.columns:
            data["temperature"] = df["temperature"].values
        if "sample_temperature" in df.columns:
            data["sample_temperature"] = df["sample_temperature"].values
        if "heat_flow" in df.columns:
            data["heat_flow"] = df["heat_flow"].values
        if "weight" in df.columns:
            data["weight"] = df["weight"].values
            # Calculate weight percent if weight is available
            if data["weight"] is not None:
                initial_weight = data["weight"][0]
                data["weight_percent"] = (data["weight"] / initial_weight) * 100

        return data

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Setaram file: {str(e)}")
        raise ValueError(f"Unable to read Setaram file. Error: {str(e)}")


def _import_ta_instruments(file_path: str) -> ReturnDict:
    """
    Import TGA data from TA Instruments format.

    Args:
        file_path (str): Path to the TA Instruments data file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing temperature, time, weight, and weight_percent data.

    Raises:
        ValueError: If the file format is not recognized as a valid TA Instruments format.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path, skiprows=1, encoding="iso-8859-1")
        return {
            "temperature": df["Temperature (°C)"].values,
            "time": df["Time (min)"].values,
            "weight": df["Weight (mg)"].values,
            "weight_percent": df["Weight (%)"].values,
            "sample_temperature": None,
            "heat_flow": None,
        }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading TA Instruments file: {str(e)}")
        raise ValueError(f"Unable to read TA Instruments file. Error: {str(e)}")


def _import_mettler_toledo(file_path: str) -> ReturnDict:
    """
    Import TGA data from Mettler Toledo format.

    Args:
        file_path (str): Path to the Mettler Toledo data file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing temperature, time, weight, and weight_percent data.

    Raises:
        ValueError: If the file format is not recognized as a valid Mettler Toledo format.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path, skiprows=2, delimiter="\t", encoding="iso-8859-1")
        return {
            "temperature": df["Temperature [°C]"].values,
            "time": df["Time [min]"].values,
            "weight": df["Weight [mg]"].values,
            "weight_percent": df["Weight [%]"].values,
            "sample_temperature": None,
            "heat_flow": None,
        }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Mettler Toledo file: {str(e)}")
        raise ValueError(f"Unable to read Mettler Toledo file. Error: {str(e)}")


def _import_netzsch(file_path: str) -> ReturnDict:
    """
    Import TGA data from Netzsch format.

    Args:
        file_path (str): Path to the Netzsch data file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing temperature, time, weight, and weight_percent data.

    Raises:
        ValueError: If the file format is not recognized as a valid Netzsch format.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path, skiprows=10, delimiter="\t", encoding="iso-8859-1")
        return {
            "temperature": df["Temperature/°C"].values,
            "time": df["Time/min"].values,
            "weight": df["Mass/mg"].values,
            "weight_percent": df["Mass/%"].values,
            "sample_temperature": None,
            "heat_flow": None,
        }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Netzsch file: {str(e)}")
        raise ValueError(f"Unable to read Netzsch file. Error: {str(e)}")
