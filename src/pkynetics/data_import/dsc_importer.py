"""Import functions for DSC data."""

import logging
from typing import Dict, Optional, Union

import chardet
import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray

logger = logging.getLogger(__name__)

# Type aliases
DataArray = Union[np.ndarray, ExtensionArray]
ReturnDict = Dict[str, Optional[DataArray]]


def dsc_importer(file_path: str, manufacturer: str = "auto") -> ReturnDict:
    """
    Import DSC data from common file formats.

    Args:
        file_path (str): Path to the DSC data file.
        manufacturer (str): Instrument manufacturer. Options: "auto", "TA", "Mettler", "Netzsch", "Setaram".
            Default is "auto" for automatic detection.

    Returns:
        Dict[str, Optional[np.ndarray]]: Dictionary containing temperature, time, heat_flow, and heat_capacity data.

    Raises:
        ValueError: If the file format is not recognized or supported.
        FileNotFoundError: If the specified file does not exist.
    """
    logger.info(f"Importing DSC data from {file_path}")

    try:
        if manufacturer == "auto":
            manufacturer = _detect_manufacturer(file_path)
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
        logger.error(f"Error importing DSC data: {str(e)}")
        raise


def import_setaram(file_path: str) -> ReturnDict:
    """
    Import Setaram DSC or simultaneous DSC-TGA data.
    Handles both old and new Setaram file formats.

    Args:
        file_path (str): Path to the Setaram data file.

    Returns:
        Dict[str, Optional[np.ndarray]]: Dictionary containing time, temperature,
        sample_temperature, heat_flow, and weight (if available) data.
    """
    logger.info(f"Importing Setaram data from {file_path}")

    try:
        # Detect file encoding
        with open(file_path, "rb") as file:
            raw_data = file.read()
            detection_result = chardet.detect(raw_data)
            encoding = detection_result["encoding"]

        # Try to read file in new format first
        try:
            df = pd.read_csv(
                file_path,
                sep=";",
                decimal=",",
                encoding=encoding,
                dtype=str,
                skiprows=13 if file_path.lower().endswith(".txt") else 0,
            )
            # Verify if it's really the new format by checking column names
            if "Time (s)" in df.columns:
                logger.info("Detected new Setaram format")
                column_mapping = {
                    "Time (s)": "time",
                    "Furnace Temperature (°C)": "temperature",
                    "Sample Temperature (°C)": "sample_temperature",
                    "TG (mg)": "weight",
                    "HeatFlow (mW)": "heat_flow",
                }
            else:
                raise ValueError("Not new format")

        except (pd.errors.ParserError, ValueError):
            # If new format fails, try old format
            logger.info("Trying old Setaram format")
            df = pd.read_csv(
                file_path,
                delim_whitespace=True,
                decimal=".",
                encoding=encoding,
                dtype=str,
                skiprows=12,
            )
            column_mapping = {
                "Index": "index",
                "Time": "time",
                "Furnace": "temperature",
                "Sample": "sample_temperature",
                "TG": "weight",
                "HeatFlow": "heat_flow",
            }

        # Clean column names and rename
        df.columns = df.columns.str.strip()
        df = df.rename(columns=column_mapping)

        # Convert string values to float, handling both decimal separators
        for col in df.columns:
            if col in column_mapping.values():
                df[col] = pd.to_numeric(
                    df[col].str.replace(",", ".").str.strip(), errors="coerce"
                )

        # Initialize data dictionary
        data: ReturnDict = {
            "time": None,
            "temperature": None,
            "sample_temperature": None,
            "heat_flow": None,
            "weight": None,
        }

        # Fill available data
        for key in data.keys():
            if key in df.columns:
                data[key] = df[key].values

        return data

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
        with open(file_path, "rb") as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"]

        logger.info(f"Detected file encoding: {encoding}")

        with open(file_path, "r", encoding=encoding) as f:
            header = f.read(1000)  # Read first 1000 characters

        if "TA Instruments" in header:
            return "TA"
        elif "METTLER TOLEDO" in header:
            return "Mettler"
        elif "NETZSCH" in header:
            return "Netzsch"
        elif "Setaram" in header or (
            "Time (s)" in header and "Furnace Temperature (°C)" in header
        ):
            return "Setaram"
        else:
            raise ValueError(
                "Unable to detect manufacturer automatically. Please specify manually."
            )
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error detecting manufacturer: {str(e)}")
        raise ValueError(f"Unable to detect manufacturer. Error: {str(e)}")


def _import_ta_instruments(file_path: str) -> ReturnDict:
    """
    Import DSC data from TA Instruments format.

    Args:
        file_path (str): Path to the TA Instruments data file.

    Returns:
        Dict[str, Optional[np.ndarray]]: Dictionary containing temperature, time, heat_flow, and heat_capacity data.

    Raises:
        ValueError: If the file format is not recognized as a valid TA Instruments format.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path, skiprows=1, encoding="iso-8859-1")
        data: ReturnDict = {
            "time": df["Time (min)"].values,
            "temperature": df["Temperature (°C)"].values,
            "heat_flow": df["Heat Flow (mW)"].values,
            "heat_capacity": None,
            "sample_temperature": None,
            "weight": None,
        }
        if "Heat Capacity (J/(g·°C))" in df.columns:
            data["heat_capacity"] = df["Heat Capacity (J/(g·°C))"].values
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading TA Instruments file: {str(e)}")
        raise ValueError(f"Unable to read TA Instruments file. Error: {str(e)}")


def _import_mettler_toledo(file_path: str) -> ReturnDict:
    """
    Import DSC data from Mettler Toledo format.

    Args:
        file_path (str): Path to the Mettler Toledo data file.

    Returns:
        Dict[str, Optional[np.ndarray]]: Dictionary containing temperature, time, heat_flow, and heat_capacity data.

    Raises:
        ValueError: If the file format is not recognized as a valid Mettler Toledo format.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path, skiprows=2, delimiter="\t", encoding="iso-8859-1")
        data: ReturnDict = {
            "temperature": df["Temperature [°C]"].values,
            "time": df["Time [min]"].values,
            "heat_flow": df["Heat Flow [mW]"].values,
            "heat_capacity": None,
            "sample_temperature": None,
            "weight": None,
        }
        if "Specific Heat Capacity [J/(g·K)]" in df.columns:
            data["heat_capacity"] = df["Specific Heat Capacity [J/(g·K)]"].values
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Mettler Toledo file: {str(e)}")
        raise ValueError(f"Unable to read Mettler Toledo file. Error: {str(e)}")


def _import_netzsch(file_path: str) -> ReturnDict:
    """
    Import DSC data from Netzsch format.

    Args:
        file_path (str): Path to the Netzsch data file.

    Returns:
        Dict[str, Optional[np.ndarray]]: Dictionary containing temperature, time, heat_flow, and heat_capacity data.

    Raises:
        ValueError: If the file format is not recognized as a valid Netzsch format.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path, skiprows=10, delimiter="\t", encoding="iso-8859-1")
        data: ReturnDict = {
            "temperature": df["Temperature/°C"].values,
            "time": df["Time/min"].values,
            "heat_flow": df["DSC/(mW/mg)"].values,
            "heat_capacity": None,
            "sample_temperature": None,
            "weight": None,
        }
        if "Specific Heat Capacity/(J/(g·K))" in df.columns:
            data["heat_capacity"] = df["Specific Heat Capacity/(J/(g·K))"].values
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Netzsch file: {str(e)}")
        raise ValueError(f"Unable to read Netzsch file. Error: {str(e)}")
