"""Import functions for DSC data."""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray

from ._encoding import detect_encoding

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
        Values are returned in the units of the instrument export: temperature in
        °C; time in s (Setaram) or min (TA, Mettler, Netzsch); heat flow in mW
        (mW/mg for Netzsch).

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
        encoding = detect_encoding(file_path)

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
                sep=r"\s+",
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
        encoding = detect_encoding(file_path)
        logger.info(f"Detected file encoding: {encoding}")

        with open(file_path, "r", encoding=encoding) as f:
            header = f.read(4000)  # TA Universal Analysis headers are long

        if "TA Instruments" in header or "StartOfData" in header:
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
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            if "StartOfData" in f.read(4000):
                return _import_ta_universal_analysis(file_path, encoding)

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


def _import_ta_universal_analysis(file_path: str, encoding: str) -> ReturnDict:
    """
    Import a TA Instruments Universal Analysis text export.

    The header lists the signals ("Sig1<TAB>Time (min)", ...) and the data
    follow the "StartOfData" line, whitespace separated. Rows with negative
    time (marker rows before the run) are dropped.

    Args:
        file_path: Path to the exported file
        encoding: Text encoding of the file

    Returns:
        Dictionary with time (min), temperature (°C), heat_flow (mW) and,
        if exported, heat_capacity; missing signals are None

    Raises:
        ValueError: If the file has no StartOfData section or no time column
    """
    with open(file_path, "r", encoding=encoding) as f:
        lines = f.read().splitlines()

    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "StartOfData")
    except StopIteration:
        raise ValueError("No StartOfData section in TA Universal Analysis file")

    signals = []
    for line in lines[:start]:
        key, _, value = line.partition("\t")
        if key.startswith("Sig") and key[3:].isdigit():
            signals.append(value.strip())

    rows = [line.split() for line in lines[start + 1 :] if line.strip()]
    values = np.array([row[: len(signals)] for row in rows], dtype=np.float64)

    prefixes = {
        "Time": "time",
        "Temperature": "temperature",
        "Heat Flow": "heat_flow",
        "Heat Capacity": "heat_capacity",
    }
    data: ReturnDict = {
        "time": None,
        "temperature": None,
        "heat_flow": None,
        "heat_capacity": None,
        "sample_temperature": None,
        "weight": None,
    }
    for column, name in enumerate(signals):
        for prefix, key in prefixes.items():
            if name.startswith(prefix) and data[key] is None:
                data[key] = values[:, column]

    time = data["time"]
    if time is None:
        raise ValueError("No time signal in TA Universal Analysis file")
    keep = np.asarray(time) >= 0
    for key, array in data.items():
        if array is not None:
            data[key] = np.asarray(array)[keep]
    return data


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
