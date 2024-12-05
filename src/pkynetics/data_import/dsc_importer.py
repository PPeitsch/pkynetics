import logging
from typing import Dict, Optional, Mapping

import chardet
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dsc_importer(file_path: str, manufacturer: str = "auto") -> Mapping[str, Optional[np.ndarray]]:
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
        data: Dict[str, Optional[np.ndarray]]
        if manufacturer == "auto":
            manufacturer = _detect_manufacturer(file_path)
            logger.info(f"Detected manufacturer: {manufacturer}")
            if manufacturer == "TA":
                data_raw = _import_ta_instruments(file_path)
            elif manufacturer == "Mettler":
                data_raw = _import_mettler_toledo(file_path)
            elif manufacturer == "Netzsch":
                data_raw = _import_netzsch(file_path)
            else:  # Setaram
                return import_setaram(file_path)
            data = {k: v for k, v in data_raw.items()}
        elif manufacturer == "TA":
            data_raw = _import_ta_instruments(file_path)
            data = {k: v for k, v in data_raw.items()}
        elif manufacturer == "Mettler":
            data_raw = _import_mettler_toledo(file_path)
            data = {k: v for k, v in data_raw.items()}
        elif manufacturer == "Netzsch":
            data_raw = _import_netzsch(file_path)
            data = {k: v for k, v in data_raw.items()}
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


def import_setaram(file_path: str) -> Mapping[str, Optional[np.ndarray]]:
    """
    Import Setaram DSC or simultaneous DSC-TGA data.

    Args:
        file_path (str): Path to the Setaram data file.

    Returns:
        Dict[str, Optional[np.ndarray]]: Dictionary containing time, temperature,
        sample_temperature, heat_flow, and weight (if available) data.

    Raises:
        ValueError: If the file format is not recognized as a valid Setaram format.
        FileNotFoundError: If the specified file does not exist.
    """
    logger.info(f"Importing Setaram data from {file_path}")

    try:
        # Detect file encoding
        with open(file_path, "rb") as file:
            raw_data = file.read()
            detection_result = chardet.detect(raw_data)
            encoding = detection_result["encoding"]

        logger.info(f"Detected file encoding: {encoding}")

        # Read the file with detected encoding
        df = pd.read_csv(
            file_path,
            delim_whitespace=True,
            decimal=".",
            encoding=encoding,
            dtype=str,
            skiprows=12,
        )

        # Clean column names
        df.columns = df.columns.str.strip()

        # Rename columns to match expected format
        column_mapping = {
            "Index": "index",
            "Time": "time",
            "Furnace": "temperature",
            "Sample": "sample_temperature",
            "TG ": "weight",
            "HeatFlow": "heat_flow",
        }
        df = df.rename(columns=column_mapping)

        # Convert string values to float
        for col in df.columns:
            df[col] = pd.to_numeric(
                df[col].str.replace(",", ".").str.strip(), errors="coerce"
            )

        data: Dict[str, Optional[np.ndarray]] = {
            "time": df["time"].values,
            "temperature": df["temperature"].values,
            "sample_temperature": df["sample_temperature"].values,
            "heat_flow": None,
            "weight": None
        }

        if "heat_flow" in df.columns:
            data["heat_flow"] = df["heat_flow"].values

        if "weight" in df.columns:
            data["weight"] = df["weight"].values

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


def _import_ta_instruments(file_path: str) -> Dict[str, Optional[np.ndarray]]:
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
        data: Dict[str, Optional[np.ndarray]] = {
            "temperature": df["Temperature (°C)"].values,
            "time": df["Time (min)"].values,
            "heat_flow": df["Heat Flow (mW)"].values,
            "heat_capacity": None
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


def _import_mettler_toledo(file_path: str) -> Dict[str, Optional[np.ndarray]]:
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
        data: Dict[str, Optional[np.ndarray]] = {
            "temperature": df["Temperature [°C]"].values,
            "time": df["Time [min]"].values,
            "heat_flow": df["Heat Flow [mW]"].values,
            "heat_capacity": None
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


def _import_netzsch(file_path: str) -> Dict[str, Optional[np.ndarray]]:
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
        data: Dict[str, Optional[np.ndarray]] = {
            "temperature": df["Temperature/°C"].values,
            "time": df["Time/min"].values,
            "heat_flow": df["DSC/(mW/mg)"].values,
            "heat_capacity": None
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
