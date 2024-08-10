"""TGA data importer for Pkynetics."""

import pandas as pd
import numpy as np
from typing import Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tga_importer(file_path: str, manufacturer: str = "auto") -> Dict[str, np.ndarray]:
    """
    Import TGA data from common file formats.

    Args:
        file_path (str): Path to the TGA data file.
        manufacturer (str): Instrument manufacturer. Options: "auto", "TA", "Mettler", "Netzsch", "Setaram".
            Default is "auto" for automatic detection.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing temperature, time, weight, and weight_percent data.

    Raises:
        ValueError: If the file format is not recognized or supported.

    Examples:
        >>> data = tga_importer("path/to/tga_data.csv")
        >>> print(data.keys())
        dict_keys(['temperature', 'time', 'weight', 'weight_percent'])
    """
    logger.info(f"Importing TGA data from {file_path}")

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

def import_setaram(file_path: str) -> Dict[str, Union[np.ndarray, None]]:
    """
    Import Setaram TGA or simultaneous DSC-TGA data.

    Args:
        file_path (str): Path to the Setaram data file.

    Returns:
        Dict[str, Union[np.ndarray, None]]: Dictionary containing time, furnace_temperature, 
        sample_temperature, weight, and heat_flow (if available) data.

    Raises:
        ValueError: If the file format is not recognized as a valid Setaram format.

    Examples:
        >>> data = import_setaram("path/to/setaram_data.txt")
        >>> print(data.keys())
        dict_keys(['time', 'furnace_temperature', 'sample_temperature', 'weight', 'heat_flow'])
    """
    logger.info(f"Importing Setaram data from {file_path}")

    try:
        # Try to read the file with all possible columns
        df = pd.read_csv(file_path, sep=r'\s+', engine='python', 
                         names=['Time', 'Furnace_Temperature', 'Sample_Temperature', 'TG', 'HeatFlow'])
        
        # Check if HeatFlow column exists (simultaneous DSC-TGA)
        if 'HeatFlow' in df.columns:
            logger.info("Detected simultaneous DSC-TGA data")
            return {
                'time': df['Time'].values,
                'furnace_temperature': df['Furnace_Temperature'].values,
                'sample_temperature': df['Sample_Temperature'].values,
                'weight': df['TG'].values,
                'heat_flow': df['HeatFlow'].values
            }
        else:
            logger.info("Detected TGA-only data")
            return {
                'time': df['Time'].values,
                'furnace_temperature': df['Furnace_Temperature'].values,
                'sample_temperature': df['Sample_Temperature'].values,
                'weight': df['TG'].values,
                'heat_flow': None
            }
    except Exception as e:
        logger.error(f"Error reading Setaram file: {e}")
        raise ValueError(f"Unable to read Setaram file. Error: {e}")

def _detect_manufacturer(file_path: str) -> str:
    """Detect the instrument manufacturer based on file content."""
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

def _import_ta_instruments(file_path: str) -> Dict[str, np.ndarray]:
    """Import TGA data from TA Instruments format."""
    df = pd.read_csv(file_path, skiprows=1)
    return {
        "temperature": df["Temperature (°C)"].values,
        "time": df["Time (min)"].values,
        "weight": df["Weight (mg)"].values,
        "weight_percent": df["Weight (%)"].values
    }

def _import_mettler_toledo(file_path: str) -> Dict[str, np.ndarray]:
    """Import TGA data from Mettler Toledo format."""
    df = pd.read_csv(file_path, skiprows=2, delimiter='\t')
    return {
        "temperature": df["Temperature [°C]"].values,
        "time": df["Time [min]"].values,
        "weight": df["Weight [mg]"].values,
        "weight_percent": df["Weight [%]"].values
    }

def _import_netzsch(file_path: str) -> Dict[str, np.ndarray]:
    """Import TGA data from Netzsch format."""
    df = pd.read_csv(file_path, skiprows=10, delimiter='\t')
    return {
        "temperature": df["Temperature/°C"].values,
        "time": df["Time/min"].values,
        "weight": df["Mass/mg"].values,
        "weight_percent": df["Mass/%"].values
    }
