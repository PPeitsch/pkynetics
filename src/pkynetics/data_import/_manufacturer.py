"""Instrument manufacturer detection, shared by the DSC and TGA importers."""

import logging

from ._encoding import detect_encoding

logger = logging.getLogger(__name__)

# TA Universal Analysis headers are long, so read enough of the file to reach
# the column names of every supported export
HEADER_SIZE = 4000


def detect_manufacturer(file_path: str) -> str:
    """
    Detect the instrument manufacturer from the file header.

    Args:
        file_path: Path to the data file.

    Returns:
        One of "TA", "Mettler", "Netzsch", "Setaram".

    Raises:
        ValueError: If the manufacturer cannot be detected.
        FileNotFoundError: If the file does not exist.
    """
    try:
        encoding = detect_encoding(file_path)
        logger.info(f"Detected file encoding: {encoding}")

        with open(file_path, "r", encoding=encoding) as f:
            header = f.read(HEADER_SIZE)

        if "TA Instruments" in header or "StartOfData" in header:
            return "TA"
        elif "METTLER TOLEDO" in header:
            return "Mettler"
        elif "NETZSCH" in header:
            return "Netzsch"
        elif "Setaram" in header or _has_setaram_columns(header):
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


def _has_setaram_columns(header: str) -> bool:
    """
    Recognize a Setaram export by its column names.

    The exports do not name the manufacturer anywhere. Their column line
    starts with "Index" or "Time (s)" and lists the furnace temperature
    next to the TG or heat flow signal, with either ";" or whitespace as
    the separator ("Index Time Furnace Sample TG HeatFlow").
    """
    for line in header.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("Index") or stripped.startswith("Time (s)")):
            continue
        if "Furnace" in stripped and ("HeatFlow" in stripped or "TG" in stripped):
            return True
    return False
