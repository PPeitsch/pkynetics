"""Text encoding detection for instrument export files."""

import codecs
from typing import Optional

import chardet


def detect_encoding(file_path: str, sample_size: int = 65536) -> str:
    """
    Detect the text encoding of a data file.

    Checks, in order: byte order marks, BOM-less UTF-16 (every other byte
    zero in ASCII-range text; chardet does not detect it reliably), UTF-8,
    then a confident chardet guess, falling back to Latin-1 (which decodes
    any byte sequence and covers the degree sign of Western exports).

    Args:
        file_path: Path to the file
        sample_size: Number of bytes to inspect

    Returns:
        Encoding name usable with open() and pandas

    Raises:
        FileNotFoundError: If the file does not exist
    """
    with open(file_path, "rb") as file:
        raw = file.read(sample_size)

    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"

    pairs = raw[: len(raw) - len(raw) % 2]
    if len(pairs) >= 4:
        even_zeros = pairs[0::2].count(0) / (len(pairs) / 2)
        odd_zeros = pairs[1::2].count(0) / (len(pairs) / 2)
        if odd_zeros > 0.4 and even_zeros < 0.05:
            return "utf-16-le"
        if even_zeros > 0.4 and odd_zeros < 0.05:
            return "utf-16-be"

    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # chardet guesses poorly on mostly-ASCII files with a few Western
    # characters (e.g. a degree sign): trust it only when confident
    result = chardet.detect(raw)
    guess: Optional[str] = result.get("encoding")
    if guess and (result.get("confidence") or 0) >= 0.9:
        return guess
    return "latin-1"
