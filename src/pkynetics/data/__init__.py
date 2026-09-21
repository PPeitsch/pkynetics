"""Example thermal-analysis data, fetched on demand.

The example runs are not shipped with the package: they weigh about 11 MB against
~3000 lines of code, and every release published to PyPI would keep its own copy of
them forever. They live in `PPeitsch/pkynetics-data <https://github.com/PPeitsch/pkynetics-data>`_
and are downloaded on first use, verified by SHA256 and cached on disk, so a given
file is fetched once per machine.

Most users want one of the named loaders, which return the data already imported::

    from pkynetics.data import load_dilatometry_heating

    data = load_dilatometry_heating()

:func:`fetch` returns the path to the raw file instead, for anyone who wants to read
it themselves.

To work offline, point the ``PKYNETICS_DATA_DIR`` environment variable at a directory
holding the files and the registry reads from there instead of downloading. Note that
the files go in a subdirectory named after :data:`DATA_VERSION`, so with ``v1`` the
layout is ``$PKYNETICS_DATA_DIR/v1/dilatometry_zry4_heating.asc``.
"""

from typing import Dict, List

import numpy as np
import pooch
from numpy.typing import NDArray

from ..data_import import dilatometry_importer, dsc_importer, tga_importer
from ..data_import.dsc_importer import ReturnDict

__all__ = [
    "DATA_VERSION",
    "fetch",
    "available",
    "load_dilatometry_heating",
    "load_dilatometry_cooling",
    "load_dsc_setaram",
    "load_dsc_eicosane",
    "load_tga_setaram",
    "load_cp_three_step",
]

#: Version of the data set, independent of the version of the library. It is the tag of
#: the release the files are fetched from, and the name of the subdirectory they are
#: cached in, so an old cache is never confused with a new data set.
DATA_VERSION = "v1"

_REGISTRY = pooch.create(
    path=pooch.os_cache("pkynetics"),
    base_url="https://github.com/PPeitsch/pkynetics-data/releases/download/{version}/",
    version=DATA_VERSION,
    registry={
        "dilatometry_zry4_heating.asc": "sha256:d436f146742b8fae880aa8ced0aa7cddd5827f945be5f0e0b1343043e48e689b",
        "dilatometry_zry4_cooling.asc": "sha256:75c0e2ef8c2be93a9526d2d7a138f731f5a5060eb970fb0041308e12a8463394",
        "dsc_setaram_duran.txt": "sha256:e977a59944e8f7835c35fc6ee9831a397bbf543a8a79b0b1f6d828fe4e37693b",
        "dsc_setaram_duran.csv": "sha256:8befcf8b53fd982a5dd7d570671936baa55a9f35dfd96666539f165a5c262f26",
        "dsc_tainstruments_eicosane.txt": "sha256:6b72c4912a6e770c9a812f1d28c61f13c3671487c72e22f86d6d1340703b9f32",
        "cp_setaram_sapphire.txt": "sha256:0332b8a5207ed13a89967df3fb8c23cc11ba78061cf265536dbd039e990afa59",
        "cp_setaram_zero.txt": "sha256:c069c5198721cc286e1fd1ecfe034f76ee770d541cac38ec06022f75cf95c485",
        "cp_setaram_sample.txt": "sha256:969d5bb76bc7ced0ae1d63d59044c315a5ad589808ef11e0f5763b3b1293f161",
        "tga_setaram_duran.csv": "sha256:574b54f7c4151b7f863c4572ea1a263a3e18b58b3610a914f9111dbea894b6eb",
    },
    env="PKYNETICS_DATA_DIR",
)


def available() -> List[str]:
    """Names of the example files, as accepted by :func:`fetch`."""
    return sorted(_REGISTRY.registry)


def fetch(name: str) -> str:
    """Return the local path to an example file, downloading it on first use.

    Args:
        name: File name, as listed by :func:`available`.

    Returns:
        Absolute path to the cached file.

    Raises:
        ValueError: If `name` is not one of the example files. Unknown names are
            rejected before any network access, since a file with no registered
            hash could not be verified anyway.

    Examples:
        >>> from pkynetics.data import fetch
        >>> path = fetch("dilatometry_zry4_heating.asc")  # doctest: +SKIP
    """
    if name not in _REGISTRY.registry:
        raise ValueError(
            f"{name!r} is not an example file. Available: {', '.join(available())}"
        )
    return str(_REGISTRY.fetch(name))


def load_dilatometry_heating() -> Dict[str, NDArray[np.float64]]:
    """Zircaloy-4 heated to 1000 degC at 10 degC/s, trimmed to 630-1000 degC.

    Contains the alpha->beta transformation at roughly 855-935 degC.
    """
    return dilatometry_importer(fetch("dilatometry_zry4_heating.asc"))


def load_dilatometry_cooling() -> Dict[str, NDArray[np.float64]]:
    """Zircaloy-4 cooled from 1050 degC at 0.5 degC/s.

    The beta->alpha transformation runs from about 945 to 760 degC. Restrict the
    temperature range (1040-700 degC works) before analysing it: over the full curve
    the default ``margin=0.2`` puts the baseline window inside the transformation.
    """
    return dilatometry_importer(fetch("dilatometry_zry4_cooling.asc"))


def load_dsc_setaram(fmt: str = "txt") -> ReturnDict:
    """Duran glass on a Setaram instrument, simultaneous TG and heat flow.

    Args:
        fmt: ``"txt"`` for the UTF-16LE instrument export, ``"csv"`` for the same run
            exported as CSV.
    """
    if fmt not in ("txt", "csv"):
        raise ValueError(f"fmt must be 'txt' or 'csv', not {fmt!r}")
    return dsc_importer(fetch(f"dsc_setaram_duran.{fmt}"), manufacturer="Setaram")


def load_dsc_eicosane() -> ReturnDict:
    """Eicosane melting, TA Instruments 2920 MDSC, 9.00 mg under nitrogen."""
    return dsc_importer(fetch("dsc_tainstruments_eicosane.txt"))


def load_tga_setaram() -> ReturnDict:
    """The Duran glass run again, as a TGA export without the heat-flow column."""
    return tga_importer(fetch("tga_setaram_duran.csv"), manufacturer="Setaram")


def load_cp_three_step() -> Dict[str, ReturnDict]:
    """The three runs of a heat-capacity determination by the three-step method.

    Returns:
        A dict with keys ``"sapphire"``, ``"zero"`` and ``"sample"``, each holding the
        imported run.
    """
    return {
        step: dsc_importer(fetch(f"cp_setaram_{step}.txt"), manufacturer="Setaram")
        for step in ("sapphire", "zero", "sample")
    }
