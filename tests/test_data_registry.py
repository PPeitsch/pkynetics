"""Tests for the example-data registry.

None of these download anything: they check the registry itself and the two ways
of staying off the network (an unknown name, and PKYNETICS_DATA_DIR).
"""

import hashlib
import os

import pytest

from pkynetics import data as data_module
from pkynetics.data import DATA_VERSION, available, fetch


def test_every_entry_declares_a_sha256():
    registry = data_module._REGISTRY.registry
    assert registry, "the registry is empty"
    for name, checksum in registry.items():
        assert checksum.startswith("sha256:"), name
        assert len(checksum) == len("sha256:") + 64, name


def test_available_lists_the_registry():
    assert available() == sorted(data_module._REGISTRY.registry)
    assert "dilatometry_zry4_heating.asc" in available()


def test_the_data_version_is_a_tag():
    assert DATA_VERSION.startswith("v")


def test_an_unknown_name_is_rejected_without_touching_the_network():
    with pytest.raises(ValueError, match="is not an example file"):
        fetch("no_such_file.asc")


def test_a_local_directory_is_used_instead_of_downloading(tmp_path, monkeypatch):
    """PKYNETICS_DATA_DIR points pooch at files already on disk."""
    name = "tga_setaram_duran.csv"
    contents = b"Time (s);TG (mg)\n0;1,0\n"
    checksum = hashlib.sha256(contents).hexdigest()

    # pooch appends the data version to the directory it is given
    local = tmp_path / "data" / DATA_VERSION
    local.mkdir(parents=True)
    (local / name).write_bytes(contents)

    monkeypatch.setenv("PKYNETICS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setitem(data_module._REGISTRY.registry, name, f"sha256:{checksum}")
    monkeypatch.setattr(data_module._REGISTRY, "path", local)

    path = fetch(name)
    assert os.path.samefile(path, local / name)


def test_a_corrupt_cached_file_is_not_returned(tmp_path, monkeypatch):
    """The hash is what makes the cache safe: a cached file that does not match it
    is re-downloaded rather than handed back. The download is stubbed out, so what
    this asserts is that fetch reaches for it instead of trusting the bad file."""
    name = "tga_setaram_duran.csv"
    local = tmp_path / "data" / DATA_VERSION
    local.mkdir(parents=True)
    (local / name).write_bytes(b"not the real file")

    def _no_downloads(*args, **kwargs):
        raise RuntimeError("download attempted")

    monkeypatch.setenv("PKYNETICS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(data_module._REGISTRY, "path", local)
    monkeypatch.setattr(data_module.pooch.core, "stream_download", _no_downloads)

    with pytest.raises(RuntimeError, match="download attempted"):
        fetch(name)
