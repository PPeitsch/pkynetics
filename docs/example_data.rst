Example data
============

Pkynetics ships with no data. The example runs used in the documentation, the
examples and the tests live in a separate repository,
`PPeitsch/pkynetics-data <https://github.com/PPeitsch/pkynetics-data>`_, and are
downloaded the first time they are used, verified by SHA256 and cached on disk.

They weigh about 11 MB against some 3000 lines of code, and every release published
to PyPI would keep its own copy of them forever. They also have their own life
cycle — runs get added, corrected and cited — which has no reason to drag a version
of the library with it.

Loading a run
-------------

The named loaders return the data already imported:

.. code-block:: python

    from pkynetics.data import load_dilatometry_heating

    data = load_dilatometry_heating()
    temperature = data["temperature"]
    strain = data["relative_change"]

:func:`~pkynetics.data.fetch` returns the path to the raw file instead, for anyone
who would rather read it themselves, and :func:`~pkynetics.data.available` lists the
names it accepts:

.. code-block:: python

    from pkynetics.data import available, fetch

    print(available())
    path = fetch("dsc_tainstruments_eicosane.txt")

Where the files are cached
--------------------------

In the platform cache directory (``~/.cache/pkynetics`` on Linux,
``~/Library/Caches/pkynetics`` on macOS, ``%LOCALAPPDATA%\\pkynetics`` on Windows),
under a subdirectory named after the data version. A file is downloaded once per
machine, and a file whose hash does not match is re-downloaded rather than used.

Working offline
---------------

Point ``PKYNETICS_DATA_DIR`` at a directory holding the files and nothing goes to the
network. The files go in a subdirectory named after
:data:`~pkynetics.data.DATA_VERSION`:

.. code-block:: text

    $PKYNETICS_DATA_DIR/
        v1/
            dilatometry_zry4_heating.asc
            dsc_setaram_duran.txt
            ...

The tests that need a run carry the ``network`` marker, so the rest of the suite runs
on a machine with no network at all::

    pytest -m "not network"

Versioning
----------

:data:`~pkynetics.data.DATA_VERSION` is the tag of the data release and is independent
of the version of the library. Releases are immutable: adding a run means a new tag and
a new registry entry, never replacing a published file — replacing one would break the
hashes recorded by every already-released version of Pkynetics.

API
---

.. automodule:: pkynetics.data
   :members:
   :undoc-members:
