"""Shared pytest configuration."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_global_numpy_rng():
    """Seed numpy's global RNG so tests that draw noise are reproducible."""
    np.random.seed(0)


#: Fixtures that load an example run. Anything asking for one of these needs the
#: data, which is downloaded on first use, so it gets the `network` marker without
#: every test having to remember to declare it.
NETWORK_FIXTURES = frozenset({"real_curve", "real_cooling_curve"})


def pytest_collection_modifyitems(items):
    for item in items:
        if NETWORK_FIXTURES.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.network)
