"""Shared pytest configuration."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_global_numpy_rng():
    """Seed numpy's global RNG so tests that draw noise are reproducible."""
    np.random.seed(0)
