"""Shared pytest hooks for the fastsim repositories (core, crv, tracker, umbrella).

Each repository's ``tests/conftest.py`` imports these hooks, so one test process stays bounded in memory:

- After every test module, JAX's compilation caches are cleared and garbage is collected. A suite otherwise keeps
  every compiled XLA program and its buffers for the rest of the process, which is how a full run grows to several GB.
- ``@pytest.mark.slow`` marks statistical validations with large samples (GB-scale or tens of seconds). They are
  skipped unless ``--runslow`` is given (or selected explicitly with ``-m slow``).
- ``XLA_PYTHON_CLIENT_PREALLOCATE`` defaults to false, so a test run on a GPU host does not reserve most of the card.

Selecting the tests a change affects is ``python -m mu2efastsim.testselect`` (see that module).
"""
import gc
import os
import sys

import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run @pytest.mark.slow statistical validations (large samples, GB-scale memory)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: large-sample statistical validation; skipped unless --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow validation: run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="module")
def _release_jax_caches():
    """Drop compiled programs and their buffers when a test module finishes."""
    yield
    jax = sys.modules.get("jax")
    if jax is not None:
        jax.clear_caches()
    gc.collect()
