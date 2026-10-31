"""Shared fastsim test hooks (mu2efastsim.testing): JAX caches released per module; @pytest.mark.slow opt-in."""
from mu2efastsim.testing import (_release_jax_caches, pytest_addoption,  # noqa: F401
                                 pytest_collection_modifyitems, pytest_configure)

import pytest

# Placeholder site configuration for tests of the endpoints and the kit (mu2efastsim.site): nothing here names a real
# site, user or storage area. Tests that check fail-closed behaviour delete the variables themselves.
SITE_PLACEHOLDERS = {
    "M2FS_DATA": "/path/to/fnal/data",
    "M2FS_CODE": "/path/to/fnal/code",
    "FASTSIMKIT_SCRATCH": "/path/to/scratch",
    "FASTSIMKIT_FNAL_HOST": "fnal-host.example",
    "FASTSIMKIT_SLURM_HOST": "slurm-host.example",
    "FASTSIMKIT_SLURM_ACCOUNT": "test_account",
    "FASTSIMKIT_EAF_SHARED_PREFIX": "/path/to/fnal/",
    "FASTSIMKIT_GEOMETRY_PREFIX": "/path/to/fnal/",
}
SITE_OPTIONAL = ("M2FS_INPUTS", "FASTSIMKIT_FNAL_DATA", "CRVFS_DATA", "FASTSIMKIT_WORKDIR", "FASTSIMKIT_CODE",
                 "FASTSIMKIT_SIF", "FASTSIMKIT_EAF_WORKDIR", "FASTSIMKIT_EAF_CODE", "FASTSIMKIT_EAF_VENV")


@pytest.fixture
def site_env(monkeypatch):
    """Set the placeholder site variables (and clear optional overrides) for one test."""
    for name in SITE_OPTIONAL:
        monkeypatch.delenv(name, raising=False)
    for name, value in SITE_PLACEHOLDERS.items():
        monkeypatch.setenv(name, value)
    return monkeypatch
