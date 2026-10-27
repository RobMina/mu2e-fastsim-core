"""Shared fastsim test hooks (mu2efastsim.testing): JAX caches released per module; @pytest.mark.slow opt-in."""
from mu2efastsim.testing import (_release_jax_caches, pytest_addoption,  # noqa: F401
                                 pytest_collection_modifyitems, pytest_configure)
