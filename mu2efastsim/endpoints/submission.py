"""Durable local submission intent and locking: hependpoints.submission under its old name.

The module object itself is shared (not a copy), so code and tests that patch its functions patch the ones in use.
"""
import sys

from hependpoints import submission as _impl

sys.modules[__name__] = _impl
