"""Job environments (the project image, the EAF venv, GPVM pyenv ana) do not have hep-endpoints: every core module
outside mu2efastsim.endpoints, in particular mu2efastsim.site, must import without it, and every module of the
detector packages on PYTHONPATH (crv, the umbrella's beam, tracker) except the kit-host streaming delegates
(``kit_uva``) must import without mu2efastsim.endpoints."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parent.parent

PROBE = r"""
import importlib, os, pkgutil, sys
sys.modules["hependpoints"] = None                 # import hependpoints raises ImportError
import mu2efastsim
import mu2efastsim.testing                         # the test hooks must come before mu2efastsim.precision
from mu2efastsim import site
assert issubclass(site.SiteConfigError, RuntimeError) and site.SiteConfigError.__module__ == "mu2efastsim.site"
os.environ.pop("M2FS_DATA", None)
try:
    site.output_root()
except site.SiteConfigError as exc:
    assert "M2FS_DATA" in str(exc)
else:
    raise AssertionError("site.output_root() without M2FS_DATA")
import mu2efastsim.inputs
blocked, other = [], []
for m in pkgutil.walk_packages(mu2efastsim.__path__, "mu2efastsim."):
    if m.name == "mu2efastsim.endpoints" or m.name.startswith("mu2efastsim.endpoints."):
        continue
    try:
        importlib.import_module(m.name)
    except ImportError as exc:
        (blocked if "hependpoints" in str(exc) else other).append(f"{m.name}: {exc}")
sys.modules["mu2efastsim.endpoints"] = None       # detector packages: job-side code never needs the endpoints
for pkg in ("crvfastsim", "beamfastsim", "trkfastsim"):
    try:
        mod = importlib.import_module(pkg)
    except ImportError:
        continue
    for m in pkgutil.walk_packages(mod.__path__, pkg + "."):
        if m.name.rsplit(".", 1)[-1] in ("kit_uva", "__main__"):
            continue
        try:
            importlib.import_module(m.name)
        except ImportError as exc:
            (blocked if "endpoints" in str(exc) else other).append(f"{m.name}: {exc}")
print("BLOCKED", blocked)
print("OTHER", other)
"""


@pytest.mark.slow
def test_core_modules_import_without_hependpoints():
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, cwd=CORE,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": os.pathsep.join([str(CORE), os.environ.get("PYTHONPATH", "")]),
                            "JAX_PLATFORMS": "cpu"})
    assert r.returncode == 0, r.stderr[-3000:]
    assert "BLOCKED []" in r.stdout, r.stdout


def test_site_shares_the_error_class_when_hependpoints_is_importable():
    import pytest
    hep_site = pytest.importorskip("hependpoints.site")
    from mu2efastsim import site
    assert site.SiteConfigError is hep_site.SiteConfigError
