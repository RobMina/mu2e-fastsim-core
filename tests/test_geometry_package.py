"""The geometry namespace leaves optional device backends unloaded."""
import subprocess
import sys


def test_geometry_namespace_does_not_import_jax():
    subprocess.run([sys.executable, "-c", "import sys; import mu2efastsim.geometry; "
                    "assert not any(n == 'jax' or n.startswith('jax.') for n in sys.modules)"], check=True)
