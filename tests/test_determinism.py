"""GPU determinism setting (mu2efastsim.determinism): importing the package appends the XLA flag unless the setting is
off or the user set the flag; each case runs in a subprocess because the setting acts at import."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ("import json, os, mu2efastsim\nfrom mu2efastsim import determinism as d\n"
         "print(json.dumps({'flags': os.environ.get('XLA_FLAGS'), 'state': d.state()}))")


def run(code=PROBE, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "MU2EFS_DEVICE_DETERMINISTIC")}
    e.update(PYTHONPATH=str(ROOT), **env)
    r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300)
    return r


def probe(**env):
    r = run(**env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_default_appends_flag():
    out = probe()
    assert out["flags"] == "--xla_gpu_autotune_level=0"
    assert out["state"] == {"requested": True, "flag_source": "setting", "late": False,
                            "xla_gpu_autotune_level": "0"}


def test_existing_flags_kept():
    out = probe(XLA_FLAGS="--xla_gpu_enable_triton_gemm=true  --xla_dump_to=x")
    assert out["flags"] == "--xla_gpu_enable_triton_gemm=true  --xla_dump_to=x --xla_gpu_autotune_level=0"


def test_user_value_of_the_flag_wins():
    out = probe(XLA_FLAGS="--xla_gpu_autotune_level=4")
    assert out["flags"] == "--xla_gpu_autotune_level=4"
    assert out["state"]["flag_source"] == "user" and out["state"]["xla_gpu_autotune_level"] == "4"


@pytest.mark.parametrize("off", ["0", "false", "off"])
def test_setting_off_leaves_flags(off):
    out = probe(MU2EFS_DEVICE_DETERMINISTIC=off, XLA_FLAGS="--a=1")
    assert out["flags"] == "--a=1"
    assert out["state"]["requested"] is False and out["state"]["flag_source"] == "none"
    assert out["state"]["xla_gpu_autotune_level"] is None


def test_bad_value_refused():
    r = run(MU2EFS_DEVICE_DETERMINISTIC="maybe")
    assert r.returncode != 0 and "MU2EFS_DEVICE_DETERMINISTIC" in r.stderr


def test_late_application_is_recorded_and_provenance_carries_it():
    pytest.importorskip("jax")
    code = ("import json, jax\njax.config.update('jax_platforms', 'cpu')\njax.devices()\n"
            "import mu2efastsim\nfrom mu2efastsim import precision\n"
            "print(json.dumps(precision.provenance()['deterministic']))")
    r = run(code)
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout.strip().splitlines()[-1])
    assert s["late"] is True and s["flag_source"] == "setting" and s["xla_gpu_autotune_level"] == "0"
