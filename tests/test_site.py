"""mu2efastsim.site: site paths come only from the environment and fail closed."""
import pytest

from mu2efastsim import site

VARS = ("M2FS_DATA", "M2FS_INPUTS", "M2FS_CODE", "FASTSIMKIT_SCRATCH", "FASTSIMKIT_FNAL_DATA", "CRVFS_DATA")


@pytest.fixture
def clean(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def test_unset_variables_raise_and_name_the_variable(clean):
    for call, name in ((site.scratch, "FASTSIMKIT_SCRATCH"), (site.output_root, "M2FS_DATA"),
                       (site.code_root, "M2FS_CODE"), (site.data_root, "M2FS_INPUTS or M2FS_DATA")):
        with pytest.raises(site.SiteConfigError, match=name):
            call()


def test_values_and_fallbacks(clean):
    clean.setenv("M2FS_DATA", "/data/area")
    assert site.data_root() == "/data/area"
    assert site.crv_data() == "/data/area/crv_fastsim"
    clean.setenv("M2FS_INPUTS", "/inputs")
    assert site.crv_data() == "/inputs/crv_fastsim"
    clean.setenv("CRVFS_DATA", "/crv")
    assert site.crv_data() == "/crv"
    clean.setenv("FASTSIMKIT_FNAL_DATA", "/fnal")
    assert site.crv_data() == "/fnal"
    clean.setenv("FASTSIMKIT_SCRATCH", "/s")
    assert site.scratch() == "/s"
    assert site.setting("UNSET_FOR_TEST", "d") == "d"


def test_endpoint_settings_fail_closed(monkeypatch):
    names = {site.fnal_host: "FASTSIMKIT_FNAL_HOST", site.slurm_host: "FASTSIMKIT_SLURM_HOST",
             site.slurm_account: "FASTSIMKIT_SLURM_ACCOUNT", site.eaf_shared_prefix: "FASTSIMKIT_EAF_SHARED_PREFIX"}
    for call, name in names.items():
        monkeypatch.delenv(name, raising=False)
        with pytest.raises(site.SiteConfigError, match=name):
            call()
        monkeypatch.setenv(name, "value")
        assert call() == "value"
    monkeypatch.delenv("FASTSIMKIT_GEOMETRY_PREFIX", raising=False)
    assert site.geometry_prefix() is None
