"""Portable input manifests use explicit roots and never silently repair corrupt files."""
import hashlib
import json
from pathlib import Path

import pytest

from mu2efastsim import inputs, site


def record(path="nested/input.bin", redistribution="approved"):
    return dict(path=path, size_bytes=3, sha256=hashlib.sha256(b"abc").hexdigest(),
                provenance="synthetic fixture", redistribution=redistribution)


def obj(row=None):
    return dict(schema_version=1, release="test", files=[row or record()])


def source(tmp_path):
    root = tmp_path / "source"
    (root / "nested").mkdir(parents=True)
    (root / "nested/input.bin").write_bytes(b"abc")
    return root


def test_inventory(tmp_path):
    src = source(tmp_path) / "nested/input.bin"
    spec = dict(release="fixture", files=[dict(path="out.bin", source=str(src), provenance="fixture", redistribution="pending")])
    out = inputs.inventory(spec)
    assert out["files"][0]["size_bytes"] == 3
    assert out["files"][0]["sha256"] == record()["sha256"]
    assert "source" not in out["files"][0]


def test_check_and_fetch(tmp_path):
    src = source(tmp_path)
    dst = tmp_path / "dest"
    assert inputs.fetch(obj(), dst, source_dir=src)[0]["status"] == "would-fetch"
    assert not dst.exists()
    assert inputs.fetch(obj(), dst, source_dir=src, execute=True)[0]["status"] == "fetched"
    assert inputs.check(obj(), dst)[0]["status"] == "ok"
    assert inputs.fetch(obj(), dst, source_dir=src, execute=True)[0]["status"] == "ok"
    (dst / "nested/input.bin").write_bytes(b"abd")
    assert inputs.check(obj(), dst)[0]["status"] == "hash-mismatch"
    with pytest.raises(ValueError, match="overwrite"):
        inputs.fetch(obj(), dst, source_dir=src, execute=True)


@pytest.mark.parametrize("data", [b"bad", b"a", b"toolong"])
def test_failed_download_is_atomic(tmp_path, data):
    src = source(tmp_path)
    (src / "nested/input.bin").write_bytes(data)
    dst = tmp_path / "dest"
    with pytest.raises(ValueError, match="mismatch|oversize"):
        inputs.fetch(obj(), dst, source_dir=src, execute=True)
    assert not (dst / "nested/input.bin").exists()
    assert not list(dst.rglob(".input-*"))


def test_unapproved(tmp_path):
    src = source(tmp_path)
    with pytest.raises(ValueError, match="not approved"):
        inputs.fetch(obj(record(redistribution="pending")), tmp_path / "dest", source_dir=src, execute=True)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../../x", "a//b", "a/./b", "", "a\\b"])
def test_unsafe_path(path, tmp_path):
    with pytest.raises(ValueError):
        inputs.under(tmp_path, path)


def test_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "nested").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        inputs.check(obj(), root)


def test_dangling_symlink_not_replaced(tmp_path):
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    (root / "nested/input.bin").symlink_to(root / "missing")
    with pytest.raises(ValueError, match="existing"):
        inputs.fetch(obj(), root, source_dir=source(tmp_path), execute=True)


@pytest.mark.parametrize("url", ["http://example.org", "file:///tmp", "https://u:p@example.org", "https://example.org?q=1"])
def test_bad_url(tmp_path, url):
    with pytest.raises(ValueError, match="HTTPS"):
        inputs.fetch(obj(), tmp_path, base_url=url)


def test_manifest_validation_and_cli(tmp_path, capsys):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(obj()))
    assert inputs.manifest(p)["release"] == "test"
    assert inputs.main(["check-inputs", "--manifest", str(p), "--root", str(tmp_path)]) == 1
    assert "missing" in capsys.readouterr().out
    p.write_text(json.dumps(obj(dict(record(), sha256="bad"))))
    assert inputs.main(["check-inputs", "--manifest", str(p), "--root", str(tmp_path)]) == 2
    p.write_text(json.dumps(dict(obj(), files=[record(), record()])))
    with pytest.raises(ValueError, match="duplicate"):
        inputs.manifest(p)


def test_site_explicit_overrides(monkeypatch):
    monkeypatch.setenv("M2FS_INPUTS", "/group/release")
    monkeypatch.setenv("M2FS_DATA", "/my/output")
    monkeypatch.setenv("FASTSIMKIT_SCRATCH", "/my/scratch")
    monkeypatch.delenv("CRVFS_DATA", raising=False)
    monkeypatch.delenv("FASTSIMKIT_FNAL_DATA", raising=False)
    assert site.crv_data() == "/group/release/crv_fastsim"
    assert site.scratch() == "/my/scratch"
    monkeypatch.setenv("FASTSIMKIT_FNAL_DATA", "/other/input")
    assert site.crv_data() == "/other/input"


def test_default_sites_fail_closed_without_configuration(tmp_path, monkeypatch):
    from mu2efastsim.endpoints.chainpoint import FnalSource
    from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint
    from mu2efastsim.endpoints.eaf import EAFEndpoint
    for key in ("M2FS_INPUTS", "M2FS_DATA", "M2FS_CODE", "FASTSIMKIT_SCRATCH", "FASTSIMKIT_FNAL_DATA", "CRVFS_DATA",
                "FASTSIMKIT_WORKDIR", "FASTSIMKIT_EAF_WORKDIR", "FASTSIMKIT_EAF_SHARED_PREFIX", "FASTSIMKIT_FNAL_HOST",
                "FASTSIMKIT_SLURM_HOST", "FASTSIMKIT_SLURM_ACCOUNT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(site.SiteConfigError, match="M2FS_INPUTS or M2FS_DATA"):
        FnalSource()
    with pytest.raises(site.SiteConfigError, match="FASTSIMKIT_SCRATCH"):
        SlurmSSHEndpoint(state_dir=tmp_path / "slurm")
    with pytest.raises(site.SiteConfigError, match="M2FS_DATA"):
        EAFEndpoint(state_dir=tmp_path / "eaf")
    ep = SlurmSSHEndpoint(state_dir=tmp_path / "slurm", workdir="/path/to/scratch/kit")
    for attr, name in (("host", "FASTSIMKIT_SLURM_HOST"), ("account", "FASTSIMKIT_SLURM_ACCOUNT")):
        with pytest.raises(site.SiteConfigError, match=name):
            getattr(ep, attr)
    with pytest.raises(site.SiteConfigError, match="FASTSIMKIT_FNAL_HOST"):
        FnalSource(root="/path/to/fnal/crv_fastsim").host
    monkeypatch.setenv("M2FS_DATA", "/path/to/fnal/data")
    with pytest.raises(site.SiteConfigError, match="FASTSIMKIT_EAF_SHARED_PREFIX"):
        EAFEndpoint(state_dir=tmp_path / "eaf")
    monkeypatch.setenv("FASTSIMKIT_EAF_SHARED_PREFIX", "/path/to/fnal/")
    eaf = EAFEndpoint(state_dir=tmp_path / "eaf")
    assert eaf.workdir == "/path/to/fnal/data/crv_fastsim/eafkit"
    assert eaf.fnal_data == "/path/to/fnal/data/crv_fastsim"
    with pytest.raises(site.SiteConfigError, match="M2FS_CODE"):
        eaf.code


@pytest.mark.parametrize("value", [[], None, {"schema_version": 1, "files": [None]}])
def test_malformed_json_object(value):
    with pytest.raises(ValueError):
        inputs.validate(value)


def test_eaf_output_does_not_default_to_read_only_release(tmp_path, monkeypatch):
    from mu2efastsim.endpoints.eaf import EAFEndpoint
    monkeypatch.setenv("M2FS_INPUTS", "/readonly/release")
    monkeypatch.setenv("M2FS_DATA", "/path/to/fnal/output")
    monkeypatch.setenv("FASTSIMKIT_EAF_SHARED_PREFIX", "/path/to/fnal/")
    monkeypatch.delenv("FASTSIMKIT_EAF_VENV", raising=False)
    monkeypatch.delenv("FASTSIMKIT_EAF_WORKDIR", raising=False)
    ep = EAFEndpoint(state_dir=tmp_path)
    assert ep.venv.startswith("/path/to/fnal/output/")
    assert ep.workdir.startswith("/path/to/fnal/output/")


def test_endpoint_hosts_and_account_can_be_retargeted(tmp_path, monkeypatch):
    from mu2efastsim.endpoints import Resources
    from mu2efastsim.endpoints.chainpoint import FnalSource
    from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint
    monkeypatch.setenv("FASTSIMKIT_SLURM_HOST", "site-host")
    monkeypatch.setenv("FASTSIMKIT_SLURM_ACCOUNT", "site-account")
    monkeypatch.setenv("FASTSIMKIT_FNAL_HOST", "site-fnal")
    ep = SlurmSSHEndpoint(state_dir=tmp_path, workdir="/path/to/scratch/kit")
    assert (ep.host, ep.account) == ("site-host", "site-account")
    ep.host, ep.account = "other-host", "other-account"
    assert (ep.host, ep.account) == ("other-host", "other-account")
    assert "#SBATCH -A other-account" in ep.header("h", Resources())
    src = FnalSource(root="/path/to/fnal/crv_fastsim")
    assert src.host == "site-fnal"
    src.host = "other-fnal"
    assert src.host == "other-fnal" and src.argv("x")[3] == "other-fnal"

