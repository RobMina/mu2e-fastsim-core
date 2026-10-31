"""Site configuration: every site, user and storage path comes from the environment, never from the package.

Variables (documented with examples in the umbrella repository's site.env.example):
  M2FS_DATA            writable data/output area on the FNAL side
  M2FS_INPUTS          read-only input area (default: M2FS_DATA)
  M2FS_CODE            code checkout visible to FNAL-side jobs (EAF, GPVM)
  FASTSIMKIT_SCRATCH   working area on the Slurm (UVA) cluster
  FASTSIMKIT_FNAL_DATA / CRVFS_DATA   CRV input tables (default: <M2FS_INPUTS>/crv_fastsim)
  FASTSIMKIT_FNAL_HOST     FNAL interactive host that serves the Mu2e-internal inputs over ssh
  FASTSIMKIT_SLURM_HOST    ssh host of the Slurm cluster;  FASTSIMKIT_SLURM_ACCOUNT  its Slurm account
  FASTSIMKIT_EAF_SHARED_PREFIX   storage prefix mounted by both EAF and the FNAL host (EAF work areas live below it)
  FASTSIMKIT_GEOMETRY_PREFIX     storage prefix under which an absolute beam geometry path may point
A missing variable raises SiteConfigError naming it; nothing falls back to a user name or a built-in path.
"""
import os


class SiteConfigError(RuntimeError):
    """A site variable needed by this call is not set."""


def setting(name, default=None):
    """Value of a site variable, or default when it is unset or empty."""
    return os.environ.get(name) or default


def require(name, *alternatives):
    """Value of the first set variable among name and alternatives; SiteConfigError when none is set."""
    for n in (name, *alternatives):
        value = os.environ.get(n)
        if value:
            return value
    names = " or ".join((name, *alternatives))
    raise SiteConfigError(f"set {names} (see site.env.example in the mu2e-fastsim repository)")


def output_root():
    return require("M2FS_DATA")


def data_root():
    return require("M2FS_INPUTS", "M2FS_DATA")


def scratch():
    return require("FASTSIMKIT_SCRATCH")


def crv_data():
    return setting("FASTSIMKIT_FNAL_DATA") or setting("CRVFS_DATA") or data_root() + "/crv_fastsim"


def code_root():
    return require("M2FS_CODE")


def fnal_host():
    return require("FASTSIMKIT_FNAL_HOST")


def slurm_host():
    return require("FASTSIMKIT_SLURM_HOST")


def slurm_account():
    return require("FASTSIMKIT_SLURM_ACCOUNT")


def eaf_shared_prefix():
    return require("FASTSIMKIT_EAF_SHARED_PREFIX")


def geometry_prefix():
    """Prefix for absolute geometry paths, or None when unset (absolute paths are then refused)."""
    return setting("FASTSIMKIT_GEOMETRY_PREFIX")
