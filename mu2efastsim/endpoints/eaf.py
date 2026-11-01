"""Fermilab EAF for the fast chain: hependpoints.eaf with the Mu2e settings and the fast-chain runtime.

Configuration: ``token_file`` (default $EAF_TOKEN_FILE), ``hub`` (default https://analytics-hub.fnal.gov), ``workdir``
(default $FASTSIMKIT_EAF_WORKDIR, else <M2FS_DATA>/crv_fastsim/eafkit), ``shared_prefix`` (default
$FASTSIMKIT_EAF_SHARED_PREFIX), ``fnal_data`` (default mu2efastsim.site.crv_data()), ``code`` and ``venv`` (default
$FASTSIMKIT_EAF_CODE or $M2FS_CODE, and $FASTSIMKIT_EAF_VENV or <M2FS_DATA>/crv_fastsim/venv_eaf; needed only to run a
job), ``state_dir`` (local job records). A missing setting raises mu2efastsim.site.SiteConfigError naming it.
``hub_client`` is injectable for tests. Server, queue, linger and cancel behaviour: hependpoints.eaf.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from hependpoints.eaf import IMAGE, PROFILE, SLICE_GB, HubClient, slice_of  # noqa: F401
from hependpoints.eaf import EAFEndpoint as _EAFEndpoint
from hependpoints.eaf import main as _main

from mu2efastsim import site

from . import mu2e_config as cfg
from .mu2e_config import ANA_PY, JAX_PKG  # noqa: F401


class EAFEndpoint(_EAFEndpoint):
    SHARED_PREFIX_VARS = ("FASTSIMKIT_EAF_SHARED_PREFIX",)
    CONFIG_HINT = site.HINT
    WATCHER_MODULE = "mu2efastsim.endpoints.eaf"

    def __init__(self, workdir: str | None = None, code: str | None = None,
                 venv: str | None = None, shared_prefix: str | None = None, fnal_data: str | None = None,
                 state_dir: str | Path = cfg.EAF_STATE_DIR, token_file: Optional[str] = None,
                 hub: str = "https://analytics-hub.fnal.gov", slices: Sequence[str] = ("4g.40gb", "2g.20gb", "1g.10gb"),
                 start_timeout_s: float = 420.0, poll_s: float = 10.0, hub_client: Optional[HubClient] = None,
                 linger_s: Optional[float] = None, max_age_s: Optional[float] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 name: str = "eaf", **_ignored):
        workdir = (workdir or site.setting("FASTSIMKIT_EAF_WORKDIR")
                   or site.output_root() + "/crv_fastsim/eafkit").rstrip("/")
        self._code, self._venv = code, venv          # resolved when a job runs (runtime)
        self.fnal_data = (fnal_data or site.crv_data()).rstrip("/")     # Mu2e-internal inputs read in place (kit FNAL_TASKS)
        shared_prefix = shared_prefix or site.eaf_shared_prefix()
        super().__init__(workdir=workdir, shared_prefix=shared_prefix, state_dir=state_dir,
                         token_file=token_file or os.environ.get("EAF_TOKEN_FILE", ""), hub=hub, slices=slices,
                         start_timeout_s=start_timeout_s, poll_s=poll_s, hub_client=hub_client, linger_s=linger_s,
                         max_age_s=max_age_s, clock=clock, sleep=sleep, name=name, environment=cfg.ENVIRONMENT)

    @property
    def code(self) -> str:
        return (self._code or site.setting("FASTSIMKIT_EAF_CODE") or site.code_root()).rstrip("/")

    @property
    def venv(self) -> str:
        venv = self._venv or site.setting("FASTSIMKIT_EAF_VENV") or site.output_root() + "/crv_fastsim/venv_eaf"
        return venv.rstrip("/")

    def runtime(self) -> Dict[str, str]:
        """Shell definitions of A and T for the kit's task bodies (mu2e_config.eaf_runtime): the venv on shared storage
        instead of an Apptainer image, which EAF cannot start."""
        return cfg.eaf_runtime(self.venv, self.code)


def main(argv=None) -> int:
    """python -m mu2efastsim.endpoints.eaf {server|jobs|stop-idle} [--state-dir D] (needs EAF_TOKEN_FILE)."""
    return _main(argv, endpoint_cls=EAFEndpoint,
                 default_state_dir=os.environ.get("FASTSIMKIT_STATE_DIR", "~/.mu2efastsim/kit") + "/jobs")


if __name__ == "__main__":
    raise SystemExit(main())
