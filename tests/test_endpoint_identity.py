"""Endpoint behaviour identical to core main before the move to hep-endpoints: job scripts, headers, sbatch arguments,
records, intents, kit task records and refusals over a grid of submissions (ci/endpoint_identity.py), rendered
in-process and compared per case with the SHA-256 digests of core main's rendering
(tests/data/endpoint_identity_main.sha256.json); the only allowed differences are the listed KNOWN_TEXT ones. On a
mismatch, render both versions with the script and diff them with --compare --known."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("hependpoints")
CORE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("endpoint_identity", CORE / "ci" / "endpoint_identity.py")
EI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EI)
MAIN = json.loads((CORE / "tests" / "data" / "endpoint_identity_main.sha256.json").read_text())


def test_endpoint_rendering_matches_core_main():
    pytest.importorskip("crvfastsim", reason="the grid includes crv tasks (CI puts crv on PYTHONPATH)")
    pytest.importorskip("beamfastsim", reason="the grid includes beam tasks (the umbrella's beam/)")
    fresh = EI.render()
    assert sorted(fresh) == sorted(MAIN["cases"])
    ours = {k: EI.case_digest(EI.masked(v) if k in MAIN["changed"] else v) for k, v in fresh.items()}
    assert {k for k in ours if ours[k] != MAIN["cases"][k]} == set()


def test_known_differences_are_only_the_snapshot_check():
    assert MAIN["changed"] == ["streamed/beam_point", "streamed/crv_response"]
