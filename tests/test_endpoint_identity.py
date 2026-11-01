"""Endpoint behaviour identical to core main before the move to hep-endpoints: job scripts, headers, sbatch arguments,
records, intents, kit task records and refusals over a grid of submissions (ci/endpoint_identity.py), compared with
the rendering of core main in tests/data/endpoint_identity_main.json; the only allowed differences are the listed
KNOWN_TEXT ones."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("hependpoints")
CORE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("endpoint_identity", CORE / "ci" / "endpoint_identity.py")
EI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EI)


def test_endpoint_rendering_matches_core_main():
    pytest.importorskip("crvfastsim", reason="the grid includes crv tasks (CI puts crv on PYTHONPATH)")
    pytest.importorskip("beamfastsim", reason="the grid includes beam tasks (the umbrella's beam/)")
    golden = json.loads((CORE / "tests" / "data" / "endpoint_identity_main.json").read_text())
    fresh = EI.render()
    assert sorted(fresh) == sorted(golden)
    diffs = EI.compare(golden, fresh, known=True)
    assert diffs == [], "\n".join(diffs)[:20000]


def test_known_differences_are_only_the_snapshot_check():
    golden = json.loads((CORE / "tests" / "data" / "endpoint_identity_main.json").read_text())
    changed = sorted(k for k, case in golden.items() if EI.apply_known(case)["changed"])
    assert changed == ["streamed/beam_point", "streamed/crv_response"]
