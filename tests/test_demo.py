"""mu2efastsim.demo on synthetic inputs: a fake runner writes what each entry point writes, the demo reads the headline,
compares it with its reference and writes the summary; without the release it skips with a message and exit 2."""
import json
from pathlib import Path

import pytest

from mu2efastsim import demo


@pytest.fixture
def release(tmp_path, monkeypatch):
    inputs = tmp_path / "release"
    for f in [demo.BEAM_TABLE, demo.CRV_GEN, *demo.CRV_FILES]:
        (inputs / f).parent.mkdir(parents=True, exist_ok=True)
        (inputs / f).write_bytes(b"x")
    off, maps = tmp_path / "offline", tmp_path / "maps"
    (off / "Offline/Mu2eG4/geom").mkdir(parents=True)
    (off / "Offline/Mu2eG4/geom/geom_common.txt").write_text("")
    maps.mkdir()
    for m in ("DSMap.header", "TSdMap.header"):
        (maps / m).write_text("")
    monkeypatch.setenv("M2FS_INPUTS", str(inputs))
    monkeypatch.setenv("M2FS_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("BEAMFS_OFFLINE_ROOT", str(off))
    monkeypatch.setenv("BEAMFS_MAPS", str(maps))
    return tmp_path


def fake(shift=0.0, fail=()):
    """Runner writing synthetic outputs at the reference values (shifted by `shift` reference-sized units)."""
    calls = []

    def run(cmd, log):
        calls.append(cmd)
        chain = "beam" if "point" in cmd else "ce" if "beamfastsim.cechain" in cmd else "crv"
        log.write_text("synthetic")
        if chain in fail:
            return 6, "", 0.1
        ref = demo.REFERENCES[chain].value * (1 + shift)
        n = int(cmd[cmd.index("--n" if chain != "crv" else "--max-events") + 1])
        if chain == "beam":
            d = Path(cmd[cmd.index("--out") + 1]); d.mkdir(parents=True)
            (d / "metrics.json").write_text(json.dumps({"stop_frac": ref, "stop_frac_err": 0.003, "muons": n}))
            return 0, "", 1.0
        if chain == "ce":
            return 0, "log line\n" + json.dumps({"n": n, "fates": {"plane": round(ref * n), "stopped": 1}}), 1.0
        d = Path(cmd[cmd.index("--outdir") + 1]); d.mkdir(parents=True)
        (d / "runchain_stats.json").write_text(json.dumps({"generated_events": n,
                                                           "clusters_per_generated": {"EX": ref * 0.6, "T1": ref * 0.4}}))
        return 0, "", 1.0
    run.calls = calls
    return run


def test_references_are_filled():
    for name, ref in demo.REFERENCES.items():
        assert ref.source, name
        assert ref.value > 0 and ref.error > 0, name


def test_all_chains_pass_at_the_reference(release, capsys):
    run = fake()
    assert demo.main([], runner=run) == 0
    out = capsys.readouterr().out
    assert out.count(" ok ") == 3
    s = json.loads((release / "data/demo/demo_summary.json").read_text())
    assert s["passed"] and [r["chain"] for r in s["chains"]] == ["beam", "ce", "crv"]
    assert {int(c[c.index("--n") + 1]) for c in run.calls if "--n" in c} == {20000, 2000}


def test_quick_uses_a_quarter(release):
    run = fake()
    assert demo.main(["--quick", "--only", "ce,crv"], runner=run) == 0
    assert run.calls[0][run.calls[0].index("--n") + 1] == "500"
    assert run.calls[1][run.calls[1].index("--max-events") + 1] == "250"


def test_outside_tolerance_and_failed_chain(release, capsys):
    assert demo.main(["--only", "beam"], runner=fake(shift=0.5)) == 1
    assert demo.main(["--only", "crv"], runner=fake(shift=0.5)) == 1
    assert "OUTSIDE" in capsys.readouterr().out
    assert demo.main(["--only", "ce"], runner=fake(fail=("ce",))) == 3


def test_ce_headline_ignores_other_output():
    out = 'jax: {"not": "it"}\n' + json.dumps({"n": 10, "fates": {"plane": 7}}, indent=1) + "\n"
    v, e, n = demo.headline("ce", Path("/unused"), out)
    assert (v, n) == (0.7, 10) and e > 0
    with pytest.raises(ValueError):
        demo.headline("ce", Path("/unused"), "no json here")


def test_only_checks_the_selected_chains_files(release, capsys):
    (release / "release" / demo.CRV_GEN).unlink()
    assert demo.main(["--only", "beam"], runner=fake()) == 0
    assert demo.main(["--only", "crv"], runner=fake()) == 2
    assert "demo skipped: missing" in capsys.readouterr().out


def test_skips_without_release(monkeypatch, capsys):
    monkeypatch.delenv("M2FS_INPUTS", raising=False)
    assert demo.main([], runner=fake()) == 2
    assert "demo skipped: M2FS_INPUTS is not set" in capsys.readouterr().out


def test_skips_when_a_file_is_missing(release, capsys):
    (release / "release" / demo.BEAM_TABLE).unlink()
    assert demo.main([], runner=fake()) == 2
    assert "demo skipped: missing" in capsys.readouterr().out


def test_commands_use_existing_entry_points(release):
    c = demo.commands("/r", Path("/o"), demo.SAMPLES)
    assert c["beam"][1:4] == ["-m", "beamfastsim", "point"]
    assert c["ce"][1:3] == ["-m", "beamfastsim.cechain"] and "--stops" in c["ce"]
    assert c["crv"][1:4] == ["-m", "crvfastsim", "runchain"] and "--max-events" in c["crv"]
